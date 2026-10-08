"""Wheel image upload and inspection API.

Uploaded image files are persisted in Supabase Storage. Model inference still
uses a temporary local copy when the requested image is stored remotely; that
copy is removed after inference finishes.
"""

import io
import json
import logging
import os
import tempfile
import uuid

import httpx
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

try:
    from dotenv import load_dotenv
except ImportError:  # The API can start, but .env files will not be loaded.
    load_dotenv = None

from app.schemas.quality import (
    HybridRimResult,
    HybridLocalizedDefect,
    HybridLocalizationInfo,
    HybridClassificationInfo,
    RimCNNResult,
    WheelAIOutputContract,
    WheelInspectionRequest,
)
from app.services.hybrid_inspection import hybrid_inspector
from app.services.inference import run_rim_cnn_inference_async
from app.services.tyre_yolo_detector import tyre_yolo_detector

logger = logging.getLogger(__name__)

router = APIRouter()

# This file is backend/app/api/inspection.py, so parents[3] is the repository
# root. The user's project keeps .env in that root directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(BACKEND_ROOT / ".env", override=False)

_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
_CONTENT_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _bucket_name() -> str:
    return os.getenv("SUPABASE_BUCKET", "wheel-images").strip() or "wheel-images"


def _get_supabase_credentials() -> tuple[str, str]:
    """Return the project URL and a server-side Supabase API key.

    Prefer modern secret keys or the legacy service-role key. SUPABASE_KEY is
    accepted for backwards compatibility, but must contain a privileged server
    key, not a publishable/anon key, for private Storage and database writes.
    """
    supabase_url = (os.getenv("SUPABASE_URL") or "").strip()
    supabase_key = (
            os.getenv("SUPABASE_SECRET_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_KEY")
            or ""
    ).strip()

    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Supabase is not configured. Set SUPABASE_URL and either "
                "SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY in the repository-root .env file."
            ),
        )

    if supabase_key.startswith("sb_publishable_"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The backend loaded a publishable Supabase key. Set SUPABASE_SECRET_KEY "
                "or SUPABASE_SERVICE_ROLE_KEY to a server-side key; do not use a publishable key "
                "for private image downloads or privileged database writes."
            ),
        )

    return supabase_url.rstrip("/"), supabase_key


def _supabase_request_headers(api_key: str) -> dict[str, str]:
    """Build headers for modern Supabase API keys and legacy JWT API keys.

    Modern sb_secret_ keys must be sent as `apikey`, not as a Bearer JWT.
    Legacy anon/service_role keys are JWTs and are also sent in Authorization.
    """
    headers = {"apikey": api_key}
    if api_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _normalize_optional_uuid(value, field_name: str) -> str | None:
    """Return a canonical UUID for the predictions FK fields, or omit it.

    The predictions schema declares machine_id and batch_id as UUID foreign keys.
    Older clients may send non-UUID identifiers; omit those instead of causing an
    invalid UUID cast. A valid UUID still must reference an existing row.
    """
    if value is None or str(value).strip() == "":
        return None
    try:
        return str(uuid.UUID(str(value)))
    except (ValueError, TypeError, AttributeError):
        logger.warning(
            "Omitting %s from predictions row because it is not a valid UUID: %r",
            field_name, value,
        )
        return None


async def _save_prediction_record(
        *,
        component: str,
        image_path: str,
        wheel_id: str | None,
        machine_id: str | None,
        batch_id: str | None,
        defect_type: str | None,
        probability,
        model_version: str | None,
        location=None,
        details=None,
) -> dict | None:
    """Persist the completed model prediction to public.predictions.

    The image bytes remain in Supabase Storage. The Storage object path is written
    to predictions.image_path; wheel_id and detailed inference metadata are stored
    as JSON text in predictions.explanation.
    """
    prediction_type = (str(defect_type).strip() if defect_type is not None else "")
    if not prediction_type:
        prediction_type = f"{component}_unknown"
    prediction_type = prediction_type[:100]

    try:
        numeric_probability = float(probability) if probability is not None else None
    except (TypeError, ValueError):
        numeric_probability = None

    # The predictions schema contains a dedicated image_path column. Store the
    # Storage object path there (not the binary image and not an expiring URL).
    # Legacy local-file inspections remain supported, but have no Storage path.
    storage_image_path = None
    if _local_image_candidate(image_path) is None:
        storage_image_path = _storage_object_path(image_path)

    explanation_payload = {
        "component": component,
        "wheel_id": str(wheel_id) if wheel_id is not None else None,
        "image_path": storage_image_path,
        "requested_image_reference": image_path,
        "location": location,
        "details": details,
    }
    record = {
        "target_domain": component,
        "image_path": storage_image_path,
        "prediction_type": prediction_type,
        "probability": numeric_probability,
        "model_version": (str(model_version)[:100] if model_version else None),
        "explanation": json.dumps(explanation_payload, ensure_ascii=False, default=str),
    }

    normalized_machine_id = _normalize_optional_uuid(machine_id, "machine_id")
    normalized_batch_id = _normalize_optional_uuid(batch_id, "batch_id")
    if normalized_machine_id is not None:
        record["machine_id"] = normalized_machine_id
    if normalized_batch_id is not None:
        record["batch_id"] = normalized_batch_id

    try:
        supabase_url, supabase_key = _get_supabase_credentials()
        headers = {
            **_supabase_request_headers(supabase_key),
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
        async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as http_client:
            response = await http_client.post(
                f"{supabase_url}/rest/v1/predictions?select=*",
                headers=headers,
                json=record,
            )

        if response.is_error:
            logger.error(
                "PostgREST prediction insert failed: status=%s response=%s",
                response.status_code,
                response.text[:1200],
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "AI inspection completed, but saving to public.predictions failed. "
                    "Verify that predictions has image_path and target_domain columns, "
                    "the backend key is privileged, and supplied machine_id/batch_id UUIDs exist."
                ),
            )

        response_rows = response.json() if response.content else []
        saved_row = response_rows[0] if isinstance(response_rows, list) and response_rows else None
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to save prediction to public.predictions")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The AI inspection completed, but saving its prediction to Supabase failed. "
                "Check the backend credentials, Data API access, table schema, and foreign-key UUIDs."
            ),
        ) from exc

    logger.info(
        "Saved prediction row to public.predictions (id=%s, type=%s)",
        (saved_row or {}).get("id") if isinstance(saved_row, dict) else None,
        prediction_type,
    )
    return saved_row


def _cleanup_temporary_file(path: str | None, is_temporary: bool) -> None:
    if not is_temporary or not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        logger.warning("Could not remove temporary inspection image: %s", path, exc_info=True)


def _local_image_candidate(image_reference: str) -> Path | None:
    """Return a matching existing local file, supporting old API callers."""
    parsed = urlparse(image_reference)
    if parsed.scheme in {"http", "https"}:
        return None

    path = Path(image_reference).expanduser()
    candidates = [path] if path.is_absolute() else [
        Path.cwd() / path,
        BACKEND_ROOT / path,
        REPO_ROOT / path,
        ]
    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def _storage_object_path(image_reference: str) -> str:
    """Convert a bucket object path or Supabase Storage URL to an object path."""
    bucket = _bucket_name()
    parsed = urlparse(image_reference)

    if parsed.scheme in {"http", "https"}:
        segments = [unquote(part) for part in parsed.path.split("/") if part]
        # Examples: /storage/v1/object/public/<bucket>/<object>
        #           /storage/v1/object/sign/<bucket>/<object>
        for index, segment in enumerate(segments):
            if segment in {"public", "sign", "authenticated"} and index + 1 < len(segments):
                if segments[index + 1] == bucket:
                    object_parts = segments[index + 2 :]
                    if object_parts:
                        return "/".join(object_parts)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The supplied image URL is not a recognised URL for the configured Supabase Storage bucket.",
        )

    reference = unquote(image_reference).replace("\\", "/").strip().lstrip("/")
    bucket_prefix = f"{bucket}/"
    if reference.startswith(bucket_prefix):
        reference = reference[len(bucket_prefix) :]
    if not reference:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="image_path must contain a Supabase Storage object path.",
        )
    return reference


async def _get_local_image_for_inspection(image_reference: str) -> tuple[str, bool]:
    """Resolve local paths directly or download a Storage object to a temp file."""
    local_candidate = _local_image_candidate(image_reference)
    if local_candidate is not None:
        return str(local_candidate), False

    object_path = _storage_object_path(image_reference)
    try:
        supabase_url, supabase_key = _get_supabase_credentials()
        # Private Storage downloads use the authenticated-object route. URL-encode
        # each path segment but preserve '/' folder separators.
        encoded_bucket = quote(_bucket_name(), safe="")
        encoded_object_path = quote(object_path, safe="/")
        download_url = (
            f"{supabase_url}/storage/v1/object/authenticated/"
            f"{encoded_bucket}/{encoded_object_path}"
        )
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http_client:
            response = await http_client.get(
                download_url,
                headers=_supabase_request_headers(supabase_key),
            )

        if response.is_error:
            logger.error(
                "Supabase Storage download failed: status=%s bucket=%s path=%s response=%s",
                response.status_code,
                _bucket_name(),
                object_path,
                response.text[:1000],
            )
            if response.status_code in (401, 403):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        "Supabase denied the image download. Set SUPABASE_SECRET_KEY or "
                        "SUPABASE_SERVICE_ROLE_KEY to a valid server-side key in the repository-root .env, "
                        "then restart FastAPI. Do not use a publishable/anon key for this private bucket."
                    ),
                )
            if response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=(
                        f"Image not found in bucket '{_bucket_name()}' at '{object_path}'. "
                        "Use the exact image_path returned by POST /api/inspection/upload."
                    ),
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Supabase Storage download failed with HTTP {response.status_code}.",
            )

        image_bytes = response.content
        if not image_bytes:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="The downloaded Supabase Storage image is empty.",
            )
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.exception("Network error downloading image from Supabase Storage: %s", object_path)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not connect to Supabase Storage. Check SUPABASE_URL and network connectivity.",
        ) from exc
    except Exception as exc:
        logger.exception("Failed to download image from Supabase Storage: %s", object_path)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "Could not download image from Supabase Storage. Check the object path, bucket, "
                "backend key, and Storage permissions. See the backend log for the underlying error."
            ),
        ) from exc

    suffix = Path(object_path).suffix.lower()
    if suffix not in _ALLOWED_EXTENSIONS:
        suffix = ".img"

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=suffix, delete=False) as temporary_file:
            temporary_file.write(image_bytes)
            temporary_file.flush()
            temporary_path = temporary_file.name
    except Exception:
        _cleanup_temporary_file(temporary_path, True)
        raise

    return temporary_path, True


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload an aluminium wheel image to Supabase Storage",
)
async def upload_wheel_image(image: UploadFile = File(...)):
    """Validate the uploaded image and persist it in the configured Supabase bucket."""
    filename = image.filename or "wheel-image"
    extension = Path(filename).suffix.lower()
    if extension not in _ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported image format. Upload a JPG, JPEG, PNG, or WEBP image.",
        )

    contents = await image.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image is empty.",
        )

    # Validate actual image bytes; do not trust the filename or Content-Type alone.
    try:
        with Image.open(io.BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is not a valid image.",
        ) from exc

    # Validate configuration before making the Storage request. Use the REST
    # endpoint with the image bytes in memory: this avoids creating a local copy
    # and avoids Windows file-lock errors when a failed SDK upload leaves a handle open.
    supabase_url, supabase_key = _get_supabase_credentials()
    bucket = _bucket_name()
    object_path = f"inspections/{uuid.uuid4().hex}{extension}"
    content_type = _CONTENT_TYPES[extension]
    object_url = (
        f"{supabase_url}/storage/v1/object/"
        f"{bucket}/{object_path}"
    )
    headers = {
        **_supabase_request_headers(supabase_key),
        "x-upsert": "false",
    }

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as http_client:
            response = await http_client.post(
                object_url,
                headers=headers,
                files={"file": (filename, contents, content_type)},
            )

        if response.is_error:
            response_detail = response.text[:1000]
            logger.error(
                "Supabase Storage rejected upload: status=%s response=%s",
                response.status_code,
                response_detail,
            )
            if response.status_code in (401, 403):
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=(
                        "Supabase rejected the upload (401/403). Configure a valid "
                        "SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY in the backend .env, "
                        "or review the bucket's Storage INSERT RLS policy. Keep secret keys backend-only."
                    ),
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Supabase Storage upload failed with HTTP {response.status_code}: {response_detail}",
            )
    except HTTPException:
        raise
    except httpx.HTTPError as exc:
        logger.exception("Network error while uploading %s to Supabase Storage", filename)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not connect to Supabase Storage. Check SUPABASE_URL and network connectivity.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected Supabase Storage upload failure for %s", filename)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Supabase Storage upload failed. Check credentials, bucket name, and Storage permissions.",
        ) from exc

    logger.info("Uploaded wheel image to Supabase Storage bucket=%s path=%s", bucket, object_path)
    return {
        "filename": filename,
        "image_path": object_path,
        "storage_bucket": bucket,
        "image_url": None,
        "message": "Wheel image uploaded successfully to Supabase Storage",
    }


@router.post(
    "/inspect",
    response_model=WheelAIOutputContract,
    status_code=status.HTTP_200_OK,
    summary="Inspect an aluminium alloy wheel rim or tyre component",
)
async def inspect_wheel(request: WheelInspectionRequest):
    """
    Submit an image path for visual inspection (rim or tyre).

    Components:
        - "rim": Hybrid YOLO localization + CNN classification pipeline
        - "tyre": Standalone Tyre YOLO object detection pipeline
    """
    comp = (request.component or "rim").lower().strip()

    if comp not in ("rim", "tyre"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid component '{request.component}'. Must be 'rim' or 'tyre'.",
        )

    # ── Tyre Inspection Path ──────────────────────────────────────────────────
    if comp == "tyre":
        if not tyre_yolo_detector.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Tyre YOLO detection service is not available",
            )
        local_image_path, is_temporary = await _get_local_image_for_inspection(request.image_path)
        try:
            tyre_result = await tyre_yolo_detector.detect_async(local_image_path)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Tyre inspection failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tyre inspection internal failure",
            ) from exc
        finally:
            _cleanup_temporary_file(local_image_path, is_temporary)

        top_defect = None
        top_confidence = None
        top_bbox = None
        if tyre_result.detections:
            top_det = tyre_result.detections[0]
            top_defect = top_det.class_name
            top_confidence = top_det.confidence
            top_bbox = top_det.bbox

        tyre_details = tyre_result.model_dump()
        await _save_prediction_record(
            component="tyre",
            image_path=request.image_path,
            wheel_id=request.wheel_id,
            machine_id=request.machine_id,
            batch_id=request.batch_id,
            defect_type=top_defect,
            probability=top_confidence,
            model_version=getattr(tyre_result, "model", None) or "tyre_yolo",
            location=top_bbox,
            details=tyre_details,
        )

        return WheelAIOutputContract(
            wheel_id=request.wheel_id,
            batch_id=request.batch_id,
            machine_id=request.machine_id,
            component="tyre",
            hybrid=None,
            rim=None,
            tyre=tyre_result.model_dump(),
            defect_type=top_defect,
            location=top_bbox,
            severity=None,
            defect_confidence=top_confidence,
            root_cause=None,
            root_cause_confidence=None,
            future_risk=None,
            recommended_action=None,
            affected_batches=[],
        )

    # ── Rim Inspection Path (Hybrid YOLO + CNN) ──────────────────────────────
    if hybrid_inspector.is_ready:
        local_image_path, is_temporary = await _get_local_image_for_inspection(request.image_path)
        try:
            hybrid_result = await hybrid_inspector.inspect_async(local_image_path)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
        except Exception as exc:
            logger.error("Hybrid inspection failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Hybrid inspection internal failure",
            ) from exc
        finally:
            _cleanup_temporary_file(local_image_path, is_temporary)

        # Map internal service types → API schema types
        api_localized = []
        for ld in hybrid_result.localized_defects:
            api_localized.append(
                HybridLocalizedDefect(
                    localization=HybridLocalizationInfo(
                        source=ld.localization.source,
                        yolo_class_id=ld.localization.yolo_class_id,
                        yolo_defect_type=ld.localization.yolo_defect_type,
                        yolo_confidence=ld.localization.yolo_confidence,
                        bbox=ld.localization.bbox,
                        mask_status=ld.localization.mask_status,
                        mask_area_pixels=ld.localization.mask_area_pixels,
                        mask_area_ratio=ld.localization.mask_area_ratio,
                        mask_polygon=ld.localization.mask_polygon,
                    ),
                    classification=HybridClassificationInfo(
                        source=ld.classification.source,
                        cnn_class_id=ld.classification.cnn_class_id,
                        cnn_defect_type=ld.classification.cnn_defect_type,
                        cnn_confidence=ld.classification.cnn_confidence,
                        inference_ms=ld.classification.inference_ms,
                        device=ld.classification.device,
                    ),
                    classification_agreement=ld.classification_agreement,
                )
            )

        hybrid_api = HybridRimResult(
            model="hybrid_yolo_cnn",
            localization_status=hybrid_result.localization_status,
            yolo_inference_ms=hybrid_result.yolo_inference_ms,
            cnn_total_inference_ms=hybrid_result.cnn_total_inference_ms,
            total_hybrid_ms=hybrid_result.total_hybrid_ms,
            device=hybrid_result.device,
            yolo_conf_threshold=hybrid_result.yolo_conf_threshold,
            localized_defects=api_localized,
            full_image_cnn=hybrid_result.full_image_cnn,
        )

        # Determine top-level defect_type and confidence for backward-compat fields
        # Priority: first localized CNN defect → full-image CNN fallback
        fic = hybrid_result.full_image_cnn or {}
        if api_localized:
            top = api_localized[0].classification
            top_defect = top.cnn_defect_type
            top_confidence = top.cnn_confidence
            # Location is the first YOLO bbox
            top_bbox = api_localized[0].localization.bbox
        else:
            top_defect = fic.get("defect_type")
            top_confidence = fic.get("confidence")
            top_bbox = None

        # Legacy rim field: use full-image CNN result
        rim_legacy = None
        if fic:
            rim_legacy = RimCNNResult(
                model=fic.get("model", "rim_cnn_v1"),
                defect_type=fic.get("defect_type", "unknown"),
                class_id=fic.get("class_id", 0),
                confidence=fic.get("confidence", 0.0),
                inference_ms=fic.get("inference_ms", 0.0),
                device=fic.get("device", "cpu"),
            )

        await _save_prediction_record(
            component="rim",
            image_path=request.image_path,
            wheel_id=request.wheel_id,
            machine_id=request.machine_id,
            batch_id=request.batch_id,
            defect_type=top_defect,
            probability=top_confidence,
            model_version=hybrid_api.model,
            location=top_bbox,
            details=hybrid_api.model_dump(),
        )

        return WheelAIOutputContract(
            wheel_id=request.wheel_id,
            batch_id=request.batch_id,
            machine_id=request.machine_id,
            component="rim",
            hybrid=hybrid_api,
            rim=rim_legacy,
            tyre=None,
            defect_type=top_defect,
            location=top_bbox,
            severity=None,
            defect_confidence=top_confidence,
            root_cause=None,
            root_cause_confidence=None,
            future_risk=None,
            recommended_action=None,
            affected_batches=[],
        )

    # ── Fallback: CNN-only (hybrid not ready) ──────────────────────────────────
    logger.warning("Hybrid inspector not ready — falling back to CNN-only inspection")
    local_image_path, is_temporary = await _get_local_image_for_inspection(request.image_path)
    try:
        cnn_pred = await run_rim_cnn_inference_async(local_image_path)
    finally:
        _cleanup_temporary_file(local_image_path, is_temporary)

    rim_result = RimCNNResult(
        model=cnn_pred.model,
        defect_type=cnn_pred.defect_type,
        class_id=cnn_pred.class_id,
        confidence=cnn_pred.confidence,
        inference_ms=cnn_pred.inference_ms,
        device=cnn_pred.device,
    )

    await _save_prediction_record(
        component="rim",
        image_path=request.image_path,
        wheel_id=request.wheel_id,
        machine_id=request.machine_id,
        batch_id=request.batch_id,
        defect_type=cnn_pred.defect_type,
        probability=cnn_pred.confidence,
        model_version=cnn_pred.model,
        location=None,
        details={
            "class_id": cnn_pred.class_id,
            "inference_ms": cnn_pred.inference_ms,
            "device": cnn_pred.device,
        },
    )

    return WheelAIOutputContract(
        wheel_id=request.wheel_id,
        batch_id=request.batch_id,
        machine_id=request.machine_id,
        component="rim",
        hybrid=None,
        rim=rim_result,
        tyre=None,
        defect_type=cnn_pred.defect_type,
        location=None,
        severity=None,
        defect_confidence=cnn_pred.confidence,
        root_cause=None,
        root_cause_confidence=None,
        future_risk=None,
        recommended_action=None,
        affected_batches=[],
    )


@router.get(
    "/{wheel_id}",
    response_model=WheelAIOutputContract,
    summary="Get inspection result for a wheel",
)
def get_inspection(wheel_id: str):
    return WheelAIOutputContract(
        wheel_id=wheel_id,
        component="rim",
        defect_type=None,
        location=None,
        severity=None,
        defect_confidence=None,
        root_cause=None,
        root_cause_confidence=None,
        future_risk=None,
        recommended_action=None,
        affected_batches=[],
    )