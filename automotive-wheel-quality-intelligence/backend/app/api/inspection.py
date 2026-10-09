"""Wheel image upload and inspection API.

Uploaded image files are persisted in Supabase Storage. Model inference still
uses a temporary local copy when the requested image is stored remotely; that
copy is removed after inference finishes.
"""

import io
import json
import logging
import os
import re
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
from app.services.root_cause_engine import root_cause_engine

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


def _to_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bbox_columns(bbox) -> dict:
    """Convert a model bbox into the defects table's x1/y1/x2/y2 columns."""
    if bbox is None:
        return {}
    if hasattr(bbox, "model_dump"):
        try:
            bbox = bbox.model_dump()
        except Exception:
            pass
    if isinstance(bbox, dict):
        # Accommodate common names emitted by detection libraries.
        candidates = (
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        )
        for keys in candidates:
            if all(key in bbox for key in keys):
                values = [_to_float(bbox[key]) for key in keys]
                if all(value is not None for value in values):
                    return dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), values))
        if "xyxy" in bbox:
            bbox = bbox["xyxy"]
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        values = [_to_float(value) for value in bbox[:4]]
        if all(value is not None for value in values):
            return dict(zip(("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"), values))
    return {}


def _is_defect_label(label) -> bool:
    if label is None:
        return False
    value = str(label).strip().lower().replace("-", "_").replace(" ", "_")
    return value not in {
        "", "unknown", "normal", "good", "ok", "pass", "no_defect",
        "defect_free", "no_defects", "background", "none", "null",
    }


def _derive_severity(defect_type: str | None) -> str | None:
    """Rule-based severity estimate; validate/tune these class rules for production."""
    if not defect_type:
        return None
    label = str(defect_type).lower().replace("_", " ").replace("-", " ")
    high_terms = ("crack", "fracture", "broken", "breakage", "structural", "collapse")
    medium_terms = (
        "scratch", "dent", "corrosion", "porosity", "pitting", "deformation",
        "burr", "inclusion", "shrinkage", "oxide", "cavity", "void",
    )
    if any(term in label for term in high_terms):
        return "high"
    if any(term in label for term in medium_terms):
        return "medium"
    # Do not invent a physical severity for unrecognised class labels.
    return None


def _response_error(response: httpx.Response) -> tuple[str | None, str]:
    try:
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("code"), str(payload.get("message") or response.text)
    except Exception:
        pass
    return None, response.text


async def _rest_request(
        method: str,
        table: str,
        *,
        params: dict | None = None,
        payload=None,
        prefer: str | None = None,
) -> httpx.Response:
    supabase_url, supabase_key = _get_supabase_credentials()
    headers = {
        **_supabase_request_headers(supabase_key),
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)) as client:
        return await client.request(
            method,
            f"{supabase_url}/rest/v1/{table}",
            params=params,
            headers=headers,
            json=payload,
        )


async def _find_reference_row(
        table: str,
        value,
        code_column: str,
        *,
        select: str = "id",
) -> dict | None:
    """Resolve either a table UUID or a human-readable code to a row."""
    if value is None or str(value).strip() == "":
        return None
    raw_value = str(value).strip()
    try:
        parsed_uuid = str(uuid.UUID(raw_value))
    except (ValueError, TypeError, AttributeError):
        filters = [(code_column, raw_value)]
    else:
        filters = [("id", parsed_uuid), (code_column, raw_value)]

    for filter_column, filter_value in filters:
        response = await _rest_request(
            "GET",
            table,
            params={
                "select": select,
                filter_column: f"eq.{filter_value}",
                "limit": "1",
            },
        )
        if response.is_error:
            logger.warning(
                "Could not resolve %s reference %r using %s: status=%s response=%s",
                table, value, filter_column, response.status_code, response.text[:500],
            )
            continue
        try:
            rows = response.json()
        except Exception:
            rows = []
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            return rows[0]
    logger.warning("No matching row found in public.%s for reference %r", table, value)
    return None


async def _resolve_machine_batch_ids(machine_value, batch_value) -> tuple[str | None, str | None]:
    machine_row = await _find_reference_row(
        "machines", machine_value, "machine_id", select="id"
    ) if machine_value is not None else None
    batch_row = await _find_reference_row(
        "batches", batch_value, "batch_id", select="id,machine_id"
    ) if batch_value is not None else None

    machine_uuid = machine_row.get("id") if machine_row else None
    batch_uuid = batch_row.get("id") if batch_row else None

    # If the caller supplied a batch but no resolvable machine, reuse the machine
    # UUID already associated with that batch.
    if not machine_uuid and batch_row:
        machine_uuid = batch_row.get("machine_id")

    return machine_uuid, batch_uuid


async def _upsert_component(
        *, component_code: str, component: str, storage_image_path: str | None,
        machine_uuid: str | None, batch_uuid: str | None,
) -> dict:
    record = {
        "component_id": component_code[:100],
        "component_type": "aluminium_wheel" if component == "rim" else "tyre",
        "image_path": storage_image_path,
    }
    if machine_uuid:
        record["machine_id"] = machine_uuid
    if batch_uuid:
        record["batch_id"] = batch_uuid

    response = await _rest_request(
        "POST",
        "components",
        params={"on_conflict": "component_id"},
        payload=record,
        prefer="resolution=merge-duplicates,return=representation",
    )
    if response.is_error:
        logger.error(
            "Component upsert failed: status=%s response=%s",
            response.status_code, response.text[:1000],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "AI inspection ran, but saving the component row to public.components failed. "
                "Verify component_id is unique, image_path exists, and backend table permissions are configured."
            ),
        )

    try:
        rows = response.json()
    except Exception:
        rows = []
    if isinstance(rows, list) and rows and rows[0].get("id"):
        return rows[0]

    # Some API configurations may return no representation. Resolve the row by its unique code.
    row = await _find_reference_row(
        "components", component_code, "component_id", select="id,component_id"
    )
    if row and row.get("id"):
        return row
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="The component was submitted but its database UUID could not be retrieved.",
    )


async def _insert_prediction_row(record: dict, defect_type: str) -> dict | None:
    """Insert using the prediction class column name used by the current schema.

    The project originally used prediction_type; this also supports common renamed
    alternatives such as defect_type/predicted_class without silently losing the class.
    """
    prediction_columns = (
        "defect_type", "defect_type", "predicted_class",
        "predicted_label", "prediction_class", "defect_class",
    )
    dropped_optional: set[str] = set()
    last_error = ""

    for field_name in prediction_columns:
        attempt = {key: value for key, value in record.items() if key not in dropped_optional}
        attempt[field_name] = defect_type
        response = await _rest_request(
            "POST", "predictions", params={"select": "*"}, payload=attempt,
            prefer="return=representation",
        )
        if not response.is_error:
            try:
                rows = response.json()
            except Exception:
                rows = []
            return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None

        code, message = _response_error(response)
        last_error = message
        match = re.search(r"Could not find the '([^']+)' column", message)
        missing_column = match.group(1) if match else None
        if code == "PGRST204" and missing_column == field_name:
            logger.info(
                "predictions.%s is not present in the current schema; trying another prediction-class column name",
                field_name,
            )
            continue
        # The extended schema recommends component_id. Older versions of the
        # user's existing predictions table may not have it; explanation still
        # carries the component code and the defects table retains the FK link.
        if code == "PGRST204" and missing_column == "component_id" and "component_id" in attempt:
            dropped_optional.add("component_id")
            # Retry this field name with the optional column removed.
            retry = {key: value for key, value in record.items() if key not in dropped_optional}
            retry[field_name] = defect_type
            retry_response = await _rest_request(
                "POST", "predictions", params={"select": "*"}, payload=retry,
                prefer="return=representation",
            )
            if not retry_response.is_error:
                try:
                    rows = retry_response.json()
                except Exception:
                    rows = []
                return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None
            retry_code, retry_message = _response_error(retry_response)
            last_error = retry_message
            retry_match = re.search(r"Could not find the '([^']+)' column", retry_message)
            retry_missing = retry_match.group(1) if retry_match else None
            if retry_code == "PGRST204" and retry_missing == field_name:
                continue
        logger.error(
            "PostgREST prediction insert failed: status=%s code=%s response=%s",
            response.status_code, code, response.text[:1200],
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "AI inspection completed, but saving to public.predictions failed. "
                f"Database response: {message[:500]}"
            ),
        )

    logger.error("No supported prediction-class column found in public.predictions: %s", last_error)
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=(
            "Could not find a supported prediction class column in public.predictions. "
            "Expected one of: defect_type, defect_type, predicted_class, predicted_label, prediction_class, defect_class. "
            "Check the actual column name and the Supabase schema cache."
        ),
    )


async def _insert_related_row(table: str, record: dict, *, required: bool = True) -> dict | None:
    response = await _rest_request(
        "POST", table, params={"select": "*"}, payload=record,
        prefer="return=representation",
    )
    if response.is_error:
        logger.error(
            "Insert into public.%s failed: status=%s response=%s",
            table, response.status_code, response.text[:1200],
        )
        if required:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"AI inspection ran, but saving rows to public.{table} failed. See backend log for the database response.",
            )
        return None
    try:
        rows = response.json()
    except Exception:
        rows = []
    return rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else None


async def _save_prediction_record(
        *,
        component: str,
        image_path: str,
        wheel_id: str | None,
        machine_id: str | None,
        batch_id: str | None,
        defect_type: str | None,
        confidence_score,
        model_version: str | None,
        location=None,
        details=None,
        defect_items: list[dict] | None = None,
        root_cause_info: dict | None = None,
) -> dict | None:
    """Persist a component, overall prediction, individual defects and high-severity alerts."""
    prediction_class = str(defect_type).strip() if defect_type is not None else ""
    if not prediction_class:
        prediction_class = f"{component}_unknown"
    prediction_class = prediction_class[:100]
    numeric_probability = _to_float(confidence_score)

    # Root-cause lookup is independent of production history and future-risk
    # prediction. It uses only the predicted defect label and curated CSV rules.
    if root_cause_info is None:
        root_cause_info = root_cause_engine.get_root_cause(prediction_class)

    enriched_defect_items: list[dict] = []
    for original_item in defect_items or []:
        item = dict(original_item)
        mappings = root_cause_engine.get_root_causes(item.get("defect_type"))
        if mappings:
            item["root_cause_analysis"] = mappings
        enriched_defect_items.append(item)
    defect_items = enriched_defect_items

    storage_image_path = None
    if _local_image_candidate(image_path) is None:
        storage_image_path = _storage_object_path(image_path)

    resolved_machine_uuid, resolved_batch_uuid = await _resolve_machine_batch_ids(
        machine_id, batch_id
    )
    component_code = (str(wheel_id).strip() if wheel_id else "") or f"WHEEL-{uuid.uuid4().hex[:12]}"
    component_row = await _upsert_component(
        component_code=component_code,
        component=component,
        storage_image_path=storage_image_path,
        machine_uuid=resolved_machine_uuid,
        batch_uuid=resolved_batch_uuid,
    )
    component_uuid = component_row.get("id")

    explanation_payload = {
        "component": component,
        "wheel_id": component_code,
        "component_uuid": component_uuid,
        "image_path": storage_image_path,
        "requested_image_reference": image_path,
        "machine_reference": machine_id,
        "machine_uuid": resolved_machine_uuid,
        "batch_reference": batch_id,
        "batch_uuid": resolved_batch_uuid,
        "location": location,
        "details": details,
        "root_cause_analysis": root_cause_info,
        "root_cause_confidence_interpretation": (
            "Curated knowledge-base confidence; not a statistically calibrated cause probability."
            if root_cause_info else None
        ),
        "defects": defect_items or [],
    }
    prediction_record = {
        "component_id": component_uuid,
        "machine_id": resolved_machine_uuid,
        "batch_id": resolved_batch_uuid,
        "target_domain": component,
        "image_path": storage_image_path,
        "confidence_score": numeric_probability,
        "model_version": str(model_version)[:100] if model_version else None,
        "explanation": json.dumps(explanation_payload, ensure_ascii=False, default=str),
    }
    # Avoid explicitly posting null FK values; Postgres defaults/NULLs remain in effect.
    prediction_record = {key: value for key, value in prediction_record.items() if value is not None}
    saved_prediction = await _insert_prediction_row(prediction_record, prediction_class)

    # Save one defects row per actual model-detected defect.
    created_defects: list[dict] = []
    for item in defect_items or []:
        label = str(item.get("defect_type") or "").strip()
        if not _is_defect_label(label):
            continue
        severity = item.get("severity") or _derive_severity(label)
        defect_record = {
            "component_id": component_uuid,
            "target_domain": item.get("target_domain") or component,
            "defect_type": label[:100],
            "confidence": _to_float(item.get("confidence")),
            "severity": severity,
            "yolo_class": str(item["yolo_class"])[:100] if item.get("yolo_class") else None,
            "yolo_confidence": _to_float(item.get("yolo_confidence")),
            "cnn_class": str(item["cnn_class"])[:100] if item.get("cnn_class") else None,
            "cnn_confidence": _to_float(item.get("cnn_confidence")),
            "classification_agreement": item.get("classification_agreement"),
            "model_version": str(item.get("model_version") or model_version or "unknown")[:100],
            **_bbox_columns(item.get("bbox")),
        }
        defect_record = {key: value for key, value in defect_record.items() if value is not None}
        saved_defect = await _insert_related_row("defects", defect_record, required=True)
        if saved_defect:
            created_defects.append(saved_defect)

            # Save one recommendation for each curated cause/action mapping.
            # The recommendations.confidence value is knowledge-base confidence,
            # not a statistically calibrated probability of root cause.
            for mapping in root_cause_engine.get_root_causes(label):
                process_factors = mapping.get("process_factors") or "not specified"
                process_stage = mapping.get("process_stage") or "not specified"
                recommendation_reason = (
                    f"Probable root cause: {mapping.get('probable_root_cause')}. "
                    f"Process stage: {process_stage}. Process factors to review: {process_factors}. "
                    "Knowledge confidence is a curated engineering score, not a calibrated probability."
                )
                recommendation_record = {
                    "machine_id": resolved_machine_uuid,
                    "batch_id": resolved_batch_uuid,
                    "defect_id": saved_defect.get("id"),
                    "action": mapping.get("corrective_action"),
                    "reason": recommendation_reason,
                    "confidence": mapping.get("knowledge_confidence"),
                    "status": "pending",
                }
                recommendation_record = {
                    key: value for key, value in recommendation_record.items()
                    if value is not None
                }
                await _insert_related_row(
                    "recommendations", recommendation_record, required=False
                )

            # Create alerts only for high-severity defects by default. This avoids
            # generating noisy alerts for every predicted class. Configure
            # ALERT_SEVERITIES=high,medium in .env to include medium-severity alerts.
            alert_severities = {
                value.strip().lower()
                for value in os.getenv("ALERT_SEVERITIES", "high").split(",")
                if value.strip()
            }
            if severity and severity.lower() in alert_severities:
                message = (
                    f"{severity.upper()} severity {component} defect '{label}' detected "
                    f"on component '{component_code}' with confidence "
                    f"{_to_float(item.get('confidence')) if _to_float(item.get('confidence')) is not None else 'unknown'}."
                )
                alert_record = {
                    "machine_id": resolved_machine_uuid,
                    "batch_id": resolved_batch_uuid,
                    "defect_id": saved_defect.get("id"),
                    "alert_type": "high_severity_defect",
                    "severity": severity,
                    "message": message,
                }
                alert_record = {key: value for key, value in alert_record.items() if value is not None}
                # A failed alert should be visible in logs but must not discard a
                # successful prediction/defect write.
                await _insert_related_row("alerts", alert_record, required=False)

    logger.info(
        "Persisted inspection: component_id=%s prediction_id=%s defects=%d component_code=%s",
        component_uuid,
        (saved_prediction or {}).get("id") if isinstance(saved_prediction, dict) else None,
        len(created_defects),
        component_code,
    )
    return saved_prediction

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
    wheel_id = (str(request.wheel_id).strip() if request.wheel_id else "") or f"WHEEL-{uuid.uuid4().hex[:12]}"

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
        tyre_defect_items = []
        for detection in (tyre_result.detections or []):
            detected_class = getattr(detection, "class_name", None)
            if not _is_defect_label(detected_class):
                continue
            detected_confidence = _to_float(getattr(detection, "confidence", None))
            tyre_defect_items.append({
                "target_domain": "tyre",
                "defect_type": str(detected_class),
                "confidence": detected_confidence,
                "severity": _derive_severity(str(detected_class)),
                "bbox": getattr(detection, "bbox", None),
                "yolo_class": str(detected_class),
                "yolo_confidence": detected_confidence,
                "cnn_class": None,
                "cnn_confidence": None,
                "classification_agreement": None,
                "model_version": getattr(tyre_result, "model", None) or "tyre_yolo",
            })
        root_cause_result = root_cause_engine.get_root_cause(top_defect)
        await _save_prediction_record(
            component="tyre",
            image_path=request.image_path,
            wheel_id=wheel_id,
            machine_id=request.machine_id,
            batch_id=request.batch_id,
            defect_type=top_defect,
            confidence_score=top_confidence,
            model_version=getattr(tyre_result, "model", None) or "tyre_yolo",
            location=top_bbox,
            details=tyre_details,
            defect_items=tyre_defect_items,
            root_cause_info=root_cause_result,
        )

        return WheelAIOutputContract(
            wheel_id=wheel_id,
            batch_id=request.batch_id,
            machine_id=request.machine_id,
            component="tyre",
            hybrid=None,
            rim=None,
            tyre=tyre_result.model_dump(),
            defect_type=top_defect,
            location=top_bbox,
            severity=_derive_severity(top_defect),
            defect_confidence=top_confidence,
            root_cause=(root_cause_result.get("probable_root_cause") if root_cause_result else None),
            root_cause_confidence=(root_cause_result.get("knowledge_confidence") if root_cause_result else None),
            future_risk=None,
            recommended_action=(root_cause_result.get("corrective_action") if root_cause_result else None),
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

        hybrid_defect_items = []
        for ld in hybrid_result.localized_defects:
            localization = ld.localization
            classification = ld.classification
            yolo_label = localization.yolo_defect_type
            cnn_label = classification.cnn_defect_type
            selected_label = cnn_label or yolo_label
            if not _is_defect_label(selected_label):
                continue
            selected_confidence = (
                classification.cnn_confidence
                if classification.cnn_confidence is not None
                else localization.yolo_confidence
            )
            hybrid_defect_items.append({
                "target_domain": "rim",
                "defect_type": str(selected_label),
                "confidence": _to_float(selected_confidence),
                "severity": _derive_severity(str(selected_label)),
                "bbox": localization.bbox,
                "yolo_class": str(yolo_label) if yolo_label is not None else None,
                "yolo_confidence": _to_float(localization.yolo_confidence),
                "cnn_class": str(cnn_label) if cnn_label is not None else None,
                "cnn_confidence": _to_float(classification.cnn_confidence),
                "classification_agreement": ld.classification_agreement,
                "model_version": hybrid_api.model,
            })
        if not hybrid_defect_items and _is_defect_label(fic.get("defect_type")):
            hybrid_defect_items.append({
                "target_domain": "rim",
                "defect_type": str(fic.get("defect_type")),
                "confidence": _to_float(fic.get("confidence")),
                "severity": _derive_severity(str(fic.get("defect_type"))),
                "bbox": None,
                "yolo_class": None,
                "yolo_confidence": None,
                "cnn_class": str(fic.get("defect_type")),
                "cnn_confidence": _to_float(fic.get("confidence")),
                "classification_agreement": None,
                "model_version": str(fic.get("model", hybrid_api.model)),
            })

        root_cause_result = root_cause_engine.get_root_cause(top_defect)
        await _save_prediction_record(
            component="rim",
            image_path=request.image_path,
            wheel_id=wheel_id,
            machine_id=request.machine_id,
            batch_id=request.batch_id,
            defect_type=top_defect,
            confidence_score=top_confidence,
            model_version=hybrid_api.model,
            location=top_bbox,
            details=hybrid_api.model_dump(),
            defect_items=hybrid_defect_items,
            root_cause_info=root_cause_result,
        )

        return WheelAIOutputContract(
            wheel_id=wheel_id,
            batch_id=request.batch_id,
            machine_id=request.machine_id,
            component="rim",
            hybrid=hybrid_api,
            rim=rim_legacy,
            tyre=None,
            defect_type=top_defect,
            location=top_bbox,
            severity=_derive_severity(top_defect),
            defect_confidence=top_confidence,
            root_cause=(root_cause_result.get("probable_root_cause") if root_cause_result else None),
            root_cause_confidence=(root_cause_result.get("knowledge_confidence") if root_cause_result else None),
            future_risk=None,
            recommended_action=(root_cause_result.get("corrective_action") if root_cause_result else None),
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

    cnn_defect_items = []
    if _is_defect_label(cnn_pred.defect_type):
        cnn_defect_items.append({
            "target_domain": "rim",
            "defect_type": str(cnn_pred.defect_type),
            "confidence": _to_float(cnn_pred.confidence),
            "severity": _derive_severity(str(cnn_pred.defect_type)),
            "bbox": None,
            "yolo_class": None,
            "yolo_confidence": None,
            "cnn_class": str(cnn_pred.defect_type),
            "cnn_confidence": _to_float(cnn_pred.confidence),
            "classification_agreement": None,
            "model_version": cnn_pred.model,
        })

    root_cause_result = root_cause_engine.get_root_cause(cnn_pred.defect_type)
    await _save_prediction_record(
        component="rim",
        image_path=request.image_path,
        wheel_id=wheel_id,
        machine_id=request.machine_id,
        batch_id=request.batch_id,
        defect_type=cnn_pred.defect_type,
        confidence_score=cnn_pred.confidence,
        model_version=cnn_pred.model,
        location=None,
        details={
            "class_id": cnn_pred.class_id,
            "inference_ms": cnn_pred.inference_ms,
            "device": cnn_pred.device,
        },
        defect_items=cnn_defect_items,
        root_cause_info=root_cause_result,
    )

    return WheelAIOutputContract(
        wheel_id=wheel_id,
        batch_id=request.batch_id,
        machine_id=request.machine_id,
        component="rim",
        hybrid=None,
        rim=rim_result,
        tyre=None,
        defect_type=cnn_pred.defect_type,
        location=None,
        severity=_derive_severity(cnn_pred.defect_type),
        defect_confidence=cnn_pred.confidence,
        root_cause=(root_cause_result.get("probable_root_cause") if root_cause_result else None),
        root_cause_confidence=(root_cause_result.get("knowledge_confidence") if root_cause_result else None),
        future_risk=None,
        recommended_action=(root_cause_result.get("corrective_action") if root_cause_result else None),
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