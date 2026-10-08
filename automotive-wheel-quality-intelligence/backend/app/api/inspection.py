import logging

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.schemas.quality import (
    HybridRimResult,
    HybridLocalizedDefect,
    HybridLocalizationInfo,
    HybridClassificationInfo,
    RimCNNResult,
    WheelAIOutputContract,
    WheelInspectionRequest,
)
from app.services.image_service import save_wheel_image
from app.services.hybrid_inspection import hybrid_inspector
from app.services.inference import run_rim_cnn_inference_async
from app.services.tyre_yolo_detector import tyre_yolo_detector
from app.services.severity_engine import calculate_severity

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/upload",
    status_code=status.HTTP_201_CREATED,
    summary="Upload an aluminium wheel image",
)
async def upload_wheel_image(
    image: UploadFile = File(...),
):
    """
    Upload and validate an aluminium wheel image using PIL verification.
    """
    image_path = await save_wheel_image(image)

    return {
        "filename": image.filename,
        "image_path": image_path,
        "message": "Wheel image uploaded successfully",
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
        try:
            tyre_result = await tyre_yolo_detector.detect_async(request.image_path)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
        except Exception as exc:
            logger.error("Tyre inspection failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Tyre inspection internal failure",
            )

        top_defect = None
        top_confidence = None
        top_bbox = None
        if tyre_result.detections:
            top_det = tyre_result.detections[0]
            top_defect = top_det.class_name
            top_confidence = top_det.confidence
            top_bbox = top_det.bbox

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
            severity_score=None,
            severity_rationale=None,
            defect_confidence=top_confidence,
            root_cause=None,
            root_cause_confidence=None,
            future_risk=None,
            recommended_action=None,
            affected_batches=[],
        )

    # ── Rim Inspection Path (Hybrid YOLO + CNN) ──────────────────────────────
    if hybrid_inspector.is_ready:
        try:
            hybrid_result = await hybrid_inspector.inspect_async(request.image_path)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
        except Exception as exc:
            logger.error("Hybrid inspection failed: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Hybrid inspection internal failure",
            )

        # Map internal service types → API schema types & calculate severity per defect
        api_localized = []
        for ld in hybrid_result.localized_defects:
            loc_sev = calculate_severity(
                defect_type=ld.classification.cnn_defect_type,
                area_ratio=ld.localization.mask_area_ratio,
                mask_status=ld.localization.mask_status,
                confidence=ld.classification.cnn_confidence,
            )
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
                    severity_level=loc_sev["severity_level"],
                    severity_score=loc_sev["severity_score"],
                    severity_rationale=loc_sev["rationale"],
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

        # Determine top-level defect_type, confidence, and primary severity
        fic = hybrid_result.full_image_cnn or {}
        if api_localized:
            top = api_localized[0].classification
            top_defect = top.cnn_defect_type
            top_confidence = top.cnn_confidence
            top_bbox = api_localized[0].localization.bbox
            top_sev = calculate_severity(
                defect_type=top_defect,
                area_ratio=api_localized[0].localization.mask_area_ratio,
                mask_status=api_localized[0].localization.mask_status,
                confidence=top_confidence,
            )
        else:
            top_defect = fic.get("defect_type", "unknown")
            top_confidence = fic.get("confidence", 0.0)
            top_bbox = None
            top_sev = calculate_severity(
                defect_type=top_defect,
                mask_status="unavailable",
                confidence=top_confidence,
            )

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
            severity=top_sev["severity_level"],
            severity_score=top_sev["severity_score"],
            severity_rationale=top_sev["rationale"],
            defect_confidence=top_confidence,
            root_cause=None,
            root_cause_confidence=None,
            future_risk=None,
            recommended_action=None,
            affected_batches=[],
        )

    # ── Fallback: CNN-only (hybrid not ready) ──────────────────────────────────
    logger.warning("Hybrid inspector not ready — falling back to CNN-only inspection")
    cnn_pred = await run_rim_cnn_inference_async(request.image_path)

    rim_result = RimCNNResult(
        model=cnn_pred.model,
        defect_type=cnn_pred.defect_type,
        class_id=cnn_pred.class_id,
        confidence=cnn_pred.confidence,
        inference_ms=cnn_pred.inference_ms,
        device=cnn_pred.device,
    )

    fallback_sev = calculate_severity(
        defect_type=cnn_pred.defect_type,
        mask_status="unavailable",
        confidence=cnn_pred.confidence,
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
        severity=fallback_sev["severity_level"],
        severity_score=fallback_sev["severity_score"],
        severity_rationale=fallback_sev["rationale"],
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
        severity_score=None,
        severity_rationale=None,
        defect_confidence=None,
        root_cause=None,
        root_cause_confidence=None,
        future_risk=None,
        recommended_action=None,
        affected_batches=[],
    )