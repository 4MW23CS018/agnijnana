from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class WheelInspectionRequest(BaseModel):
    """Input for a wheel visual inspection."""

    wheel_id: str = Field(..., description="Unique wheel/rim tracking ID")

    image_path: str = Field(
        ..., description="Path or identifier of the wheel image to inspect"
    )

    component: str = Field(
        default="rim", description="Component type to inspect: 'rim' or 'tyre'"
    )

    batch_id: Optional[str] = Field(
        default=None, description="Casting batch or production lot, when available"
    )

    machine_id: Optional[str] = Field(
        default=None, description="Die-casting machine or production station, when available"
    )


class RimCNNResult(BaseModel):
    """ConvNeXt-Tiny CNN Rim defect classification output."""

    model: str = Field(default="rim_cnn_v1", description="Model identifier")
    defect_type: str = Field(..., description="Predicted defect category name")
    class_id: int = Field(..., ge=0, le=7, description="Class label ID (0-7)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score [0.0, 1.0]")
    inference_ms: float = Field(..., ge=0.0, description="Inference latency in milliseconds")
    device: str = Field(..., description="Inference execution device ('cpu' or 'cuda')")


# ── Hybrid YOLO + CNN schemas ──────────────────────────────────────────────────

class HybridLocalizationInfo(BaseModel):
    """YOLO spatial localization for one detected region."""

    source: str = Field(default="rim_yolo26s_seg_v1")
    yolo_class_id: int
    yolo_defect_type: str
    yolo_confidence: float = Field(..., ge=0.0, le=1.0)

    bbox: List[float] = Field(
        ...,
        description="[x1, y1, x2, y2] in original pixel coords"
    )

    mask_status: str = Field(default="pending")

    mask_area_pixels: Optional[int] = Field(
        default=None,
        description="Foreground pixels in reconstructed instance mask"
    )

    mask_area_ratio: Optional[float] = Field(
        default=None,
        description="Mask area as percentage of full image area"
    )

    mask_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description="Largest mask contour in original image pixel coordinates"
    )


class HybridClassificationInfo(BaseModel):
    """CNN classification result for one localized crop."""

    source: str = Field(default="rim_cnn_v1")
    cnn_class_id: int = Field(..., ge=0, le=7)
    cnn_defect_type: str
    cnn_confidence: float = Field(..., ge=0.0, le=1.0)
    inference_ms: float = Field(..., ge=0.0)
    device: str


class HybridLocalizedDefect(BaseModel):
    """
    One localized defect: paired YOLO region + CNN classification.
    yolo_confidence and cnn_confidence are NOT comparable — they measure different tasks.
    """

    localization: HybridLocalizationInfo
    classification: HybridClassificationInfo
    classification_agreement: bool
    severity_level: Optional[str] = Field(default=None, description="Defect severity level ('Low', 'Critical', or 'Uncertain')")
    severity_score: Optional[int] = Field(default=None, ge=0, le=100, description="Numerical severity score [0, 100]")
    severity_rationale: Optional[str] = Field(default=None, description="Human-readable explanation of severity assessment")


class HybridRimResult(BaseModel):
    """
    Hybrid YOLO + CNN inspection result embedded in the API response.

    localization_status:
        'detected'          — YOLO found ≥1 region; each was classified by CNN
        'no_yolo_detection' — YOLO found nothing above threshold; CNN ran on full image only
    """

    model: str = Field(default="hybrid_yolo_cnn")
    localization_status: str
    yolo_inference_ms: float = Field(..., ge=0.0)
    cnn_total_inference_ms: float = Field(..., ge=0.0)
    total_hybrid_ms: float = Field(..., ge=0.0)
    device: str
    yolo_conf_threshold: float
    localized_defects: List[HybridLocalizedDefect] = Field(default_factory=list)
    full_image_cnn: Optional[Dict[str, Any]] = Field(
        default=None, description="CNN result on the complete un-cropped image (fallback)"
    )


# ── Main API output contract ───────────────────────────────────────────────────

class WheelAIOutputContract(BaseModel):
    """
    API output contract for wheel inspection results.
    The 'rim' field contains the hybrid YOLO+CNN result when available,
    or the CNN-only result when YOLO is unavailable (when component == 'rim').
    The 'tyre' field contains the Tyre YOLO detection result when component == 'tyre'.
    Backward-compatible legacy fields are preserved.
    """

    wheel_id: str
    batch_id: Optional[str] = None
    machine_id: Optional[str] = None
    component: Optional[str] = Field(default="rim", description="Inspected component type ('rim' or 'tyre')")

    # Hybrid result (primary for rim) — present when hybrid pipeline ran
    hybrid: Optional[HybridRimResult] = Field(
        default=None, description="Hybrid YOLO localization + CNN classification result for rim"
    )

    # CNN-only result (backward compat / fallback for rim)
    rim: Optional[RimCNNResult] = Field(
        default=None, description="CNN-only rim classification result (legacy / fallback)"
    )

    # Tyre result — present when component == 'tyre'
    tyre: Optional[Dict[str, Any]] = Field(
        default=None, description="Tyre YOLO detection result when component is 'tyre'"
    )

    # Legacy top-level fields — maintained for backward compatibility
    defect_type: Optional[str] = None
    location: Optional[Union[str, List[float]]] = Field(
        default=None, description="Bounding box [x1,y1,x2,y2] or null"
    )
    severity: Optional[str] = None
    severity_score: Optional[int] = Field(default=None, ge=0, le=100, description="Primary defect severity score [0, 100]")
    severity_rationale: Optional[str] = Field(default=None, description="Human-readable explanation of primary severity assessment")
    defect_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    root_cause: Optional[str] = None
    root_cause_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    future_risk: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    recommended_action: Optional[str] = None
    affected_batches: List[str] = Field(default_factory=list)


class DefectQuery(BaseModel):
    batch_id: Optional[str] = None
    machine_id: Optional[str] = None
    severity: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    service: str = "Aluminium Wheel Quality Intelligence Backend"
    version: str = "0.1.0"