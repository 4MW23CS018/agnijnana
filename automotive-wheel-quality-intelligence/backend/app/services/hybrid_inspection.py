"""
HybridRimInspector — Fused YOLO localization + CNN classification pipeline.

Design principle:
    YOLO answers: "WHERE is the candidate defect region?" (spatial localization)
    CNN  answers: "WHAT defect type is this region?"     (semantic classification)

Their confidence scores measure different tasks and are NEVER compared directly.
YOLO conf = "region contains one of {bent_rim, crack, scratch}"
CNN  conf = "this crop belongs to one of 8 rim defect classes"

Supported YOLO classes (3):   bent_rim, crack, scratch
Supported CNN classes (8):    bent_rim, blow_hole, crack, incomplete_welding,
                               paint_damage, porosity, scratch, scuff

If YOLO detects nothing above its threshold:
    → localization_status = "no_yolo_detection"
    → no CNN crop inference is triggered
    → the full-image CNN result is returned separately (CNN-only path)

Class agreement:
    classification_agreement = True   if yolo_defect_type == cnn_defect_type
    classification_agreement = False  if they differ (disagreement is informative, not an error)

Mask status:
    mask_coefficients (32-dim) are passed through from YOLO.
    Full binary mask reconstruction (proto × coefficients) is marked as
    "mask_pending" until a dedicated rendering milestone. Bboxes are exact.

This module imports BOTH services and orchestrates them. It must NOT be
imported from rim_cnn_classifier.py or rim_yolo_detector.py.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from app.services.rim_cnn_classifier import RimCNNClassifier, rim_cnn_classifier
from app.services.rim_yolo_detector import (
    RimYOLODetector,
    YOLODetection,
    YOLOInferenceResult,
    rim_yolo_detector,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Padding fraction added around each YOLO bbox before CNN classification.
# Adds 10% of box dimension on each side to provide context to the CNN.
CROP_PADDING_FRACTION: float = 0.10


# ── Output Schemas ─────────────────────────────────────────────────────────────

class LocalizationInfo(BaseModel):
    """YOLO spatial localization information for one detected region."""

    source: str = Field(default="rim_yolo26s_seg_v1")
    yolo_class_id: int = Field(..., description="YOLO class ID (0–2)")
    yolo_defect_type: str = Field(..., description="YOLO defect label")
    yolo_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="YOLO region confidence"
    )

    bbox: List[float] = Field(
        ...,
        description="[x1, y1, x2, y2] in original pixel coords"
    )

    mask_status: str = Field(
        default="pending",
        description="Instance segmentation mask reconstruction status"
    )

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

class ClassificationInfo(BaseModel):
    """CNN classification result for one localized crop."""

    source: str = Field(default="rim_cnn_v1")
    cnn_class_id: int = Field(..., ge=0, le=7)
    cnn_defect_type: str = Field(..., description="CNN defect label (8-class)")
    cnn_confidence: float = Field(..., ge=0.0, le=1.0)
    inference_ms: float = Field(..., ge=0.0, description="CNN crop inference latency")
    device: str


class LocalizedDefect(BaseModel):
    """
    One localized defect: paired YOLO region + CNN classification.

    IMPORTANT: yolo_confidence and cnn_confidence are NOT comparable.
    They measure different tasks and must NOT be combined into a single score.
    """

    localization: LocalizationInfo
    classification: ClassificationInfo
    classification_agreement: bool = Field(
        ...,
        description=(
            "True if YOLO and CNN predict the same defect class name. "
            "False if they disagree — disagreement is informative, not an error."
        ),
    )


class HybridInspectionResult(BaseModel):
    """
    Full hybrid inspection result for one image.

    When YOLO finds no detections:
        localization_status = 'no_yolo_detection'
        localized_defects   = []

    When YOLO finds detections:
        localization_status = 'detected'
        localized_defects   = list of LocalizedDefect records

    The full_image_cnn field always contains the CNN result on the complete image,
    providing a fallback classification regardless of YOLO outcome.
    """

    model: str = "hybrid_yolo_cnn"
    localization_status: str = Field(
        ...,
        description="'detected' | 'no_yolo_detection'",
    )
    yolo_inference_ms: float = Field(..., ge=0.0)
    cnn_total_inference_ms: float = Field(..., ge=0.0, description="Sum of all CNN crop inferences")
    total_hybrid_ms: float = Field(..., ge=0.0, description="End-to-end wall time")
    device: str

    localized_defects: List[LocalizedDefect] = Field(default_factory=list)

    # Full-image CNN baseline — always available as fallback
    full_image_cnn: Optional[dict] = Field(
        default=None,
        description="CNN result on the complete (un-cropped) image",
    )

    yolo_conf_threshold: float


# ── Hybrid Inspector ───────────────────────────────────────────────────────────

class HybridRimInspector:
    """
    Orchestrates YOLO localization → crop extraction → CNN classification.

    YOLO is used only for spatial localization.
    CNN is used for semantic defect classification on extracted crops.
    Their confidence values are kept separate and never compared.
    """

    def __init__(
        self,
        yolo_detector: Optional[RimYOLODetector] = None,
        cnn_classifier: Optional[RimCNNClassifier] = None,
        crop_padding: float = CROP_PADDING_FRACTION,
    ):
        self._yolo = yolo_detector or rim_yolo_detector
        self._cnn = cnn_classifier or rim_cnn_classifier
        self._crop_padding = crop_padding

    @property
    def is_ready(self) -> bool:
        return self._yolo.is_ready and self._cnn.is_ready

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_pil_image(self, image_input: Union[str, Path, Image.Image]) -> Image.Image:
        """Load and validate image, resolving relative paths against project root."""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")
        path = Path(image_input)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise ValueError(f"Image path does not exist: {image_input}")
        try:
            return Image.open(path).convert("RGB")
        except Exception as exc:
            raise ValueError(f"Invalid image file: {exc}") from exc

    def _extract_crop(
        self,
        pil_img: Image.Image,
        bbox: List[float],
    ) -> Image.Image:
        """
        Crop the padded bounding-box region from the original image.

        Adds CROP_PADDING_FRACTION on each side for context, then clamps
        to image boundaries.
        """
        orig_w, orig_h = pil_img.size
        x1, y1, x2, y2 = bbox

        pad_x = (x2 - x1) * self._crop_padding
        pad_y = (y2 - y1) * self._crop_padding

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(orig_w, x2 + pad_x)
        cy2 = min(orig_h, y2 + pad_y)

        # Guard: if box is degenerate, return a minimal 4-px square
        if cx2 - cx1 < 4 or cy2 - cy1 < 4:
            cx1 = max(0, cx1)
            cy1 = max(0, cy1)
            cx2 = min(orig_w, cx1 + 4)
            cy2 = min(orig_h, cy1 + 4)

        return pil_img.crop((cx1, cy1, cx2, cy2))

    def _classify_crop(self, crop: Image.Image) -> ClassificationInfo:
        """Run CNN classification on a single cropped region."""
        result = self._cnn.predict(crop)
        return ClassificationInfo(
            source="rim_cnn_v1",
            cnn_class_id=result.class_id,
            cnn_defect_type=result.defect_type,
            cnn_confidence=result.confidence,
            inference_ms=result.inference_ms,
            device=result.device,
        )

    @staticmethod
    def _check_agreement(yolo_defect: str, cnn_defect: str) -> bool:
        """
        Return True if YOLO and CNN predict the same defect class name.

        Comparison is case-insensitive and normalises whitespace.
        Disagreement is informative — it does NOT indicate an error.
        """
        return yolo_defect.strip().lower() == cnn_defect.strip().lower()

    # ── Public API ─────────────────────────────────────────────────────────────

    def inspect(
        self,
        image_input: Union[str, Path, Image.Image],
    ) -> HybridInspectionResult:
        """
        Run the full hybrid YOLO → crop → CNN pipeline on one image.

        Pipeline:
            1. Load image.
            2. Run YOLO localization on full image.
            3. If YOLO detects nothing → return 'no_yolo_detection' status.
            4. For each YOLO detection:
                a. Extract padded crop from original image.
                b. Run CNN classification on crop.
                c. Record both YOLO and CNN results with agreement flag.
            5. Run CNN on the full image for a baseline fallback result.
            6. Return HybridInspectionResult.

        Args:
            image_input: File path (str/Path) or PIL Image.

        Returns:
            HybridInspectionResult — always fully populated.

        Raises:
            RuntimeError: If either model is not loaded.
            ValueError: If image cannot be read.
        """
        if not self.is_ready:
            missing = []
            if not self._yolo.is_ready:
                missing.append("YOLO")
            if not self._cnn.is_ready:
                missing.append("CNN")
            raise RuntimeError(
                f"HybridRimInspector: models not ready — {', '.join(missing)}"
            )

        wall_start = time.time()

        # ── 1. Load image ──────────────────────────────────────────────────────
        pil_img = self._load_pil_image(image_input)

        # ── 2. YOLO localization ───────────────────────────────────────────────
        yolo_result: YOLOInferenceResult = self._yolo.detect(pil_img)
        yolo_ms = yolo_result.inference_ms

        # ── 3. CNN on full image (baseline, always) ────────────────────────────
        full_cnn = self._cnn.predict(pil_img)
        full_cnn_dict = {
            "model": full_cnn.model,
            "defect_type": full_cnn.defect_type,
            "class_id": full_cnn.class_id,
            "confidence": full_cnn.confidence,
            "inference_ms": full_cnn.inference_ms,
            "device": full_cnn.device,
        }

        # ── 4. Handle zero detections ──────────────────────────────────────────
        if len(yolo_result.detections) == 0:
            total_ms = round((time.time() - wall_start) * 1000, 2)
            return HybridInspectionResult(
                localization_status="no_yolo_detection",
                yolo_inference_ms=yolo_ms,
                cnn_total_inference_ms=full_cnn.inference_ms,
                total_hybrid_ms=total_ms,
                device=full_cnn.device,
                localized_defects=[],
                full_image_cnn=full_cnn_dict,
                yolo_conf_threshold=self._yolo.conf_threshold,
            )

        # ── 5. Classify each YOLO crop ─────────────────────────────────────────
        localized: List[LocalizedDefect] = []
        total_cnn_ms = full_cnn.inference_ms  # include full-image pass

        for det in yolo_result.detections:
            det: YOLODetection

            # Extract crop with context padding
            crop = self._extract_crop(pil_img, det.bbox)

            # CNN classification on crop
            crop_cls = self._classify_crop(crop)
            total_cnn_ms += crop_cls.inference_ms

            # Agreement check
            agreement = self._check_agreement(det.defect_type, crop_cls.cnn_defect_type)

            if not agreement:
                logger.info(
                    "Class disagreement: YOLO=%s (conf=%.3f)  CNN=%s (conf=%.3f)",
                    det.defect_type, det.confidence,
                    crop_cls.cnn_defect_type, crop_cls.cnn_confidence,
                )

            localized.append(
                LocalizedDefect(
                    localization=LocalizationInfo(
    source="rim_yolo26s_seg_v1",
    yolo_class_id=det.class_id,
    yolo_defect_type=det.defect_type,
    yolo_confidence=det.confidence,
    bbox=det.bbox,
    mask_status="available" if det.mask_available else "unavailable",
    mask_area_pixels=det.mask_area_pixels,
    mask_area_ratio=det.mask_area_ratio,
    mask_polygon=det.mask_polygon,
),
                    classification=ClassificationInfo(
                        source="rim_cnn_v1",
                        cnn_class_id=crop_cls.cnn_class_id,
                        cnn_defect_type=crop_cls.cnn_defect_type,
                        cnn_confidence=crop_cls.cnn_confidence,
                        inference_ms=crop_cls.inference_ms,
                        device=crop_cls.device,
                    ),
                    classification_agreement=agreement,
                )
            )

        total_ms = round((time.time() - wall_start) * 1000, 2)

        return HybridInspectionResult(
            localization_status="detected",
            yolo_inference_ms=yolo_ms,
            cnn_total_inference_ms=round(total_cnn_ms, 2),
            total_hybrid_ms=total_ms,
            device=full_cnn.device,
            localized_defects=localized,
            full_image_cnn=full_cnn_dict,
            yolo_conf_threshold=self._yolo.conf_threshold,
        )

    async def inspect_async(
        self,
        image_input: Union[str, Path, Image.Image],
    ) -> HybridInspectionResult:
        """Async wrapper — offloads blocking inference to a thread pool."""
        return await asyncio.to_thread(self.inspect, image_input)


# ── Module-level singleton ─────────────────────────────────────────────────────
hybrid_inspector = HybridRimInspector()
