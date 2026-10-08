"""
TyreYOLODetector — Standalone YOLO object detection service for tyre defect localization.

Model: best.pt (YOLO object detection)
Task: Tyre defect localization (bounding boxes)
Classes (14):
    0: Generic_Defect
    1: Bulge
    2: Cracks
    3: Flat_Spots
    4: Good
    5: Pitting
    6: Puncture
    7: Bad_Tire
    8: Bald_Tire
    9: Normal_Tire
    10: Xray_Open
    11: Cord_Defect
    12: Impurity
    13: Belt_Defect

Loading strategy:
  Uses the official Ultralytics YOLO API (`from ultralytics import YOLO`).
  Loads the model once upon instantiation or when _load_model() is called.

NOTE: This service is completely independent of rim detection and hybrid inspection pipelines.
      Do NOT import or modify rim detection or hybrid inspection modules.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from PIL import Image
from pydantic import BaseModel, Field

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TYRE_YOLO_MODEL_PATH = (
    PROJECT_ROOT / "ai" / "tyre_detection" / "models" / "best.pt"
)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_IMGSZ: int = 640
DEFAULT_CONF_THRESH: float = 0.25

# Fallback class taxonomy (14 classes) if model metadata cannot be read
TYRE_YOLO_CLASS_NAMES: Dict[int, str] = {
    0: "Generic_Defect",
    1: "Bulge",
    2: "Cracks",
    3: "Flat_Spots",
    4: "Good",
    5: "Pitting",
    6: "Puncture",
    7: "Bad_Tire",
    8: "Bald_Tire",
    9: "Normal_Tire",
    10: "Xray_Open",
    11: "Cord_Defect",
    12: "Impurity",
    13: "Belt_Defect",
}


# ── Output schemas ─────────────────────────────────────────────────────────────
class TyreDetection(BaseModel):
    """Single detection returned by TyreYOLODetector."""

    class_id: int = Field(..., ge=0, description="YOLO class ID")
    class_name: str = Field(..., description="Human-readable defect/tyre label")
    defect_type: str = Field(..., description="Alias for class_name for consistent API contract")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence [0, 1]")
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in original image pixel coords",
    )


class TyreYOLOInferenceResult(BaseModel):
    """Full Tyre YOLO inference result for one image."""

    model: str = "tyre_yolo_v1"
    num_detections: int = Field(..., ge=0)
    inference_ms: float = Field(..., ge=0.0)
    device: str
    conf_threshold: float
    image_width: int = Field(..., ge=1)
    image_height: int = Field(..., ge=1)
    localization_status: str = Field(
        ..., description="Status of detection: 'success', 'no_detections', 'model_not_ready', or 'error'"
    )
    is_localized: bool = Field(..., description="True if at least one detection was found")
    detections: List[TyreDetection] = Field(default_factory=list)


# ── Detector ───────────────────────────────────────────────────────────────────
class TyreYOLODetector:
    """
    Standalone YOLO object detection service for tyre component inspection.

    Does NOT modify or depend on RimYOLODetector or HybridInspector.

    Usage:
        detector = TyreYOLODetector()
        result = detector.detect("path/to/tyre.jpg")
        for det in result.detections:
            print(det.class_name, det.confidence, det.bbox)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = DEFAULT_CONF_THRESH,
        imgsz: int = DEFAULT_IMGSZ,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_TYRE_YOLO_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz
        self.device_str = "cuda" if torch.cuda.is_available() else "cpu"

        self._model = None
        self._class_names: Dict[int, str] = TYRE_YOLO_CLASS_NAMES.copy()

        self._load_model()

    # ── Loading ────────────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Load the YOLO model once using the Ultralytics API."""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("ultralytics package is not installed.")
            return

        if not self.model_path.exists():
            logger.warning("Tyre YOLO model path not found: %s", self.model_path)
            return

        try:
            model = YOLO(str(self.model_path))
            if hasattr(model, "names") and isinstance(model.names, dict):
                self._class_names = {int(k): str(v) for k, v in model.names.items()}
            
            # Record actual device if available
            if hasattr(model, "device"):
                self.device_str = str(model.device)
            elif torch.cuda.is_available():
                self.device_str = "cuda"
            else:
                self.device_str = "cpu"

            self._model = model
            logger.info(
                "TyreYOLODetector loaded successfully: model=%s, classes=%d, device=%s",
                self.model_path.name,
                len(self._class_names),
                self.device_str,
            )
        except Exception as exc:
            logger.error("Failed to load Tyre YOLO model from %s: %s", self.model_path, exc, exc_info=True)
            self._model = None

    @property
    def is_ready(self) -> bool:
        """Return True if model is loaded and ready for inference."""
        return self._model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        """Return dictionary of class ID to class name mappings."""
        return self._class_names

    # ── Bounding Box Validation ────────────────────────────────────────────────
    @staticmethod
    def validate_bbox(bbox: List[float], img_w: int, img_h: int) -> List[float]:
        """
        Validate and clamp bounding box [x1, y1, x2, y2] to original image boundaries.
        Ensures non-negative coordinates, x2 >= x1, and y2 >= y1.
        """
        if len(bbox) != 4:
            raise ValueError(f"Bbox must contain 4 elements, got {len(bbox)}")

        x1, y1, x2, y2 = bbox
        
        # Clamp to image boundaries
        x1 = max(0.0, min(float(x1), float(img_w)))
        y1 = max(0.0, min(float(y1), float(img_h)))
        x2 = max(0.0, min(float(x2), float(img_w)))
        y2 = max(0.0, min(float(y2), float(img_h)))

        # Ensure valid non-inverted coordinates
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    # ── Public API ─────────────────────────────────────────────────────────────
    def detect(
        self,
        image_input: Union[str, Path, Image.Image],
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> TyreYOLOInferenceResult:
        """
        Run Tyre YOLO object detection on an image.

        Args:
            image_input: File path (str/Path) or PIL Image.
            conf_threshold: Optional confidence threshold override.
            imgsz: Optional image inference resolution override.

        Returns:
            TyreYOLOInferenceResult with structured detections, bboxes,
            original image dimensions, latency, and device info.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image cannot be read or path is invalid.
        """
        if not self.is_ready:
            raise RuntimeError(
                "TyreYOLODetector model is not available. "
                f"Check that model exists at {self.model_path}"
            )

        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        inference_imgsz = imgsz if imgsz is not None else self.imgsz

        # ── Resolve image input ────────────────────────────────────────────────
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            if not img_path.exists():
                raise ValueError(f"Image path does not exist: {image_input}")
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image file '{image_input}': {e}") from e
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(
                f"Unsupported image input type: {type(image_input).__name__}"
            )

        orig_w, orig_h = pil_img.size

        # ── Inference ──────────────────────────────────────────────────────────
        t0 = time.time()
        try:
            results = self._model.predict(
                source=pil_img,
                imgsz=inference_imgsz,
                conf=conf,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Inference execution error in TyreYOLODetector: %s", exc, exc_info=True)
            raise RuntimeError(f"YOLO inference failed: {exc}") from exc

        latency_ms = round((time.time() - t0) * 1000, 2)

        # ── Parse Detections ───────────────────────────────────────────────────
        detections: List[TyreDetection] = []
        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy_tensor = boxes.xyxy.cpu()
                conf_tensor = boxes.conf.cpu()
                cls_tensor = boxes.cls.cpu()

                for i in range(len(boxes)):
                    raw_bbox = xyxy_tensor[i].tolist()
                    validated_bbox = self.validate_bbox(raw_bbox, orig_w, orig_h)
                    confidence = round(float(conf_tensor[i].item()), 4)
                    class_id = int(cls_tensor[i].item())
                    class_name = self._class_names.get(class_id, f"Unknown_{class_id}")

                    detections.append(
                        TyreDetection(
                            class_id=class_id,
                            class_name=class_name,
                            defect_type=class_name,
                            confidence=confidence,
                            bbox=validated_bbox,
                        )
                    )

        num_dets = len(detections)
        status = "success" if num_dets > 0 else "no_detections"

        return TyreYOLOInferenceResult(
            model="tyre_yolo_v1",
            num_detections=num_dets,
            inference_ms=latency_ms,
            device=self.device_str,
            conf_threshold=conf,
            image_width=orig_w,
            image_height=orig_h,
            localization_status=status,
            is_localized=num_dets > 0,
            detections=detections,
        )

    async def detect_async(
        self,
        image_input: Union[str, Path, Image.Image],
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> TyreYOLOInferenceResult:
        """Asynchronous wrapper for detect() running in an executor thread."""
        import asyncio
        return await asyncio.to_thread(
            self.detect,
            image_input,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
tyre_yolo_detector = TyreYOLODetector()
