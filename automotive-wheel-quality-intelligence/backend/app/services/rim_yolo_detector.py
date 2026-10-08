"""
RimYOLODetector — Isolated YOLO segmentation inference service.

Model: rim_yolo26s_seg_v1.pt (YOLO26s-seg / SegmentationModel)
Task:  Instance segmentation — bounding boxes + mask coefficients
Classes (3):
    0: bent_rim
    1: crack
    2: scratch

Loading strategy:
  The model is stored as an unzipped PyTorch zip directory.
  We reconstruct the zip in-memory, then load via torch.load().
  YOLO() from ultralytics cannot open a directory path, so we
  bypass it and call the inner nn.Module directly.

Float32 safety:
  Weights are stored as float16 (half precision). On CPU they must
  be converted to float32 before inference or a HalfTensor/FloatTensor
  mismatch RuntimeError will be raised.

NOTE: This service is completely independent of rim_cnn_classifier.py.
      Do NOT import or reference the CNN service from here.
"""

import io
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from pydantic import BaseModel, Field
from torchvision.ops import nms as tv_nms

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_YOLO_MODEL_PATH = (
    PROJECT_ROOT / "ai" / "defect_detection" / "models" / "rim_yolo26s_seg_v1.pt"
)

# ── Constants ──────────────────────────────────────────────────────────────────
YOLO_INPUT_SIZE: int = 640       # Standard YOLO input resolution
DEFAULT_CONF_THRESH: float = 0.10  # Conservative threshold; model is lightly trained
DEFAULT_IOU_THRESH: float = 0.45

# ── YOLO class taxonomy (3 classes) ───────────────────────────────────────────
YOLO_CLASS_NAMES: Dict[int, str] = {
    0: "bent_rim",
    1: "crack",
    2: "scratch",
}

# CNN covers 8 classes; these are NOT supported by this YOLO model:
YOLO_UNSUPPORTED_CLASSES = [
    "blow_hole",
    "incomplete_welding",
    "paint_damage",
    "porosity",
    "scuff",
]


# ── Output schemas ─────────────────────────────────────────────────────────────
class YOLODetection(BaseModel):
    """Single detection returned by RimYOLODetector."""

    class_id: int = Field(..., description="YOLO class ID (0=bent_rim, 1=crack, 2=scratch)")
    defect_type: str = Field(..., description="Human-readable defect name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence [0, 1]")
    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in original image pixel coords",
    )
    mask_available: bool = Field(
        ...,
        description="True if mask coefficients are present for this detection",
    )
    mask_coefficients: Optional[List[float]] = Field(
        default=None,
        description=(
            "32-dimensional mask coefficient vector. "
            "Combine with proto matrix to reconstruct the instance mask."
        ),
    )


class YOLOInferenceResult(BaseModel):
    """Full YOLO inference result for one image."""

    model: str = "rim_yolo26s_seg_v1"
    num_detections: int = Field(..., ge=0)
    inference_ms: float = Field(..., ge=0.0)
    device: str
    conf_threshold: float
    detections: List[YOLODetection] = Field(default_factory=list)


# ── Detector ───────────────────────────────────────────────────────────────────
class RimYOLODetector:
    """
    Isolated YOLO segmentation service for aluminium rim defect localization.

    Supports 3 YOLO classes: bent_rim (0), crack (1), scratch (2).

    Does NOT use the CNN classifier (rim_cnn_classifier.py).
    Does NOT perform severity, root cause, or any downstream analysis.

    Usage:
        detector = RimYOLODetector()
        result = detector.detect("path/to/rim.jpg")
        for det in result.detections:
            print(det.defect_type, det.confidence, det.bbox)
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = DEFAULT_CONF_THRESH,
        iou_threshold: float = DEFAULT_IOU_THRESH,
    ):
        self.model_path = Path(model_path) if model_path else DEFAULT_YOLO_MODEL_PATH
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_str = str(self.device.type)

        self._nn_model: Optional[torch.nn.Module] = None
        self._class_names: Dict[int, str] = {}
        self._nc: int = 0

        self._load_model()

    # ── Loading ────────────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """
        Load the YOLO SegmentationModel from the unzipped directory format.

        Strategy:
          1. Walk the 'best/' sub-directory of the model directory.
          2. Re-pack all files into an in-memory zip under the 'archive/' prefix
             that PyTorch zip format expects.
          3. Load via torch.load(buf, weights_only=False).
          4. Convert weights to float32 for CPU-safe inference.
        """
        if not self.model_path.exists():
            logger.warning("YOLO model path not found: %s", self.model_path)
            return

        # The stored structure is: rim_yolo26s_seg_v1.pt/best/<tensor files>
        best_dir = self.model_path / "best" if self.model_path.is_dir() else self.model_path
        if not best_dir.exists():
            logger.warning("YOLO model 'best' dir not found: %s", best_dir)
            return

        try:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
                for root, _dirs, files in os.walk(best_dir):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        rel = os.path.relpath(full_path, best_dir).replace("\\", "/")
                        zf.write(full_path, arcname=f"archive/{rel}")
            buf.seek(0)

            ckpt = torch.load(buf, weights_only=False, map_location=self.device)

            nn_model = ckpt.get("model")
            if nn_model is None:
                raise ValueError("Checkpoint missing 'model' key")

            # Convert to float32 — weights may be stored as float16 (half).
            # On CPU, half-precision inference raises RuntimeError.
            nn_model = nn_model.float().eval().to(self.device)

            self._nn_model = nn_model
            self._class_names = nn_model.names  # {0: 'bent_rim', 1: 'crack', 2: 'scratch'}
            self._nc = len(self._class_names)

            logger.info(
                "RimYOLODetector loaded: %s classes=%s device=%s",
                type(nn_model).__name__,
                self._class_names,
                self.device_str,
            )

        except Exception as exc:
            logger.error("RimYOLODetector failed to load: %s", exc, exc_info=True)
            self._nn_model = None

    @property
    def is_ready(self) -> bool:
        return self._nn_model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    # ── Preprocessing ──────────────────────────────────────────────────────────
    @staticmethod
    def _preprocess(pil_img: Image.Image) -> torch.Tensor:
        """Resize to 640×640, normalize to [0,1], return [1,3,640,640] float32."""
        img = pil_img.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0  # HWC
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
        return tensor

    # ── Post-processing / NMS ──────────────────────────────────────────────────
    def _apply_nms(
        self,
        raw_pred: torch.Tensor,
        orig_w: int,
        orig_h: int,
    ) -> List[YOLODetection]:
        """
        Apply NMS to raw YOLO output tensor and return structured detections.

        raw_pred shape: [1, 4+nc+32, 8400]
          - 4       = cx, cy, w, h (xywh format, in 640px coords)
          - nc      = class probability per class (3)
          - 32      = mask coefficients

        Returns list of YOLODetection objects with bbox scaled to original image size.
        """
        # [1, 39, 8400] -> [8400, 39]
        preds = raw_pred.squeeze(0).T

        boxes_xywh = preds[:, :4]          # [8400, 4]
        class_probs = preds[:, 4:4 + self._nc]  # [8400, nc]
        mask_coeffs = preds[:, 4 + self._nc:]   # [8400, 32]

        # Best class and confidence per anchor
        conf, cls_ids = class_probs.max(dim=1)  # [8400], [8400]

        above_thresh = conf > self.conf_threshold
        if above_thresh.sum() == 0:
            return []

        # Filter
        f_boxes = boxes_xywh[above_thresh]
        f_conf = conf[above_thresh]
        f_cls = cls_ids[above_thresh]
        f_masks = mask_coeffs[above_thresh]

        # xywh -> xyxy (still in 640px space)
        cx, cy, w, h = f_boxes[:, 0], f_boxes[:, 1], f_boxes[:, 2], f_boxes[:, 3]
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        # NMS
        keep = tv_nms(xyxy, f_conf, iou_threshold=self.iou_threshold)

        # Scale factors from 640px back to original image size
        sx = orig_w / YOLO_INPUT_SIZE
        sy = orig_h / YOLO_INPUT_SIZE

        detections: List[YOLODetection] = []
        for i in keep:
            cls_id = int(f_cls[i].item())
            cls_name = self._class_names.get(cls_id, f"unknown_{cls_id}")
            confidence = round(float(f_conf[i].item()), 4)

            # Scale bbox to original image coordinates
            x1, y1, x2, y2 = xyxy[i].tolist()
            bbox = [
                round(x1 * sx, 1),
                round(y1 * sy, 1),
                round(x2 * sx, 1),
                round(y2 * sy, 1),
            ]

            # Mask coefficients (32-dim vector for instance segmentation)
            mask_coeff_list = [round(float(v), 6) for v in f_masks[i].tolist()]

            detections.append(
                YOLODetection(
                    class_id=cls_id,
                    defect_type=cls_name,
                    confidence=confidence,
                    bbox=bbox,
                    mask_available=True,
                    mask_coefficients=mask_coeff_list,
                )
            )

        return detections

    # ── Public API ─────────────────────────────────────────────────────────────
    def detect(
        self,
        image_input: Union[str, Path, Image.Image],
    ) -> YOLOInferenceResult:
        """
        Run YOLO segmentation on a rim image.

        Args:
            image_input: File path (str/Path) or PIL Image.

        Returns:
            YOLOInferenceResult with detections, bboxes, mask coefficients,
            inference latency, and device info.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If image cannot be read.
        """
        if not self.is_ready:
            raise RuntimeError(
                "RimYOLODetector model is not available. "
                "Check that rim_yolo26s_seg_v1.pt exists at the expected path."
            )

        # ── Load image ─────────────────────────────────────────────────────────
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.is_absolute():
                img_path = PROJECT_ROOT / img_path
            if not img_path.exists():
                raise ValueError(f"Image path does not exist: {image_input}")
            try:
                pil_img = Image.open(img_path).convert("RGB")
            except Exception as e:
                raise ValueError(f"Invalid image file: {e}") from e
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(
                f"Unsupported image input type: {type(image_input).__name__}"
            )

        orig_w, orig_h = pil_img.size

        # ── Preprocess ─────────────────────────────────────────────────────────
        tensor = self._preprocess(pil_img).to(self.device)

        # ── Inference ──────────────────────────────────────────────────────────
        t0 = time.time()
        with torch.inference_mode():
            outputs = self._nn_model(tensor)
        latency_ms = round((time.time() - t0) * 1000, 2)

        # ── Parse raw output ───────────────────────────────────────────────────
        # outputs is a tuple: (predictions, proto_masks)
        # predictions is a list/tuple: [det_tensor, ...]
        # det_tensor shape: [1, 4+nc+32, 8400]
        if isinstance(outputs, (list, tuple)):
            raw_preds = outputs[0]
            if isinstance(raw_preds, (list, tuple)):
                raw_preds = raw_preds[0]
        else:
            raw_preds = outputs

        detections = self._apply_nms(raw_preds, orig_w, orig_h)

        return YOLOInferenceResult(
            model="rim_yolo26s_seg_v1",
            num_detections=len(detections),
            inference_ms=latency_ms,
            device=self.device_str,
            conf_threshold=self.conf_threshold,
            detections=detections,
        )


# ── Module-level singleton ─────────────────────────────────────────────────────
rim_yolo_detector = RimYOLODetector()
