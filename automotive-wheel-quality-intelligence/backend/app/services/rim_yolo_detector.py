"""
RimYOLODetector — Isolated YOLO segmentation inference service.

Model: rim_yolo26s_seg_v1.pt (YOLO26s-seg / SegmentationModel)
Task:  Instance segmentation

Output geometry per detection:
    bbox        = Object localization — bounding box [x1,y1,x2,y2] in original px coords.
    mask        = Pixel-level defect localization — reconstructed binary mask, summarised as:
                      mask_area_pixels : int   — foreground pixel count inside the bbox
                      mask_area_ratio  : float — mask area as % of FULL IMAGE area (image-relative,
                                                  NOT wheel-relative; document this to caller)
                      mask_polygon     : list  — largest contour [[x,y],...] in original px coords

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

Mask reconstruction (YOLO26s-seg):
    The model outputs two tensors:
        predictions  [1, 4+nc+32, 8400]  — anchors with class probs & 32-dim mask coefficients
        proto_masks  [1, 32, mask_h, mask_w]  — prototype mask bank (typically 160×160)

    Per detection:
        coeff         = 32-dim coefficient vector for this anchor
        proto_flat    = proto reshaped to [32, mask_h * mask_w]
        mask_logits   = coeff @ proto_flat          → [mask_h * mask_w]
        mask_sigmoid  = sigmoid(mask_logits)        → [mask_h, mask_w]
        binary_mask   = mask_sigmoid >= 0.5

    Resize binary_mask: proto resolution → 640×640 → original image size.
    Crop to the detection bbox (already in original-image coordinates).
    Count foreground pixels → mask_area_pixels, mask_area_ratio.
    Extract largest contour → mask_polygon.

Severity is NOT computed here.
Severity is the responsibility of severity_engine.py.

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

import cv2
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
YOLO_INPUT_SIZE: int = 640        # Standard YOLO input resolution (px)
DEFAULT_CONF_THRESH: float = 0.10  # Conservative; model is lightly trained
DEFAULT_IOU_THRESH: float = 0.45
MASK_BINARIZE_THRESH: float = 0.5  # sigmoid threshold for binary mask

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
    """
    Single detection returned by RimYOLODetector.

    Geometry fields:
        bbox         — Object localization: [x1, y1, x2, y2] in original image px.
        mask_*       — Pixel-level defect localization derived from the reconstructed
                        instance segmentation mask.

    mask_area_ratio is image-relative (mask px / total image px × 100),
    NOT wheel-relative. The caller is responsible for any wheel-area normalisation.
    """

    class_id: int = Field(..., description="YOLO class ID (0=bent_rim, 1=crack, 2=scratch)")
    defect_type: str = Field(..., description="Human-readable defect name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence [0, 1]")

    bbox: List[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Bounding box [x1, y1, x2, y2] in original image pixel coords",
    )

    # ── Mask presence ──────────────────────────────────────────────────────────
    mask_available: bool = Field(
        ...,
        description=(
            "True if mask reconstruction succeeded. "
            "False if proto was unavailable or malformed — bbox is still valid."
        ),
    )

    # ── Raw coefficients (kept for downstream use) ─────────────────────────────
    mask_coefficients: Optional[List[float]] = Field(
        default=None,
        description=(
            "32-dimensional mask coefficient vector. "
            "Combine with proto matrix to reconstruct the instance mask."
        ),
    )

    # ── Reconstructed mask summary ─────────────────────────────────────────────
    mask_area_pixels: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of foreground pixels in the reconstructed defect mask (inside bbox).",
    )

    mask_area_ratio: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description=(
            "Defect mask area as percentage of the FULL IMAGE area "
            "(image-relative, NOT wheel-relative). "
            "mask_area_pixels / (orig_w * orig_h) * 100."
        ),
    )

    mask_polygon: Optional[List[List[float]]] = Field(
        default=None,
        description=(
            "Largest segmentation contour as [[x, y], ...] in original image coordinates. "
            "Suitable for SVG/canvas polygon rendering. "
            "None if no contour was found."
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

    Per-detection output:
        bbox        — object localization (always valid when confidence passes threshold)
        mask_*      — pixel-level defect localization (valid when proto tensor is available)

    Does NOT use the CNN classifier (rim_cnn_classifier.py).
    Does NOT perform severity, root cause, or any downstream analysis.

    Usage:
        detector = RimYOLODetector()
        result = detector.detect("path/to/rim.jpg")
        for det in result.detections:
            print(det.defect_type, det.confidence, det.bbox)
            print(det.mask_area_pixels, det.mask_area_ratio)
            print(det.mask_polygon)
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
        """Resize to 640×640, normalize to [0,1], return [1,3,640,640] float32.

        No letterboxing — direct resize. Coordinate mapping:
            x_orig = x_640 * (orig_w / 640)
            y_orig = y_640 * (orig_h / 640)
        """
        img = pil_img.resize((YOLO_INPUT_SIZE, YOLO_INPUT_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0  # HWC
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1CHW
        return tensor

    # ── Mask reconstruction ────────────────────────────────────────────────────
    def _reconstruct_instance_mask(
        self,
        mask_coefficients: List[float],
        proto_masks: torch.Tensor,
        bbox: List[float],
        orig_w: int,
        orig_h: int,
    ):
        """
        Reconstruct a binary instance segmentation mask from YOLO mask coefficients
        and the prototype mask tensor.

        Args:
            mask_coefficients : 32-dim coefficient list for this detection.
            proto_masks       : Prototype tensor, shape [1, 32, mask_h, mask_w].
            bbox              : [x1, y1, x2, y2] in original image coordinates.
            orig_w, orig_h    : Original image dimensions (pixels).

        Returns:
            Tuple of:
                binary_mask        (np.ndarray uint8, shape [orig_h, orig_w])
                mask_area_pixels   (int)
                mask_area_ratio    (float, image-relative percentage)
                mask_polygon       (List[List[float]] or None)

        If reconstruction fails for any reason, returns (None, None, None, None).
        """
        try:
            # ── 1. Validate proto shape ────────────────────────────────────────
            if proto_masks is None:
                raise ValueError("proto_masks is None")

            # Squeeze batch dim if present: [1, 32, mh, mw] → [32, mh, mw]
            proto = proto_masks.squeeze(0)  # [32, mask_h, mask_w]
            if proto.ndim != 3 or proto.shape[0] != 32:
                raise ValueError(f"Unexpected proto shape: {list(proto.shape)}")

            n_proto, mask_h, mask_w = proto.shape

            # ── 2. Compute mask logits ─────────────────────────────────────────
            coeff = torch.tensor(mask_coefficients, dtype=torch.float32)  # [32]
            proto_flat = proto.reshape(n_proto, mask_h * mask_w)           # [32, mh*mw]
            mask_logits = coeff @ proto_flat                                # [mh*mw]
            mask_sigmoid = torch.sigmoid(mask_logits).reshape(mask_h, mask_w)  # [mh, mw]

            # ── 3. Binarise ────────────────────────────────────────────────────
            binary_proto = (mask_sigmoid >= MASK_BINARIZE_THRESH).numpy().astype(np.uint8)

            # ── 4. Resize: proto resolution → 640×640 → original ──────────────
            # Step A: proto → 640×640
            mask_640 = cv2.resize(
                binary_proto,
                (YOLO_INPUT_SIZE, YOLO_INPUT_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )
            # Step B: 640×640 → original image size
            mask_orig = cv2.resize(
                mask_640,
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            )

            # ── 5. Crop to bbox ────────────────────────────────────────────────
            x1, y1, x2, y2 = bbox
            bx1 = max(0, int(round(x1)))
            by1 = max(0, int(round(y1)))
            bx2 = min(orig_w, int(round(x2)))
            by2 = min(orig_h, int(round(y2)))

            # Zero out pixels outside the bbox
            cropped_mask = np.zeros_like(mask_orig, dtype=np.uint8)
            if bx2 > bx1 and by2 > by1:
                cropped_mask[by1:by2, bx1:bx2] = mask_orig[by1:by2, bx1:bx2]

            # ── 6. Mask area ───────────────────────────────────────────────────
            mask_area_pixels = int(cropped_mask.sum())
            image_area = orig_w * orig_h
            mask_area_ratio = round((mask_area_pixels / image_area) * 100, 4) if image_area > 0 else 0.0

            # ── 7. Polygon (largest contour) ───────────────────────────────────
            contours, _ = cv2.findContours(
                cropped_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            mask_polygon: Optional[List[List[float]]] = None
            if contours:
                # Pick the largest contour by area
                largest = max(contours, key=cv2.contourArea)
                # largest shape: [N, 1, 2] → [[x,y], ...]
                pts = largest.squeeze(axis=1)
                if pts.ndim == 2 and len(pts) >= 3:
                    mask_polygon = [
                        [round(float(pt[0]), 1), round(float(pt[1]), 1)]
                        for pt in pts
                    ]

            return cropped_mask, mask_area_pixels, mask_area_ratio, mask_polygon

        except Exception as exc:
            logger.warning(
                "Mask reconstruction failed (bbox will still be valid): %s", exc
            )
            return None, None, None, None

    # ── Post-processing / NMS ──────────────────────────────────────────────────
    def _apply_nms(
        self,
        raw_pred: torch.Tensor,
        proto_masks: Optional[torch.Tensor],
        orig_w: int,
        orig_h: int,
    ) -> List[YOLODetection]:
        """
        Apply NMS to raw YOLO output tensor and return structured detections.

        raw_pred shape: [1, 4+nc+32, 8400]
          - 4  = cx, cy, w, h (xywh, in 640px coords)
          - nc = class probability per class (3)
          - 32 = mask coefficients

        proto_masks shape: [1, 32, mask_h, mask_w]
            Used for per-detection mask reconstruction.
            If None/malformed, bbox detections are still returned (mask_available=False).

        Returns list of YOLODetection with bbox scaled to original image size,
        and reconstructed mask geometry when proto is available.
        """
        # [1, 39, 8400] → [8400, 39]
        preds = raw_pred.squeeze(0).T

        boxes_xywh  = preds[:, :4]               # [8400, 4]
        class_probs = preds[:, 4:4 + self._nc]   # [8400, nc]
        mask_coeffs = preds[:, 4 + self._nc:]    # [8400, 32]

        # Best class and confidence per anchor
        conf, cls_ids = class_probs.max(dim=1)   # [8400], [8400]

        above_thresh = conf > self.conf_threshold
        if above_thresh.sum() == 0:
            return []

        # Filter candidates
        f_boxes = boxes_xywh[above_thresh]
        f_conf  = conf[above_thresh]
        f_cls   = cls_ids[above_thresh]
        f_masks = mask_coeffs[above_thresh]

        # xywh → xyxy (still in 640px space)
        cx, cy, w, h = f_boxes[:, 0], f_boxes[:, 1], f_boxes[:, 2], f_boxes[:, 3]
        xyxy = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)

        # NMS
        keep = tv_nms(xyxy, f_conf, iou_threshold=self.iou_threshold)

        # Scale factors: 640px → original image coordinates
        sx = orig_w / YOLO_INPUT_SIZE
        sy = orig_h / YOLO_INPUT_SIZE

        detections: List[YOLODetection] = []
        for i in keep:
            cls_id   = int(f_cls[i].item())
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

            # 32-dim mask coefficient vector
            mask_coeff_list = [round(float(v), 6) for v in f_masks[i].tolist()]

            # ── Reconstruct instance mask ──────────────────────────────────────
            _, mask_area_pixels, mask_area_ratio, mask_polygon = \
                self._reconstruct_instance_mask(
                    mask_coefficients=mask_coeff_list,
                    proto_masks=proto_masks,
                    bbox=bbox,
                    orig_w=orig_w,
                    orig_h=orig_h,
                )

            mask_reconstructed = mask_area_pixels is not None

            detections.append(
                YOLODetection(
                    class_id=cls_id,
                    defect_type=cls_name,
                    confidence=confidence,
                    bbox=bbox,
                    mask_available=mask_reconstructed,
                    mask_coefficients=mask_coeff_list,
                    mask_area_pixels=mask_area_pixels,
                    mask_area_ratio=mask_area_ratio,
                    mask_polygon=mask_polygon,
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

        Returns YOLOInferenceResult with per-detection:
            bbox            — object localization [x1,y1,x2,y2] in original px
            mask_available  — True if mask reconstruction succeeded
            mask_coefficients — raw 32-dim coefficients
            mask_area_pixels  — foreground pixel count inside bbox
            mask_area_ratio   — mask area as % of full image area (image-relative)
            mask_polygon      — largest contour [[x,y],...] in original px coords

        If proto_masks are absent/malformed, detections are still returned with
        mask_available=False and mask geometry fields set to None.

        Args:
            image_input: File path (str/Path) or PIL Image.

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
        # Expected structure from YOLO26s-seg:
        #   outputs       → tuple(predictions_group, proto_masks)
        #   outputs[0]    → tuple/list containing the detection tensor
        #   outputs[0][0] → [1, 4+nc+32, 8400]  detection predictions
        #   outputs[0][1] → [1, 32, mask_h, mask_w]  prototype masks
        #
        # Handle defensively — structure may vary with ultralytics version.
        raw_preds: Optional[torch.Tensor] = None
        proto_masks: Optional[torch.Tensor] = None

        try:
            if isinstance(outputs, (list, tuple)) and len(outputs) >= 1:
                group0 = outputs[0]
                if isinstance(group0, (list, tuple)):
                    # outputs[0] is itself a tuple of (det_tensor, proto_tensor)
                    raw_preds = group0[0] if len(group0) >= 1 else None
                    proto_masks = group0[1] if len(group0) >= 2 else None
                else:
                    # outputs[0] is the detection tensor directly
                    raw_preds = group0
                    # Proto might be outputs[1]
                    if isinstance(outputs, (list, tuple)) and len(outputs) >= 2:
                        proto_masks = outputs[1]
            else:
                raw_preds = outputs
        except Exception as parse_exc:
            logger.warning("Output parsing issue: %s — will attempt raw_preds fallback", parse_exc)
            raw_preds = outputs if isinstance(outputs, torch.Tensor) else None

        if raw_preds is None:
            logger.error("Could not extract detection tensor from model outputs")
            return YOLOInferenceResult(
                model="rim_yolo26s_seg_v1",
                num_detections=0,
                inference_ms=latency_ms,
                device=self.device_str,
                conf_threshold=self.conf_threshold,
                detections=[],
            )

        # ── NMS + mask reconstruction ──────────────────────────────────────────
        detections = self._apply_nms(raw_preds, proto_masks, orig_w, orig_h)

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
