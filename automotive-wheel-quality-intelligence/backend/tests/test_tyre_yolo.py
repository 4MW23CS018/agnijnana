"""
Unit tests for TyreYOLODetector service.

Tests model loading, bounding box validation, missing model handling,
error handling, schema validation, and mock inference behavior.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image

from app.services.tyre_yolo_detector import (
    DEFAULT_CONF_THRESH,
    DEFAULT_IMGSZ,
    TYRE_YOLO_CLASS_NAMES,
    TyreDetection,
    TyreYOLODetector,
    TyreYOLOInferenceResult,
    tyre_yolo_detector,
)


def test_tyre_yolo_singleton_loads():
    """Verify default singleton detector loads successfully."""
    assert tyre_yolo_detector.is_ready is True
    assert isinstance(tyre_yolo_detector.class_names, dict)
    assert len(tyre_yolo_detector.class_names) == 14


def test_tyre_yolo_class_taxonomy():
    """Verify class taxonomy contains expected defect names."""
    names = tyre_yolo_detector.class_names
    assert names[0] == "Generic_Defect"
    assert names[1] == "Bulge"
    assert names[2] == "Cracks"
    assert names[4] == "Good"
    assert names[7] == "Bad_Tire"
    assert names[9] == "Normal_Tire"


def test_tyre_yolo_missing_model_returns_not_ready():
    """Verify detector handles non-existent model path gracefully without crashing."""
    detector = TyreYOLODetector(model_path="nonexistent_model_path.pt")
    assert detector.is_ready is False


def test_tyre_yolo_missing_model_raises_on_detect():
    """Verify detect() raises RuntimeError if model is not ready."""
    detector = TyreYOLODetector(model_path="nonexistent_model_path.pt")
    with pytest.raises(RuntimeError, match="model is not available"):
        detector.detect(Image.new("RGB", (100, 100)))


def test_tyre_yolo_nonexistent_image_raises_value_error():
    """Verify detect() raises ValueError for missing image file."""
    with pytest.raises(ValueError, match="Image path does not exist"):
        tyre_yolo_detector.detect("nonexistent_tyre_image_12345.jpg")


def test_tyre_yolo_invalid_image_type_raises_value_error():
    """Verify detect() raises ValueError for unsupported input types."""
    with pytest.raises(ValueError, match="Unsupported image input type"):
        tyre_yolo_detector.detect(12345)  # type: ignore


def test_bbox_validation():
    """Verify validate_bbox clamps coordinates and fixes inverted boxes."""
    # Out of bounds
    bbox = TyreYOLODetector.validate_bbox([-10, -5, 1200, 900], img_w=1000, img_h=800)
    assert bbox == [0.0, 0.0, 1000.0, 800.0]

    # Inverted coordinates
    bbox_inv = TyreYOLODetector.validate_bbox([500, 400, 100, 200], img_w=1000, img_h=800)
    assert bbox_inv == [100.0, 200.0, 500.0, 400.0]


def test_tyre_yolo_synthetic_inference_returns_valid_contract():
    """Verify detect() returns clean TyreYOLOInferenceResult for synthetic PIL image."""
    img = Image.new("RGB", (800, 600), color=(128, 128, 128))
    result = tyre_yolo_detector.detect(img)

    assert isinstance(result, TyreYOLOInferenceResult)
    assert result.model == "tyre_yolo_v1"
    assert result.image_width == 800
    assert result.image_height == 600
    assert result.conf_threshold == DEFAULT_CONF_THRESH
    assert result.localization_status in ("success", "no_detections")
    assert isinstance(result.inference_ms, float)
    assert result.inference_ms >= 0.0


def test_tyre_yolo_mock_detections_parsing():
    """Test detection parsing logic using mocked Ultralytics output without running real inference."""
    detector = TyreYOLODetector(model_path="nonexistent.pt")
    
    # Mock model
    mock_model = MagicMock()
    detector._model = mock_model
    detector._class_names = TYRE_YOLO_CLASS_NAMES

    # Create mock prediction boxes
    mock_boxes = MagicMock()
    mock_boxes.__len__.return_value = 1
    mock_boxes.xyxy.cpu.return_value = torch.tensor([[50.0, 60.0, 200.0, 250.0]])
    mock_boxes.conf.cpu.return_value = torch.tensor([0.885])
    mock_boxes.cls.cpu.return_value = torch.tensor([2])  # Class 2 = Cracks

    mock_result = MagicMock()
    mock_result.boxes = mock_boxes
    mock_model.predict.return_value = [mock_result]

    pil_img = Image.new("RGB", (640, 480))
    res = detector.detect(pil_img, conf_threshold=0.30)

    assert res.num_detections == 1
    assert res.is_localized is True
    assert res.localization_status == "success"
    assert res.conf_threshold == 0.30

    det = res.detections[0]
    assert isinstance(det, TyreDetection)
    assert det.class_id == 2
    assert det.class_name == "Cracks"
    assert det.defect_type == "Cracks"
    assert det.confidence == 0.885
    assert det.bbox == [50.0, 60.0, 200.0, 250.0]


def test_tyre_yolo_does_not_import_rim_or_hybrid():
    """Verify tyre_yolo_detector is isolated and does not import rim detection or hybrid inspection."""
    import sys
    import app.services.tyre_yolo_detector as tyre_module

    source_code = Path(tyre_module.__file__).read_text(encoding="utf-8")
    assert "rim_yolo_detector" not in source_code
    assert "rim_cnn_classifier" not in source_code
    assert "hybrid_inspection" not in source_code
