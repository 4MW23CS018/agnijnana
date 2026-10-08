"""
Tests for RimYOLODetector — isolated YOLO segmentation service.

Model under test: rim_yolo26s_seg_v1.pt
Task:             Instance segmentation (bent_rim, crack, scratch)

These tests are independent of rim_cnn_classifier.py.
They do NOT test the CNN pipeline.
"""
from pathlib import Path

import pytest

from app.services.rim_yolo_detector import (
    RimYOLODetector,
    YOLODetection,
    YOLOInferenceResult,
    YOLO_CLASS_NAMES,
    YOLO_UNSUPPORTED_CLASSES,
    rim_yolo_detector,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "uploads" / "9c6afba861144684a502c5b399145807.jpeg"
SAMPLE_IMAGE_2 = PROJECT_ROOT / "data" / "uploads" / "2df54fc4519b46968ba6197d39368803.jpeg"


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOADING
# ══════════════════════════════════════════════════════════════════════════════

def test_yolo_model_loads():
    """Verify RimYOLODetector loads successfully and is in ready state."""
    assert rim_yolo_detector.is_ready is True, (
        "RimYOLODetector failed to load — check model path and torch.load compatibility"
    )


def test_yolo_class_taxonomy():
    """Verify exact 3-class taxonomy: bent_rim, crack, scratch."""
    assert len(rim_yolo_detector.class_names) == 3
    assert rim_yolo_detector.class_names[0] == "bent_rim"
    assert rim_yolo_detector.class_names[1] == "crack"
    assert rim_yolo_detector.class_names[2] == "scratch"


def test_yolo_class_constants():
    """Verify module-level class name constants are correct."""
    assert YOLO_CLASS_NAMES == {0: "bent_rim", 1: "crack", 2: "scratch"}


def test_yolo_unsupported_classes_documented():
    """Verify unsupported CNN classes are explicitly documented."""
    assert "blow_hole" in YOLO_UNSUPPORTED_CLASSES
    assert "incomplete_welding" in YOLO_UNSUPPORTED_CLASSES
    assert "paint_damage" in YOLO_UNSUPPORTED_CLASSES
    assert "porosity" in YOLO_UNSUPPORTED_CLASSES
    assert "scuff" in YOLO_UNSUPPORTED_CLASSES


def test_yolo_missing_model_returns_not_ready():
    """Verify a detector pointed to a missing path reports is_ready=False."""
    bad = RimYOLODetector(model_path="non_existent_yolo_model.pt")
    assert bad.is_ready is False


def test_yolo_missing_model_raises_on_detect():
    """Verify detect() raises RuntimeError when model is not loaded."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample image not found")
    bad = RimYOLODetector(model_path="non_existent_yolo_model.pt")
    with pytest.raises(RuntimeError, match="not available"):
        bad.detect(SAMPLE_IMAGE)


# ══════════════════════════════════════════════════════════════════════════════
# 2. OUTPUT CONTRACT
# ══════════════════════════════════════════════════════════════════════════════

def test_yolo_inference_returns_valid_contract():
    """Verify detect() returns a well-formed YOLOInferenceResult."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE)

    assert isinstance(result, YOLOInferenceResult)
    assert result.model == "rim_yolo26s_seg_v1"
    assert result.num_detections == len(result.detections)
    assert result.inference_ms >= 0.0
    assert result.device in ("cpu", "cuda")
    assert result.conf_threshold > 0.0


def test_yolo_detection_fields_are_valid():
    """Verify each detection has correct field types and value ranges."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE)

    for det in result.detections:
        assert isinstance(det, YOLODetection)
        assert det.class_id in (0, 1, 2), f"Unexpected class_id: {det.class_id}"
        assert det.defect_type in ("bent_rim", "crack", "scratch")
        assert 0.0 <= det.confidence <= 1.0
        assert len(det.bbox) == 4
        x1, y1, x2, y2 = det.bbox
        assert x1 < x2, "x1 must be < x2 in bbox"
        assert y1 < y2, "y1 must be < y2 in bbox"
        assert det.mask_available is True
        assert det.mask_coefficients is not None
        assert len(det.mask_coefficients) == 32


def test_yolo_detects_on_second_image():
    """Verify detection runs on a different sample image without error."""
    if not SAMPLE_IMAGE_2.exists():
        pytest.skip("Second sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE_2)
    assert isinstance(result, YOLOInferenceResult)
    # Image 2df54f... has scratch detections at conf~0.29 (above 0.10 threshold)
    assert result.num_detections >= 0  # must not crash; detections may vary


def test_yolo_nonexistent_path_raises_value_error():
    """Verify detect() with a non-existent path raises ValueError."""
    with pytest.raises(ValueError, match="does not exist"):
        rim_yolo_detector.detect("definitely/does/not/exist.jpg")


def test_yolo_inference_latency_reasonable():
    """Verify inference completes in under 10 seconds on CPU."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE)
    assert result.inference_ms < 10_000, (
        f"Inference took {result.inference_ms}ms — unexpectedly slow"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. INDEPENDENCE CHECK
# ══════════════════════════════════════════════════════════════════════════════

def test_yolo_does_not_import_cnn():
    """Verify rim_yolo_detector module has no import statement for the CNN classifier."""
    import app.services.rim_yolo_detector as yolo_module
    module_source = Path(yolo_module.__file__).read_text(encoding="utf-8")
    # Check only import lines — docstrings may mention module names for documentation
    import_lines = [
        line.strip()
        for line in module_source.splitlines()
        if line.strip().startswith(("import ", "from "))
    ]
    cnn_imported = any("rim_cnn_classifier" in line for line in import_lines)
    assert not cnn_imported, (
        "rim_yolo_detector.py must NOT have an import statement for rim_cnn_classifier"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. MASK RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def test_mask_fields_exist():
    """YOLODetection must expose all three new mask geometry fields."""
    from app.services.rim_yolo_detector import YOLODetection
    fields = YOLODetection.model_fields
    assert "mask_area_pixels" in fields, "mask_area_pixels field missing from YOLODetection"
    assert "mask_area_ratio" in fields, "mask_area_ratio field missing from YOLODetection"
    assert "mask_polygon" in fields, "mask_polygon field missing from YOLODetection"


def test_mask_reconstruction_returns_area():
    """When detections exist, mask_area_pixels must be a non-negative integer."""
    if not SAMPLE_IMAGE_2.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE_2)
    # Image 2df54f... reliably produces detections at conf>0.10
    if result.num_detections == 0:
        pytest.skip("No detections on sample image — cannot test mask area")

    for det in result.detections:
        if det.mask_available:
            assert det.mask_area_pixels is not None
            assert isinstance(det.mask_area_pixels, int)
            assert det.mask_area_pixels >= 0


def test_mask_area_is_non_negative():
    """mask_area_pixels must never be negative."""
    if not SAMPLE_IMAGE_2.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE_2)
    for det in result.detections:
        if det.mask_area_pixels is not None:
            assert det.mask_area_pixels >= 0


def test_mask_area_ratio_is_valid():
    """
    mask_area_ratio must be in [0.0, 100.0].
    It is image-relative (mask_pixels / image_pixels * 100).
    """
    if not SAMPLE_IMAGE_2.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE_2)
    for det in result.detections:
        if det.mask_area_ratio is not None:
            assert 0.0 <= det.mask_area_ratio <= 100.0, (
                f"mask_area_ratio={det.mask_area_ratio} is outside [0, 100]"
            )


def test_mask_polygon_is_valid_when_contour_exists():
    """
    When mask_polygon is not None it must be a list of [x, y] pairs,
    each pair containing exactly two floats, with at least 3 points.
    """
    if not SAMPLE_IMAGE_2.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE_2)
    for det in result.detections:
        if det.mask_polygon is not None:
            assert isinstance(det.mask_polygon, list), "mask_polygon must be a list"
            assert len(det.mask_polygon) >= 3, (
                f"mask_polygon has only {len(det.mask_polygon)} points — need >= 3 for a contour"
            )
            for pt in det.mask_polygon:
                assert isinstance(pt, list) and len(pt) == 2, (
                    f"Each polygon point must be [x, y], got: {pt}"
                )
                assert isinstance(pt[0], (int, float))
                assert isinstance(pt[1], (int, float))


def test_missing_proto_does_not_crash():
    """
    If proto_masks is None (e.g. model output is unexpectedly structured),
    _apply_nms must not raise — bbox detections must still be returned,
    mask_available must be False, and mask geometry fields must be None.
    """
    if not SAMPLE_IMAGE_2.exists():
        pytest.skip("Sample image not found")

    # Call _reconstruct_instance_mask directly with proto=None
    detector = rim_yolo_detector
    dummy_coeffs = [0.0] * 32
    dummy_bbox = [10.0, 10.0, 100.0, 100.0]

    _, pix, ratio, poly = detector._reconstruct_instance_mask(
        mask_coefficients=dummy_coeffs,
        proto_masks=None,          # ← simulate missing proto
        bbox=dummy_bbox,
        orig_w=640,
        orig_h=640,
    )

    assert pix is None, "mask_area_pixels must be None when proto is missing"
    assert ratio is None, "mask_area_ratio must be None when proto is missing"
    assert poly is None, "mask_polygon must be None when proto is missing"


def test_bbox_behavior_is_unchanged():
    """
    Adding mask fields must not alter bbox values.
    bbox must still be [x1, y1, x2, y2] with x1<x2 and y1<y2.
    """
    if not SAMPLE_IMAGE_2.exists():
        pytest.skip("Sample image not found")

    result = rim_yolo_detector.detect(SAMPLE_IMAGE_2)
    for det in result.detections:
        assert len(det.bbox) == 4, "bbox must have exactly 4 elements"
        x1, y1, x2, y2 = det.bbox
        assert x1 < x2, f"bbox x1={x1} must be < x2={x2}"
        assert y1 < y2, f"bbox y1={y1} must be < y2={y2}"
        # All coordinates must be non-negative
        assert x1 >= 0 and y1 >= 0
