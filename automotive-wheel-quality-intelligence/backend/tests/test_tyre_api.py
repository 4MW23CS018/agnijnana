"""
API tests for tyre inspection endpoint (`POST /api/inspection/inspect` with component="tyre").
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.tyre_yolo_detector import TyreDetection, TyreYOLOInferenceResult

client = TestClient(app)


def test_api_inspect_default_component_is_rim(tmp_path):
    """Verify inspection request defaults component to 'rim' when unspecified."""
    sample_img = tmp_path / "test_rim.jpg"
    from PIL import Image
    Image.new("RGB", (640, 480)).save(sample_img)

    payload = {
        "wheel_id": "WHEEL_TEST_001",
        "image_path": str(sample_img),
    }
    response = client.post("/api/inspection/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["component"] == "rim"
    assert "tyre" in data
    assert data["tyre"] is None
    assert "hybrid" in data or "rim" in data


def test_api_inspect_invalid_component(tmp_path):
    """Verify HTTP 400 returned for invalid component value."""
    sample_img = tmp_path / "test_img.jpg"
    from PIL import Image
    Image.new("RGB", (640, 480)).save(sample_img)

    payload = {
        "wheel_id": "WHEEL_TEST_002",
        "image_path": str(sample_img),
        "component": "invalid_component",
    }
    response = client.post("/api/inspection/inspect", json=payload)
    assert response.status_code == 400
    assert "Invalid component" in response.json()["detail"]


def test_api_inspect_tyre_success(tmp_path):
    """Verify POST /api/inspection/inspect with component='tyre' returns structured tyre response."""
    sample_img = tmp_path / "test_tyre.jpg"
    from PIL import Image
    Image.new("RGB", (800, 600), color=(100, 100, 100)).save(sample_img)

    payload = {
        "wheel_id": "TYRE_WHEEL_001",
        "image_path": str(sample_img),
        "component": "tyre",
    }
    response = client.post("/api/inspection/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["wheel_id"] == "TYRE_WHEEL_001"
    assert data["component"] == "tyre"
    assert data["hybrid"] is None
    assert data["rim"] is None
    assert data["tyre"] is not None

    tyre_data = data["tyre"]
    assert tyre_data["model"] == "tyre_yolo_v1"
    assert tyre_data["image_width"] == 800
    assert tyre_data["image_height"] == 600
    assert isinstance(tyre_data["inference_ms"], float)
    assert "localization_status" in tyre_data
    assert "is_localized" in tyre_data
    assert "detections" in tyre_data


def test_api_inspect_tyre_with_mocked_detections(tmp_path):
    """Verify tyre response structure when detections are present."""
    sample_img = tmp_path / "test_tyre_det.jpg"
    from PIL import Image
    Image.new("RGB", (1000, 800)).save(sample_img)

    mock_result = TyreYOLOInferenceResult(
        model="tyre_yolo_v1",
        num_detections=1,
        inference_ms=12.5,
        device="cpu",
        conf_threshold=0.25,
        image_width=1000,
        image_height=800,
        localization_status="success",
        is_localized=True,
        detections=[
            TyreDetection(
                class_id=2,
                class_name="Cracks",
                defect_type="Cracks",
                confidence=0.9123,
                bbox=[100.0, 150.0, 300.0, 400.0],
            )
        ],
    )

    with patch("app.api.inspection.tyre_yolo_detector.detect_async", return_value=mock_result):
        payload = {
            "wheel_id": "TYRE_WHEEL_002",
            "image_path": str(sample_img),
            "component": "tyre",
        }
        response = client.post("/api/inspection/inspect", json=payload)
        assert response.status_code == 200
        data = response.json()

        assert data["component"] == "tyre"
        assert data["defect_type"] == "Cracks"
        assert data["defect_confidence"] == 0.9123
        assert data["location"] == [100.0, 150.0, 300.0, 400.0]

        tyre_info = data["tyre"]
        assert tyre_info["num_detections"] == 1
        assert tyre_info["detections"][0]["class_name"] == "Cracks"
        assert tyre_info["detections"][0]["bbox"] == [100.0, 150.0, 300.0, 400.0]


def test_api_inspect_tyre_invalid_image_path():
    """Verify HTTP 400 returned when tyre inspection image path does not exist."""
    payload = {
        "wheel_id": "TYRE_WHEEL_ERR",
        "image_path": "nonexistent_tyre_img_9999.jpg",
        "component": "tyre",
    }
    response = client.post("/api/inspection/inspect", json=payload)
    assert response.status_code == 400
    assert "Image path does not exist" in response.json()["detail"]


def test_api_inspect_tyre_service_unavailable(tmp_path):
    """Verify HTTP 503 returned when tyre YOLO detector is not ready."""
    sample_img = tmp_path / "test_tyre.jpg"
    from PIL import Image
    Image.new("RGB", (640, 480)).save(sample_img)

    with patch("app.api.inspection.tyre_yolo_detector._model", None):
        payload = {
            "wheel_id": "TYRE_WHEEL_UNAVAIL",
            "image_path": str(sample_img),
            "component": "tyre",
        }
        response = client.post("/api/inspection/inspect", json=payload)
        assert response.status_code == 503
        assert "not available" in response.json()["detail"]
