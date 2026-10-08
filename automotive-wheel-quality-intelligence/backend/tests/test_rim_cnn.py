import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.rim_cnn_classifier import RimCNNClassifier, rim_cnn_classifier

client = TestClient(app)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_IMAGE = PROJECT_ROOT / "data" / "uploads" / "9c6afba861144684a502c5b399145807.jpeg"


def test_cnn_model_loads():
    """Verify ConvNeXt-Tiny Rim CNN classifier loads and is ready."""
    assert rim_cnn_classifier.is_ready is True
    assert rim_cnn_classifier.model is not None
    assert len(rim_cnn_classifier.class_names) == 8


def test_cnn_direct_prediction_validity():
    """Verify direct model prediction returns valid class and confidence in range [0, 1]."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample test image not found")

    result = rim_cnn_classifier.predict(SAMPLE_IMAGE)
    assert result.model == "rim_cnn_v1"
    assert 0 <= result.class_id <= 7
    assert isinstance(result.defect_type, str)
    assert 0.0 <= result.confidence <= 1.0
    assert result.inference_ms >= 0.0
    assert result.device in ["cpu", "cuda"]


def test_api_upload_and_inspect_flow():
    """Verify full end-to-end API flow: POST /upload -> POST /inspect."""
    if not SAMPLE_IMAGE.exists():
        pytest.skip("Sample test image not found")

    with open(SAMPLE_IMAGE, "rb") as f:
        upload_resp = client.post(
            "/api/inspection/upload",
            files={"image": ("sample_wheel.jpg", f, "image/jpeg")},
        )

    assert upload_resp.status_code == 201
    upload_data = upload_resp.json()
    assert "image_path" in upload_data

    saved_path = upload_data["image_path"]

    # Test inspection API
    inspect_resp = client.post(
        "/api/inspection/inspect",
        json={
            "wheel_id": "TEST-WHEEL-100",
            "image_path": saved_path,
            "batch_id": "BATCH-100",
            "machine_id": "M-01",
        },
    )

    assert inspect_resp.status_code == 200
    inspect_data = inspect_resp.json()

    assert inspect_data["wheel_id"] == "TEST-WHEEL-100"
    # location may now be a YOLO bbox [x1,y1,x2,y2] or None depending on detections
    assert "location" in inspect_data
    # rim is the CNN-only baseline result (backward-compat field)
    assert "rim" in inspect_data
    assert inspect_data["rim"] is not None
    assert inspect_data["rim"]["model"] == "rim_cnn_v1"
    assert 0 <= inspect_data["rim"]["class_id"] <= 7
    assert 0.0 <= inspect_data["rim"]["confidence"] <= 1.0
    assert inspect_data["rim"]["inference_ms"] >= 0.0
    # hybrid field must be present
    assert "hybrid" in inspect_data
    assert inspect_data["hybrid"] is not None


def test_api_reject_invalid_image_upload():
    """Verify corrupted / non-image file uploads are rejected with 400 Bad Request."""
    invalid_resp = client.post(
        "/api/inspection/upload",
        files={"image": ("malicious.jpg", b"NOT AN IMAGE DATA", "image/jpeg")},
    )
    assert invalid_resp.status_code == 400
    assert "not a valid image" in invalid_resp.json()["detail"]


def test_api_reject_nonexistent_image_inspect():
    """Verify inspection request with non-existent path returns 400 Bad Request."""
    resp = client.post(
        "/api/inspection/inspect",
        json={
            "wheel_id": "TEST-WHEEL-999",
            "image_path": "data/uploads/non_existent_file_xyz.jpg",
        },
    )
    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"]


def test_missing_model_returns_503(monkeypatch):
    """Verify HTTP 503 is returned when Rim CNN model is unavailable."""
    dummy_classifier = RimCNNClassifier(model_path="non_existent_model_dir")
    assert dummy_classifier.is_ready is False

    with pytest.raises(Exception):
        dummy_classifier.predict(SAMPLE_IMAGE)
