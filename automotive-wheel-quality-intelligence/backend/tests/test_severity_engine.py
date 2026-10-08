"""
Unit and API integration tests for Severity Engine.
"""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.severity_engine import (
    calculate_severity,
    calculate_size_modifier,
    inspection_priority,
    severity_level_from_score,
)
from app.schemas.quality import WheelAIOutputContract

client = TestClient(app)


def test_scratch_low_severity():
    """Verify scratch produces Low severity with 15 base score."""
    result = calculate_severity(
        defect_type="scratch",
        area_ratio=0.3,
        mask_status="available",
        morphology="minor",
    )

    assert result["base_score"] == 15
    assert result["size_modifier"] == 0
    assert result["morphology_modifier"] == 0
    assert result["severity_score"] == 15
    assert result["severity_level"] == "Low"
    assert "Defect 'scratch' base structural risk: 15/100" in result["rationale"]


def test_crack_critical_severity():
    """Verify crack produces Critical severity."""
    result = calculate_severity(
        defect_type="crack",
        area_ratio=3.0,
        mask_status="available",
        morphology="moderate",
    )

    assert result["base_score"] == 75
    assert result["size_modifier"] == 18
    assert result["morphology_modifier"] == 8
    assert result["severity_score"] == 100
    assert result["severity_level"] == "Critical"
    assert "Defect 'crack' base structural risk: 75/100" in result["rationale"]


def test_exact_crack_calculation_trace():
    """Audit test: Trace exact calculation for a crack with base score 75 and mask area ratio 3.14%."""
    res = calculate_severity(
        defect_type="crack",
        area_ratio=3.14,
        mask_status="available",
        morphology="minor",
    )

    # 3.14 * 6 = 18.84 -> round(18.84) = 19
    assert res["base_score"] == 75
    assert res["size_modifier"] == 19
    assert res["morphology_modifier"] == 0
    assert res["raw_score"] == 94
    assert res["severity_score"] == 94
    assert res["severity_level"] == "Critical"
    assert res["mask_evidence_available"] is True
    assert "Defect 'crack' base structural risk: 75/100." in res["rationale"]
    assert "Instance mask area ratio: 3.1400% (+19 pts)." in res["rationale"]
    assert "Calculated severity score: 94/100 → Level: Critical." in res["rationale"]


def test_boundary_thresholds_for_size_modifier():
    """Audit test: Verify exact boundary threshold calculations for mask area ratios."""
    # <= 0.5% -> 0
    assert calculate_size_modifier(0.0) == 0
    assert calculate_size_modifier(0.5) == 0
    assert calculate_size_modifier(0.50) == 0

    # <= 1.0% -> 5
    assert calculate_size_modifier(0.51) == 5
    assert calculate_size_modifier(0.99) == 5
    assert calculate_size_modifier(1.0) == 5

    # <= 2.0% -> 8
    assert calculate_size_modifier(1.01) == 8
    assert calculate_size_modifier(1.99) == 8
    assert calculate_size_modifier(2.0) == 8

    # > 2.0% -> round(area_ratio * 6), capped at 25
    assert calculate_size_modifier(2.01) == 12  # round(2.01 * 6 = 12.06) = 12
    assert calculate_size_modifier(3.14) == 19  # round(3.14 * 6 = 18.84) = 19
    assert calculate_size_modifier(4.1667) == 25 # round(4.1667 * 6 = 25.0002) = 25
    assert calculate_size_modifier(5.0) == 25   # capped at 25


def test_porosity_severity_levels():
    """Verify severity level modes (binary vs tiered)."""
    # Tiered mode (backwards compat)
    res_tiered = calculate_severity(
        defect_type="porosity",
        area_ratio=1.2,
        mask_status="available",
        morphology="moderate",
        level_mode="tiered",
    )
    assert res_tiered["severity_score"] == 71
    assert res_tiered["severity_level"] == "High"

    # Binary mode (default / project required Low vs Critical)
    res_binary = calculate_severity(
        defect_type="porosity",
        area_ratio=1.2,
        mask_status="available",
        morphology="moderate",
        level_mode="binary",
    )
    assert res_binary["severity_score"] == 71
    assert res_binary["severity_level"] == "Critical"


def test_high_detection_confidence_with_low_severity():
    """Verify high AI detection confidence does NOT artificially inflate physical severity score."""
    result = calculate_severity(
        defect_type="scratch",
        area_ratio=0.2,
        mask_status="available",
        confidence=0.99,
    )
    assert result["severity_score"] == 15
    assert result["severity_level"] == "Low"
    assert "AI detection confidence: 99.0% (kept separate" in result["rationale"]


def test_low_detection_confidence_with_critical_defect():
    """Verify low AI detection confidence retains physical Critical severity for serious defects."""
    result = calculate_severity(
        defect_type="crack",
        area_ratio=0.1,
        mask_status="available",
        confidence=0.42,
    )
    assert result["severity_score"] == 75
    assert result["severity_level"] == "Critical"
    assert "AI detection confidence: 42.0% (kept separate" in result["rationale"]


def test_missing_mask_metadata_handled_safely():
    """Verify missing/unavailable mask does not use bbox area and omits size modifier safely."""
    result = calculate_severity(
        defect_type="bent_rim",
        area_ratio=None,
        mask_status="unavailable",
    )
    assert result["base_score"] == 65
    assert result["size_modifier"] == 0
    assert result["severity_score"] == 65
    assert result["severity_level"] == "Low"
    assert result["mask_evidence_available"] is False
    assert "Instance mask unavailable; size modifier omitted" in result["rationale"]


def test_invalid_area_ratio_handled_safely():
    """Verify negative area ratio is handled gracefully without crashing."""
    result = calculate_severity(
        defect_type="scuff",
        area_ratio=-5.0,
        mask_status="available",
    )
    assert result["size_modifier"] == 0
    assert result["severity_level"] == "Low"


def test_unknown_defect_class_handled_safely():
    """Verify unknown defect class returns Uncertain level with safe rationale."""
    result = calculate_severity(
        defect_type="unknown_alien_defect",
        area_ratio=1.0,
    )
    assert result["severity_level"] == "Uncertain"
    assert result["severity_score"] == 0
    assert "Unknown defect class 'unknown_alien_defect'" in result["rationale"]


def test_high_confidence_priority():
    priority = inspection_priority(
        severity_score=71,
        confidence=0.91,
    )
    assert priority["priority"] == "CRITICAL"
    assert priority["review_required"] is False


def test_low_confidence_requires_review():
    priority = inspection_priority(
        severity_score=71,
        confidence=0.45,
    )
    assert priority["priority"] == "CRITICAL"
    assert priority["review_required"] is True


def test_api_rim_severity_integration(tmp_path):
    """Verify POST /api/inspection/inspect for rim populates top-level and localized severity fields."""
    sample_img = tmp_path / "test_rim_sev.jpg"
    from PIL import Image
    Image.new("RGB", (640, 480)).save(sample_img)

    payload = {
        "wheel_id": "WHEEL_SEV_001",
        "image_path": str(sample_img),
        "component": "rim",
    }
    response = client.post("/api/inspection/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["component"] == "rim"
    assert data["severity"] in ("Low", "Critical", "Uncertain")
    assert isinstance(data["severity_score"], int)
    assert isinstance(data["severity_rationale"], str)
    assert "base structural risk" in data["severity_rationale"]


def test_api_tyre_severity_isolation(tmp_path):
    """Verify POST /api/inspection/inspect for tyre keeps severity fields unset (None)."""
    sample_img = tmp_path / "test_tyre_sev.jpg"
    from PIL import Image
    Image.new("RGB", (800, 600)).save(sample_img)

    payload = {
        "wheel_id": "TYRE_SEV_001",
        "image_path": str(sample_img),
        "component": "tyre",
    }
    response = client.post("/api/inspection/inspect", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["component"] == "tyre"
    assert data["severity"] is None
    assert data["severity_score"] is None
    assert data["severity_rationale"] is None