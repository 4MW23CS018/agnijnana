from app.services.severity_engine import (
    calculate_severity,
    inspection_priority,
)


def test_scratch_low_severity():
    result = calculate_severity(
        defect_type="scratch",
        area_ratio=0.3,
        morphology="minor",
    )

    assert result["base_score"] == 15
    assert result["size_modifier"] == 0
    assert result["morphology_modifier"] == 0
    assert result["severity_score"] == 15
    assert result["severity_level"] == "Low"


def test_crack_critical_severity():
    result = calculate_severity(
        defect_type="crack",
        area_ratio=3.0,
        morphology="moderate",
    )

    assert result["base_score"] == 75
    assert result["size_modifier"] == 18
    assert result["morphology_modifier"] == 8
    assert result["severity_score"] == 100
    assert result["severity_level"] == "Critical"


def test_porosity_high_severity():
    result = calculate_severity(
        defect_type="porosity",
        area_ratio=1.2,
        morphology="moderate",
    )

    assert result["base_score"] == 55
    assert result["size_modifier"] == 8
    assert result["morphology_modifier"] == 8
    assert result["severity_score"] == 71
    assert result["severity_level"] == "High"


def test_high_confidence_priority():
    priority = inspection_priority(
        severity_score=71,
        confidence=0.91,
    )

    assert priority["priority"] == "HIGH"
    assert priority["review_required"] is False


def test_low_confidence_requires_review():
    priority = inspection_priority(
        severity_score=71,
        confidence=0.45,
    )

    assert priority["priority"] == "HIGH"
    assert priority["review_required"] is True


def test_unknown_defect_rejected():
    try:
        calculate_severity(
            defect_type="unknown_defect",
            area_ratio=1.0,
            morphology="minor",
        )
        assert False
    except ValueError:
        assert True