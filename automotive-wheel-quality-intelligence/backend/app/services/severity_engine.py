# ============================================================
# Q-SENTINEL — SEVERITY ENGINE
# Automotive Alloy Wheel Defect Assessment
# ============================================================

"""
Prototype severity engine for the Singularity 2026 hackathon.

IMPORTANT:
These scores and thresholds are prototype assumptions for
hackathon/demo purposes. They are NOT manufacturer-certified
safety limits.

Severity is calculated from:
    1. Base defect severity
    2. Defect area ratio
    3. Morphology

AI confidence is NOT used to calculate physical severity.
It is used separately when calculating inspection priority.
"""

from __future__ import annotations

from typing import Literal


# ------------------------------------------------------------
# 1. BASE SEVERITY BY DEFECT TYPE
# ------------------------------------------------------------

BASE_SEVERITY: dict[str, int] = {
    "bent_rim": 65,
    "blow_hole": 60,
    "crack": 75,
    "incomplete_welding": 70,
    "porosity": 55,
    "scratch": 15,
    "scuff": 20,
    "paint_damage": 20,

    # Rare / future classes
    "chip": 25,
    "scuff_severe": 35,
    "corrosion": 40,
    "buckle": 70,
    "mixed_porosity_blowhole": 65,
}


# ------------------------------------------------------------
# 2. MORPHOLOGY MODIFIER
# ------------------------------------------------------------

MORPHOLOGY_MODIFIER: dict[str, int] = {
    "minor": 0,
    "moderate": 8,
    "severe": 15,
}


# ------------------------------------------------------------
# 3. SIZE MODIFIER
# ------------------------------------------------------------

def calculate_size_modifier(area_ratio: float) -> int:
    """
    Calculate severity contribution from defect area.

    area_ratio is expressed as percentage of the component/image
    area occupied by the detected defect.

    Prototype assumptions:
        <= 0.5  -> 0
        <= 1.0  -> 5
        <= 2.0  -> 8
        >  2.0  -> 6 * area_ratio

    The result is capped at 25.

    This reproduces the supplied examples:
        0.3 -> 0
        1.2 -> 8
        3.0 -> 18
    """

    if area_ratio < 0:
        raise ValueError("area_ratio cannot be negative")

    if area_ratio <= 0.5:
        return 0

    if area_ratio <= 1.0:
        return 5

    if area_ratio <= 2.0:
        return 8

    return min(25, round(area_ratio * 6))


# ------------------------------------------------------------
# 4. SEVERITY LEVEL
# ------------------------------------------------------------

def severity_level_from_score(score: int) -> str:
    """
    Convert numerical severity score into dashboard level.
    """

    if score >= 90:
        return "Critical"

    if score >= 70:
        return "High"

    if score >= 40:
        return "Medium"

    return "Low"


# ------------------------------------------------------------
# 5. MAIN SEVERITY CALCULATION
# ------------------------------------------------------------

def calculate_severity(
    defect_type: str,
    area_ratio: float = 0.0,
    morphology: str = "minor",
) -> dict:
    """
    Calculate prototype defect severity.

    Parameters
    ----------
    defect_type:
        Normalized defect class.

    area_ratio:
        Defect area ratio / percentage used by the prototype
        severity calculation.

    morphology:
        One of:
            minor
            moderate
            severe

    Returns
    -------
    dict containing:
        defect_type
        base_score
        size_modifier
        morphology_modifier
        severity_score
        severity_level
    """

    defect_type = defect_type.strip().lower()
    morphology = morphology.strip().lower()

    if defect_type not in BASE_SEVERITY:
        raise ValueError(
            f"Unknown defect type: '{defect_type}'. "
            f"Supported types: {sorted(BASE_SEVERITY)}"
        )

    if morphology not in MORPHOLOGY_MODIFIER:
        raise ValueError(
            f"Unknown morphology: '{morphology}'. "
            f"Supported values: {sorted(MORPHOLOGY_MODIFIER)}"
        )

    base_score = BASE_SEVERITY[defect_type]

    size_modifier = calculate_size_modifier(area_ratio)

    morphology_modifier = MORPHOLOGY_MODIFIER[morphology]

    raw_score = (
        base_score
        + size_modifier
        + morphology_modifier
    )

    severity_score = min(100, raw_score)

    severity_level = severity_level_from_score(severity_score)

    return {
        "defect_type": defect_type,
        "base_score": base_score,
        "size_modifier": size_modifier,
        "morphology_modifier": morphology_modifier,
        "raw_score": raw_score,
        "severity_score": severity_score,
        "severity_level": severity_level,
    }


# ------------------------------------------------------------
# 6. INSPECTION PRIORITY
# ------------------------------------------------------------

def inspection_priority(
    severity_score: int,
    confidence: float,
) -> dict:
    """
    Determine inspection priority.

    Severity determines the physical risk category.
    Model confidence determines how confidently the AI has
    identified the defect.

    Confidence does NOT change the severity score.
    """

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    if not 0 <= severity_score <= 100:
        raise ValueError("severity_score must be between 0 and 100")

    if severity_score >= 90:
        priority = "CRITICAL"
    elif severity_score >= 70:
        priority = "HIGH"
    elif severity_score >= 40:
        priority = "MEDIUM"
    else:
        priority = "LOW"

    # Low-confidence predictions should be manually reviewed,
    # even when their calculated severity is relatively low.
    if confidence < 0.60:
        review_required = True
    else:
        review_required = False

    return {
        "priority": priority,
        "confidence": confidence,
        "review_required": review_required,
    }