# ============================================================
# Q-SENTINEL — SEVERITY ENGINE
# Automotive Alloy Wheel Defect Assessment
# ============================================================

"""
Severity Engine for Automotive Alloy Wheel Inspection.

Provides deterministic, explainable severity assessment for detected rim defects.

PROVISIONAL ENGINEERING ASSUMPTIONS:
- Base severity scores reflect physical structural risk categories:
  - Critical structural defects: crack (75), incomplete_welding (70), bent_rim (65), blow_hole (60)
  - Cosmetic/surface defects: scratch (15), scuff (20), paint_damage (20), porosity (55)
- Severity levels:
  - Score >= 70 -> 'Critical' (endangers wheel structural integrity / vehicle safety)
  - Score < 70 -> 'Low' (cosmetic or minor surface defect)
- AI detection confidence is kept strictly separate from physical defect severity.
- Instance mask area ratio is used only when valid and available.
  Bounding box area is NOT used as actual defect area.

NOTE: These rules are provisional engineering heuristics for demonstration purposes,
not manufacturer-certified automotive safety standards.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Centralized Thresholds & Base Severities
CRITICAL_SEVERITY_THRESHOLD: int = 70

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

MORPHOLOGY_MODIFIER: dict[str, int] = {
    "minor": 0,
    "moderate": 8,
    "severe": 15,
}


def calculate_size_modifier(area_ratio: float) -> int:
    """
    Calculate severity contribution from defect mask area.

    area_ratio: percentage of image occupied by reconstructed defect instance mask.

    Rules:
        <= 0.5%  -> 0
        <= 1.0%  -> 5
        <= 2.0%  -> 8
        >  2.0%  -> round(area_ratio * 6), capped at 25.
    """
    if area_ratio < 0:
        return 0

    if area_ratio <= 0.5:
        return 0
    if area_ratio <= 1.0:
        return 5
    if area_ratio <= 2.0:
        return 8

    return min(25, round(area_ratio * 6))


def severity_level_from_score(score: int, mode: str = "binary") -> str:
    """
    Convert numerical severity score into severity level.

    mode='binary' (default):
        score >= 70 -> 'Critical'
        score < 70  -> 'Low'

    mode='tiered':
        score >= 90 -> 'Critical'
        score >= 70 -> 'High'
        score >= 40 -> 'Medium'
        score < 40  -> 'Low'
    """
    if mode == "tiered":
        if score >= 90:
            return "Critical"
        if score >= 70:
            return "High"
        if score >= 40:
            return "Medium"
        return "Low"

    # Default binary mode required by project specification (Low vs Critical)
    return "Critical" if score >= CRITICAL_SEVERITY_THRESHOLD else "Low"


def calculate_severity(
    defect_type: str,
    area_ratio: Optional[float] = None,
    mask_status: Optional[str] = None,
    morphology: str = "minor",
    confidence: Optional[float] = None,
    level_mode: str = "binary",
) -> dict[str, Any]:
    """
    Calculate defect severity level, score, and transparent rationale.

    Confidence is kept strictly separate from severity calculation.

    Returns dict with:
        defect_type, base_score, size_modifier, morphology_modifier,
        raw_score, severity_score, severity_level, rationale,
        mask_evidence_available
    """
    clean_defect = (defect_type or "").strip().lower()
    clean_morph = (morphology or "minor").strip().lower()

    if clean_defect not in BASE_SEVERITY:
        # Unknown or unsupported defect class — handle safely with explicit uncertainty
        return {
            "defect_type": defect_type,
            "base_score": 0,
            "size_modifier": 0,
            "morphology_modifier": 0,
            "raw_score": 0,
            "severity_score": 0,
            "severity_level": "Uncertain",
            "rationale": f"Unknown defect class '{defect_type}'. Physical severity cannot be reliably assessed.",
            "mask_evidence_available": False,
        }

    base_score = BASE_SEVERITY[clean_defect]
    morph_modifier = MORPHOLOGY_MODIFIER.get(clean_morph, 0)

    # Instance mask evidence evaluation
    mask_valid = (
        mask_status == "available"
        and area_ratio is not None
        and isinstance(area_ratio, (int, float))
        and area_ratio >= 0
    )

    if mask_valid:
        size_modifier = calculate_size_modifier(area_ratio)
        mask_evidence_note = f"Instance mask area ratio: {area_ratio:.4f}% (+{size_modifier} pts)."
    else:
        size_modifier = 0
        mask_evidence_note = "Instance mask unavailable; size modifier omitted (bbox area not used as physical mask area)."

    raw_score = base_score + size_modifier + morph_modifier
    severity_score = min(100, max(0, raw_score))
    severity_level = severity_level_from_score(severity_score, mode=level_mode)

    # Construct transparent human-readable explanation
    rationale_parts = [
        f"Defect '{clean_defect}' base structural risk: {base_score}/100.",
        mask_evidence_note,
    ]
    if morph_modifier > 0:
        rationale_parts.append(f"Morphology '{clean_morph}' (+{morph_modifier} pts).")

    if confidence is not None:
        rationale_parts.append(f"AI detection confidence: {confidence * 100:.1f}% (kept separate from physical severity).")

    rationale_parts.append(f"Calculated severity score: {severity_score}/100 → Level: {severity_level}.")

    return {
        "defect_type": clean_defect,
        "base_score": base_score,
        "size_modifier": size_modifier,
        "morphology_modifier": morph_modifier,
        "raw_score": raw_score,
        "severity_score": severity_score,
        "severity_level": severity_level,
        "rationale": " ".join(rationale_parts),
        "mask_evidence_available": mask_valid,
    }


def inspection_priority(
    severity_score: int,
    confidence: float,
) -> dict:
    """Determine inspection priority based on severity score and model confidence."""
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    if not 0 <= severity_score <= 100:
        raise ValueError("severity_score must be between 0 and 100")

    if severity_score >= 70:
        priority = "CRITICAL"
    elif severity_score >= 40:
        priority = "HIGH"
    else:
        priority = "LOW"

    review_required = confidence < 0.60

    return {
        "priority": priority,
        "confidence": confidence,
        "review_required": review_required,
    }