# Aluminium Wheel Defect Severity Classification Plan

**Owner**: Vijeath (Data + AI/ML Lead)

## 1. Objective
Assess the severity of detected defects on aluminium alloy automotive wheels into two primary operational tiers:
- **Low**: Superficial imperfections, cosmetic blemishes, minor machining burrs within rework limits.
- **Critical**: Structural cracks, shrinkage porosity in the wheel spoke or hub mounting face, severe voids requiring immediate scrapping of the wheel.

## 2. Evaluation Criteria
- Recall for `Critical` severity must approach 1.0 (zero-escape tolerance for safety-critical wheel defects).
- Confusion matrix and F1-score across validation sets.

## 3. Current Status
- SKELETON / FOUNDATION PHASE: Framework defined.
- `TODO — DECISION REQUIRED`: Determine whether severity thresholds depend on defect location on the wheel (e.g., rim lip vs wheel spoke vs hub face) or defect physical area.
