# Aluminium Wheel Defect Severity Classification Module

## 1. Module Overview
- **Owner**: Vijeath (Data + AI/ML Lead)
- **Purpose**: Classify detected wheel defects into `Low` (cosmetic, reworkable) or `Critical` (structural, safety-critical — wheel must be scrapped).
- **Current Status**: SKELETON / FOUNDATION PHASE — No models trained or downloaded.

## 2. Interface Contracts
- **Expected Input**:
  - Defect type, spatial dimensions, and confidence from defect detection.
  - Wheel zone information (e.g., spoke, rim lip, hub face) if available.
- **Expected Output**:
  - `severity`: `"Low"` | `"Critical"`
  - `severity_confidence`: Float `[0.0, 1.0]`

## 3. Planned Evaluation Metrics
- Recall for `Critical` >= 0.98 (safety-critical zero-escape target)
- Precision, F1-score, Confusion Matrix

## 4. Subdirectories
- `rules/` — Deterministic severity criteria (e.g., defect in spoke → Critical)
- `models/` — ML classifiers for borderline severity cases
- `inference/` — Unified interface for backend integration

## 5. Open Decisions
- `TODO — DECISION REQUIRED`: Determine whether severity is rule-based, model-based, or hybrid after dataset review.
