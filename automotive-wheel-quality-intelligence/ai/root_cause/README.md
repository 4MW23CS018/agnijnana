# Aluminium Wheel Casting Root-Cause Analysis Module

## 1. Module Overview
- **Owner**: Vijeath (Data + AI/ML Lead)
- **Purpose**: Correlate detected wheel defects with casting/machining process parameters to identify the probable manufacturing root cause and provide a confidence score.
- **Current Status**: SKELETON / FOUNDATION PHASE — No models trained or downloaded.

## 2. Interface Contracts
- **Expected Input**:
  - Defect information (`defect_type`, `severity`, `wheel_id`).
  - Process telemetry vector (fields TBD by dataset — do NOT assume specific sensors exist).
- **Expected Output**:
  - `root_cause`: String
  - `root_cause_confidence`: Float `[0.0, 1.0]`

## 3. Planned Evaluation Metrics
- Multi-class Accuracy / Top-2 Accuracy
- SHAP feature attribution stability

## 4. Subdirectories
- `preprocessing/` — Process telemetry ingestion and scaling
- `features/` — Feature engineering from casting cycle data
- `models/` — Root-cause classifiers
- `explainability/` — SHAP / LIME explainers
- `inference/` — Callable inference interface

## 5. Open Decisions
- `TODO — DECISION REQUIRED`: Define root-cause taxonomy after dataset inspection. No sensor fields or root causes will be fabricated.
