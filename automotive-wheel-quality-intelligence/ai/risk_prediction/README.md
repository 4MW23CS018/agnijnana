# Aluminium Wheel Machine & Batch Risk Prediction Module

## 1. Module Overview
- **Owner**: Vijeath (Data + AI/ML Lead)
- **Purpose**: Forecast defect risk for die-casting machines, casting batches, and upcoming production cycles based on historical wheel inspection outcomes and process drift.
- **Current Status**: SKELETON / FOUNDATION PHASE — No models trained or downloaded.

## 2. Interface Contracts
- **Expected Input**:
  - Historical defect records per machine/batch.
  - Process telemetry time-series (fields TBD by dataset).
- **Expected Output**:
  - `machine_id`: String
  - `future_risk`: Float `[0.0, 1.0]`
  - `affected_batches`: List of batch IDs at elevated risk

## 3. Planned Evaluation Metrics
- ROC-AUC, PR-AUC
- Brier Score for probability calibration

## 4. Subdirectories
- `preprocessing/` — Temporal sequence alignment
- `features/` — Windowed aggregations and drift indicators
- `models/` — Risk forecasting models
- `evaluation/` — Backtesting scripts
- `inference/` — Inference adapter for backend

## 5. Open Decisions
- `TODO — DECISION REQUIRED`: Define forecast horizon (next cycle vs next batch vs next shift).
