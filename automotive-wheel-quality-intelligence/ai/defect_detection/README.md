# Aluminium Wheel Defect Detection & Localization Module

## 1. Module Overview
- **Owner**: Vijeath (Data + AI/ML Lead)
- **Purpose**: Detect, classify, and localize casting/surface defects on aluminium alloy automotive wheels and rims using computer vision.
- **Current Status**: SKELETON / FOUNDATION PHASE — No models trained or downloaded.

## 2. Interface Contracts
- **Expected Input**:
  - Wheel rim image (RGB or X-ray, format TBD by dataset).
  - Wheel context: `wheel_id`, `batch_id`, `machine_id`, `production_cycle`.
- **Expected Output**:
  - `defect_detected`: Boolean
  - `defect_type`: String (from dataset — do NOT hard-code classes until verified)
  - `location`: Bounding box `[x, y, w, h]` or segmentation mask
  - `defect_confidence`: Float `[0.0, 1.0]`

## 3. Planned Evaluation Metrics
- mAP@0.5, mAP@0.5:0.95
- Per-class Precision, Recall, IoU
- Inference latency per wheel (target < 60 ms GPU)

## 4. Subdirectories
- `data/` — Wheel dataset splits and annotation caches
- `models/` — Exported model checkpoints
- `training/` — Training scripts and hyperparameters
- `inference/` — Inference engine callable by FastAPI
- `evaluation/` — Benchmark scripts

## 5. Open Decisions
- `TODO — DECISION REQUIRED`: Select model architecture after verifying wheel dataset annotation format.
- `TODO — DECISION REQUIRED`: Confirm defect class taxonomy from dataset labels.
