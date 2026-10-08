# Aluminium Wheel Defect Detection & Localization Plan

**Owner**: Vijeath (Data + AI/ML Lead)

## 1. Objective
Detect and localize surface irregularities, casting defects, and structural flaws on aluminium alloy automotive wheels and rims using computer vision models.

## 2. Target Outputs
- Defect presence (`true` / `false`).
- Defect class (derived strictly from dataset annotations — no hard-coded classes).
- Defect bounding box / polygon contour overlay on the wheel image.
- Localization confidence score `[0.0, 1.0]`.

## 3. Evaluation Metrics
- Mean Average Precision (mAP@0.5 and mAP@0.5:0.95).
- Intersection over Union (IoU).
- Per-class precision and recall.
- Inference latency per wheel inspection (< 60 ms edge target).

## 4. Current Status
- SKELETON / FOUNDATION PHASE: Architecture layout created.
- `TODO — DECISION REQUIRED`: Confirm exact defect classes from the dataset before defining model output heads.
