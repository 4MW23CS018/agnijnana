# Aluminium Wheel Casting Root-Cause Analysis (RCA) Plan

**Owner**: Vijeath (Data + AI/ML Lead)

## 1. Objective
Identify the probable manufacturing/process root cause of detected wheel defects using verified process data from the casting batch, machine state, and production cycle.

## 2. Output Contract
```json
{
  "root_cause": "...",
  "root_cause_confidence": 0.0
}
```

## 3. Methodological Approach
- Correlate defect occurrences with machine process telemetry (to be finalized based on dataset fields).
- Utilize explainable AI (SHAP / Feature Attribution) to identify top contributing parameters.

## 4. Current Status
- SKELETON / FOUNDATION PHASE.
- `TODO — DECISION REQUIRED`: Populate exact feature inputs once Vijeath verifies available telemetry fields from the chosen dataset. No sensor fields or root causes will be fabricated.
