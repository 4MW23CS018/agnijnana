# Corrective Action Recommendation Module

## 1. Module Overview
- **Owner**: Vijeath / Shared with Ashish
- **Purpose**: Map detected wheel defects, severities, root causes, and risk forecasts to actionable corrective maintenance procedures for the aluminium wheel casting and machining line.
- **Current Status**: SKELETON / FOUNDATION PHASE — No recommendations engine implemented.

## 2. Interface Contracts
- **Expected Input**:
  - `defect_type`, `severity`, `root_cause`, `future_risk`
- **Expected Output**:
  - `recommended_action`: String (e.g., "Inspect and recalibrate the casting temperature control system for Machine M-04 before the next production cycle.")
  - `action_priority`: `"IMMEDIATE"` | `"SCHEDULED"` | `"MONITOR"`

## 3. Subdirectories
- `rules/` — Deterministic decision tables for casting defect → corrective action mapping
- `engine/` — Recommendation arbitration logic

## 4. Open Decisions
- `TODO — DECISION REQUIRED`: Define corrective action vocabulary after process data and defect types are confirmed.
