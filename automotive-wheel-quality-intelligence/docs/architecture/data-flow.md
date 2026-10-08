# Aluminium Wheel Quality Data Flow Specification

## 1. End-to-End Wheel Inspection Pipeline

```
[Aluminium Wheel / Rim Image + Casting Process Parameters]
                           │
                           ▼
                 [Data Preprocessing]
     (Image normalization, process feature scaling)
                           │
                           ▼
                [AI Model Pipeline]
 ├─ Vision: Detect wheel defects, classify type, localize bounding box
 ├─ Severity: Map defect to Low vs Critical
 ├─ Root Cause: Correlate with casting batch & machine parameters
 └─ Risk: Predict next cycle / machine defect risk probability
                           │
                           ▼
         [Common Wheel AI Output Contract]
  {
    "wheel_id": "WHL-2026-0881",
    "batch_id": "CB-2026-04",
    "machine_id": "CASTING-M04",
    "defect_type": "...",
    "location": "...",
    "severity": "Low | Critical",
    "defect_confidence": 0.0,
    "root_cause": "...",
    "root_cause_confidence": 0.0,
    "future_risk": 0.0,
    "recommended_action": "...",
    "affected_batches": []
  }
                           │
                           ▼
          [FastAPI Backend (/api/inspection)]
                           │
                           ▼
            [PostgreSQL Database Transactions]
 (Save wheel, link batch/machine/cycle, insert defect & prediction)
                           │
                           ▼
            [React Quality Intelligence Cockpit]
  1. What went wrong? (Defect detected)
  2. Where is it? (Wheel rim defect bounding box overlay)
  3. How severe is it? (Severity badge: Low vs Critical)
  4. Why probably did it happen? (Probable root cause + confidence)
  5. What is likely to go wrong next? (Future machine / batch risk)
  6. What should the engineer do? (Actionable corrective recommendation)
```

## 2. Open Decisions
- `TODO — DECISION REQUIRED`: Standardize image coordinate system for wheel rims (polar/radial vs Cartesian bounding boxes).
- `TODO — DECISION REQUIRED`: Define ingestion protocol for production line cycles (HTTP POST per wheel vs batch polling).
