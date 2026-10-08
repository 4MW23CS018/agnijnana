# REST API Specification — Aluminium Wheel Quality Intelligence

**Owner**: Ashish (Backend Lead)

Base URL: `http://localhost:8000`

## 1. System Health
- **`GET /health`**
  - Returns service status and API version.

## 2. Wheel Inspection (`/api/inspection` & `/inspection`)
- **`POST /api/inspection/inspect`**
  - Description: Submit aluminium alloy wheel image and metadata (wheel_id, batch_id, machine_id, cycle).
  - Returns: Common Wheel AI Output Contract.
- **`GET /api/inspection/{wheel_id}`**
  - Description: Retrieve completed inspection report for a specific wheel.

## 3. Wheel Defects (`/api/defects`)
- **`GET /api/defects`**
  - Description: Retrieve list of defects detected on wheel rims. Filters: `batch_id`, `machine_id`, `severity`.
- **`GET /api/defects/{defect_id}`**
  - Description: Fetch defect localization coordinates and metadata.

## 4. Root Cause Analysis (`/api/root-cause`)
- **`GET /api/root-cause`**
  - Description: List historical root cause diagnoses.
- **`POST /api/root-cause/analyze`**
  - Description: Trigger process parameter correlation for a specific wheel defect.

## 5. Defect Risk Prediction (`/api/risk`)
- **`GET /api/risk`**
  - Description: Current defect risk scores across casting machines and batches.
- **`GET /api/risk/{machine_id}`**
  - Description: Risk forecast for a specific die-casting machine.

## 6. Corrective Recommendations (`/api/recommendations`)
- **`GET /api/recommendations`**
  - Description: Active industrial corrective actions for casting machines.

## 7. Alerts (`/api/alerts`)
- **`GET /api/alerts`**
  - Description: Active quality alerts and batch quarantine warnings.
- **`POST /api/alerts/{alert_id}/acknowledge`**
  - Description: Operator acknowledgment of quality alert.

## 8. Batches & Machines (`/api/batches`, `/api/machines`)
- **`GET /api/batches`**: List casting batches and lot progress.
- **`GET /api/machines`**: List casting and machining equipment statuses.

## 9. Common Wheel AI Output Contract
```json
{
  "wheel_id": "WHL-2026-0042",
  "batch_id": "CB-2026-03",
  "machine_id": "M-04",
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
```
*Note: Contract is ready for model integration once datasets are validated.*
