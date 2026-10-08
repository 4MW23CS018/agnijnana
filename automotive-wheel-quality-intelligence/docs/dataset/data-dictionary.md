# Data Dictionary — Aluminium Wheel Quality Inspection

**Owner**: Vijeath (Data + AI/ML Lead)

> **Important**: This dictionary contains provisional field mappings. Exact process parameters, telemetry channels, defect classifications, and units MUST be verified against the selected aluminium wheel dataset before model training.

## 1. Wheel Entity (Provisional)
| Field | Type | Description | Status |
| :--- | :--- | :--- | :--- |
| `wheel_id` | String | Unique tracking identifier for each wheel / rim | Provisional |
| `wheel_model` | String | Aluminium wheel design / specification code | `TODO — DECISION REQUIRED` |
| `batch_id` | String | Casting / heat-treatment batch identifier | Provisional |
| `machine_id` | String | Die casting machine or CNC station identifier | Provisional |
| `production_cycle` | Integer / String | Machine shot or cycle index | Provisional |
| `inspection_timestamp` | Timestamp | Timestamp when the wheel was imaged | Provisional |
| `image_uri` | String | Filepath or object store URI of wheel image | Provisional |

## 2. Wheel Defect Entity (Provisional)
| Field | Type | Description | Status |
| :--- | :--- | :--- | :--- |
| `defect_id` | Integer / UUID | Identifier for each detected defect | Provisional |
| `wheel_id` | String | Foreign key to inspected wheel | Provisional |
| `defect_type` | String | Defect category from dataset (e.g. crack, porosity, void) | `TODO — DECISION REQUIRED: Confirm classes from dataset` |
| `location` | JSON / List | Spatial coordinates on wheel rim `[x, y, w, h]` or mask | Provisional |
| `severity` | Enum | Classification: `Low` (reworkable) vs `Critical` (scrap) | Provisional |
| `confidence` | Float | Model detection confidence `[0.0, 1.0]` | Provisional |

## 3. Process / Casting Parameters (To be populated post dataset inspection)
| Parameter Name | Expected Source | Unit | Status |
| :--- | :--- | :--- | :--- |
| *Casting parameters* | Machine telemetry | *TBD* | `TODO — DECISION REQUIRED: Populate from verified dataset` |
| *Thermal parameters* | Pyrometer / Sensor | *TBD* | `TODO — DECISION REQUIRED: Populate from verified dataset` |
| *Pressure parameters* | Hydraulic / Injection | *TBD* | `TODO — DECISION REQUIRED: Populate from verified dataset` |
| *Cycle time* | Machine PLC | *TBD* | `TODO — DECISION REQUIRED: Populate from verified dataset` |

*Note: No sensor fields will be hardcoded until Vijeath inspects and documents the chosen dataset.*
