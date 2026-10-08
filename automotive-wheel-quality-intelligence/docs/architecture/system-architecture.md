# System Architecture Specification — Aluminium Wheel Quality Intelligence

## 1. Overview
This Quality Intelligence platform is dedicated to **aluminium alloy automotive wheels and rims**. It unifies optical/X-ray inspection imagery with casting and machining process telemetry to provide closed-loop defect detection, root cause diagnosis, and predictive risk management.

```
┌────────────────────────────────────────────────────────┐
│                   Data Sources                         │
│  - Wheel Inspection Camera / X-ray Line Station        │
│  - Casting & Machining Machine Telemetry (SCADA / PLC) │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                  Raw Ingestion Layer                   │
│  - High-res wheel images + casting cycle parameters   │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│             AI / ML Quality Pipeline (Vijeath)         │
│  1. Wheel Defect Detection & Localization (Vision)     │
│  2. Defect Severity Classifier (Low vs Critical)       │
│  3. Root Cause Analysis (Process Parameter Correlation)│
│  4. Predictive Risk Forecasting (Machine / Batch Risk) │
│  5. Corrective Action Recommendation Engine            │
└──────────────────────────┬─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│            FastAPI Backend Service (Ashish)            │
│  - REST Endpoints (/inspection, /defects, /risk, etc.) │
│  - Common Wheel AI Output Contract Validation          │
│  - Database Transaction Management                     │
│  - Security & Authentication Middleware                │
└─────────────┬────────────────────────────┬─────────────┘
              │                            │
              ▼                            ▼
┌───────────────────────────┐  ┌─────────────────────────┐
│   PostgreSQL DB (Deepak)  │  │   React UI (Ganesh)     │
│  - Wheels & Casting Batches│  │  - Wheel Inspection Cockpit│
│  - Production Cycles      │  │  - Defect Localization View │
│  - Defect Logs & Telemetry│  │  - Severity & RCA Cards     │
│  - Predictions & Alerts   │  │  - Machine Risk Forecast    │
└───────────────────────────┘  └─────────────────────────┘
```

## 2. Component Boundaries
- **Frontend Layer**: Consumer of backend REST APIs. Specifically designed for aluminium wheel defect visualization, severity indicator, root cause breakdown, and batch alerts.
- **Backend API Layer**: Central orchestrator. Ingests wheel inspection requests, validates payloads, invokes AI models, records inspection results in PostgreSQL, and serves data to the frontend.
- **Database Layer**: Wheel-specific relational schema (`wheels`, `batches`, `machines`, `production_cycles`, `defects`, `process_data`, `predictions`, `recommendations`, `alerts`).
- **AI/ML Layer**: Independent Python modules with standardized inference contracts tailored to wheel inspection.

## 3. Decisions & Open Questions
- `TODO — DECISION REQUIRED`: Determine whether wheel images are visual RGB camera surface images, radioscopic/X-ray casting images, or a multimodal combination.
- `TODO — DECISION REQUIRED`: Decide on edge inference deployment target (NVIDIA Jetson / Industrial PC at wheel inspection station).
