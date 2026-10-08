# PostgreSQL Database Architecture

## 1. Overview
- **Owner**: Deepak (Database Lead)
- **Role**: Relational data store for automotive inspection assets, manufacturing line telemetry, defect records, model predictions, alert triggers, and maintenance recommendations.
- **Current Status**: SKELETON / FOUNDATION PHASE — Provisional relational schema defined with core entity relationships. No dataset-specific fields hard-coded.

## 2. Directory Layout
- `schema/schema.sql`: Baseline DDL establishing foreign keys and indexing hooks.
- `migrations/`: Versioned migration scripts (Alembic or Sqitch).
- `seeds/`: Minimal mock records for integration and local testing.

## 3. Core Entities
1. **`components`**: Physical automotive parts inspected.
2. **`machines`**: Factory production machines / CNC stations.
3. **`batches`**: Manufacturing lot identifiers linking machines and parts.
4. **`defects`**: Visual inspection findings (bounding coordinates, severity).
5. **`process_data`**: Process telemetry (sensor streams, temperature, vibration).
6. **`predictions`**: AI model inference outputs (confidences, root-cause, risk).
7. **`recommendations`**: Corrective action procedures generated.
8. **`alerts`**: Urgent notification events and batch quarantine flags.

## 4. Decisions & Open Questions
- `TODO — DECISION REQUIRED`: Select database migration tool (Alembic vs Flyway vs raw SQL scripts).
- `TODO — DECISION REQUIRED`: Finalize sensor column schema vs JSONB semi-structured telemetry once process dataset is confirmed by Vijeath.
- `TODO — DECISION REQUIRED`: Establish partitioning strategy for high-frequency telemetry / inspection logs.
