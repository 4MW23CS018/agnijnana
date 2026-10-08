-- =============================================================================
-- Aluminium Wheel Quality Intelligence — Relational Schema (PostgreSQL 15+)
-- Owner: Deepak (Database Lead)
-- Status: SKELETON / FOUNDATION PHASE — Domain-adapted for Aluminium Alloy Wheels
--
-- Notice: Dataset-specific fields are provisional.
-- Column types will be refined after physical casting dataset is selected by Vijeath.
-- =============================================================================

-- 1. Machines (Die-casting machines, low-pressure casting units, CNC lathes)
CREATE TABLE IF NOT EXISTS machines (
    id SERIAL PRIMARY KEY,
    machine_code VARCHAR(64) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    line_identifier VARCHAR(64),
    status VARCHAR(32) DEFAULT 'OPERATIONAL', -- 'OPERATIONAL', 'MAINTENANCE', 'DEGRADED'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- TODO — DECISION REQUIRED: Add die temperature calibration, injection pressure limits, firmware version
);

-- 2. Batches (Aluminium melt / casting lots)
CREATE TABLE IF NOT EXISTS batches (
    id SERIAL PRIMARY KEY,
    batch_number VARCHAR(128) UNIQUE NOT NULL,
    machine_id INTEGER REFERENCES machines(id) ON DELETE SET NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(32) DEFAULT 'IN_PROGRESS', -- 'IN_PROGRESS', 'QUARANTINED', 'RELEASED'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- TODO — DECISION REQUIRED: Add melt heat number, alloy composition (e.g. A356), operator_id
);

-- 3. Components (Inspected aluminium wheels / rims)
CREATE TABLE IF NOT EXISTS components (
    id SERIAL PRIMARY KEY,
    component_uuid VARCHAR(128) UNIQUE NOT NULL,
    batch_id INTEGER REFERENCES batches(id) ON DELETE CASCADE,
    machine_id INTEGER REFERENCES machines(id) ON DELETE SET NULL,
    part_type VARCHAR(128) NOT NULL, -- e.g. 'alloy_wheel_18in', 'rim_forged'
    inspection_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    image_uri TEXT, -- Path or object store URI
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- TODO — DECISION REQUIRED: Add wheel dimensions, rim diameter, spoke count, surface finish grade
);

-- 4. Defects (Localized visual wheel defects)
CREATE TABLE IF NOT EXISTS defects (
    id SERIAL PRIMARY KEY,
    component_id INTEGER REFERENCES components(id) ON DELETE CASCADE,
    defect_type VARCHAR(128) NOT NULL, -- e.g. 'porosity', 'shrinkage_cavity', 'rim_crack', 'flash', 'dross'
    severity VARCHAR(32) NOT NULL,     -- 'Low' or 'Critical'
    location JSONB,                    -- [x_min, y_min, width, height] or segmentation polygon
    confidence DOUBLE PRECISION,       -- Detection confidence in [0.0, 1.0]
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- TODO — DECISION REQUIRED: Add surface_area_mm2, depth_estimate, wheel_zone (hub, spoke, rim)
);

-- 5. Process Data (Sensor & operational telemetry)
CREATE TABLE IF NOT EXISTS process_data (
    id SERIAL PRIMARY KEY,
    batch_id INTEGER REFERENCES batches(id) ON DELETE CASCADE,
    machine_id INTEGER REFERENCES machines(id) ON DELETE CASCADE,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    telemetry JSONB NOT NULL           -- Semi-structured readings: temp, vibration, speed, pressure
    -- TODO — DECISION REQUIRED: Normalize key metrics (spindle_rpm, feed_rate, vibration_rms) into dedicated columns post dataset selection
);

-- 6. Predictions (Model inference outputs: RCA & Future Risk)
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    defect_id INTEGER REFERENCES defects(id) ON DELETE CASCADE,
    component_id INTEGER REFERENCES components(id) ON DELETE CASCADE,
    batch_id INTEGER REFERENCES batches(id) ON DELETE CASCADE,
    root_cause TEXT,
    root_cause_confidence DOUBLE PRECISION,
    future_risk DOUBLE PRECISION,      -- Forecasted failure risk in [0.0, 1.0]
    model_version VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- TODO — DECISION REQUIRED: Store SHAP feature attribution vectors in JSONB
);

-- 7. Recommendations (Prescribed corrective maintenance actions)
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES predictions(id) ON DELETE CASCADE,
    recommended_action TEXT NOT NULL,
    priority VARCHAR(32) DEFAULT 'MONITOR', -- 'IMMEDIATE', 'SCHEDULED_SHIFT_END', 'MONITOR'
    sop_code VARCHAR(64),                   -- Standard Operating Procedure reference
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
    -- TODO — DECISION REQUIRED: Add assigned_technician_id, action_status (PENDING, RESOLVED)
);

-- 8. Alerts (Real-time notifications and batch quarantine warnings)
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(64) NOT NULL,        -- 'CRITICAL_DEFECT', 'BATCH_QUARANTINE', 'MACHINE_DRIFT'
    severity VARCHAR(32) NOT NULL,          -- 'WARNING', 'CRITICAL'
    message TEXT NOT NULL,
    affected_batch_id INTEGER REFERENCES batches(id) ON DELETE SET NULL,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_by VARCHAR(128),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_defects_component_id ON defects(component_id);
CREATE INDEX IF NOT EXISTS idx_defects_severity ON defects(severity);
CREATE INDEX IF NOT EXISTS idx_components_batch_id ON components(batch_id);
CREATE INDEX IF NOT EXISTS idx_process_data_batch_time ON process_data(batch_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_batch_id ON predictions(batch_id);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged ON alerts(acknowledged);
