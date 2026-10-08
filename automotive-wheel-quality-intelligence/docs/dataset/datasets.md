# Aluminium Wheel Dataset Research & Acquisition

**Owner**: Vijeath (Data + AI/ML Lead)

## 1. Scope
Acquire and validate datasets specifically containing **aluminium alloy automotive wheels and rims** with defect annotations, paired with or mappable to manufacturing process telemetry (casting, heat treatment, machining).

## 2. Candidate Datasets Under Investigation
- **GRIMA X-ray Database (GDXray) — Castings Series**:
  - Contains hundreds of real industrial radioscopic images of aluminium alloy automotive wheels and casting parts with annotated defects (shrinkage cavities, cracks, inclusions).
- **Public Casting Product Surface Inspection Datasets**:
  - High-resolution industrial optical inspection images of cast aluminium components and wheel rims.
- **Machining / Manufacturing Process Telemetry**:
  - Industrial casting/CNC telemetry datasets (e.g., die casting machine cycle logs, temperature/pressure profiles, or synthetic manufacturing logs paired with casting cycles).

## 3. Dataset Assessment Checklist (To Be Completed by Vijeath)
- [ ] Available physical fields identified from raw files.
- [ ] Ground-truth defect labels and bounding annotation formats verified.
- [ ] Casting batch, machine ID, and production cycle identifiers identified.
- [ ] Dataset licensing and distribution terms confirmed.
- [ ] Train / validation / test split strategy established.

## 4. Current Status
- SKELETON / FOUNDATION PHASE: Dataset candidate review in progress.
- `TODO — DECISION REQUIRED`: Finalize the specific aluminium wheel dataset selection and verify annotation formats before building data loaders.
