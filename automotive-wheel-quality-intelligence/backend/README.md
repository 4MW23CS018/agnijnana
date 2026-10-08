# Backend Service — Aluminium Wheel Quality Intelligence (FastAPI)

## 1. Overview
- **Owner**: Ashish (Backend Lead)
- **Role**: Core application service exposing REST APIs for aluminium wheel defect inspection, orchestrating AI model inferences, managing database persistence, and securing endpoints with input validation and access controls.
- **Current Status**: SKELETON / FOUNDATION PHASE — Minimal runnable FastAPI application with health check, wheel inspection endpoint, and domain routers.

## 2. Structure
- `app/api/`: REST endpoint routers (`inspection.py`, `defects.py`, `root_cause.py`, `risk.py`, `recommendations.py`, `alerts.py`, `batches.py`, `machines.py`)
- `app/core/`: Configuration management (`config.py`) using `pydantic-settings`
- `app/schemas/`: Pydantic data schemas defining API contracts and the `WheelAIOutputContract`
- `app/models/`: SQLAlchemy database entity models
- `app/database/`: Database engine setup and session dependency injection
- `app/services/`: Business logic and AI model wrappers
- `app/auth/`: Access control and authentication middleware
- `tests/`: Automated unit and API integration tests

## 3. Running Locally
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```
Swagger API documentation: `http://localhost:8000/docs`
