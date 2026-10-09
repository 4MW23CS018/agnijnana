# Future Defect Risk Integration

This add-on estimates which machines and upcoming batches have elevated defect risk using prediction history. It is an explainable prototype risk score, **not a calibrated probability** that a future unit will be defective. Validate against later ground-truth outcomes before production use.

## Files and destinations

Extract the ZIP into your repository root (`D:\Agnijnana\automotive-wheel-quality-intelligence`) so the files merge into these locations:

- `backend/app/services/future_risk_engine.py` — scoring functions
- `backend/app/api/future_risk.py` — FastAPI endpoints; reads historical rows from Supabase by default
- `backend/app/data/prediction_history.csv` — example CSV in the requested format
- `backend/app/data/prediction_history_sample.csv` — same small sample for explicit demo mode
- `backend/app/data/upcoming_batches.csv` — upcoming batch schedule
- `backend/tests/test_future_risk_engine.py` — built-in unit tests

## Install dependencies

Use your backend virtual environment:

```powershell
cd D:\Agnijnana\automotive-wheel-quality-intelligence\backend
.\venv\Scripts\python.exe -m pip install "numpy>=2.1.3" "pandas>=2.2.3"
```

Add these lines to `backend/requirements.txt` if they are not already present:

```text
numpy>=2.1.3
pandas>=2.2.3
```

The API also uses `httpx`, `fastapi`, and optionally `python-dotenv`, which are already used by the existing backend.

## Register the router

Open `backend/app/main.py`. Add this import alongside the existing API imports:

```python
from app.api import future_risk
```

After `app = FastAPI(...)` is created, register the router once:

```python
app.include_router(
    future_risk.router,
    prefix="/api/future-risk",
    tags=["Future Risk"],
)
```

Do not create a second FastAPI app. Do not add a second copy of this router if it is already registered.

## Backend environment

The default endpoints read actual saved predictions from Supabase. Keep a server-side key in your repository-root `.env` (never in the frontend):

```dotenv
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SECRET_KEY=YOUR_SERVER_SIDE_SECRET_KEY
SUPABASE_BUCKET=wheel-images
```

Legacy `SUPABASE_SERVICE_ROLE_KEY` is also supported. The backend uses the server-side key to read `predictions`, `machines`, `batches`, `components`, and `defects`; ensure the key has the required privileges. The code recognizes these prediction class columns: `prediction_type`, `defect_type`, `predicted_class`, `predicted_label`, `prediction_class`, or `defect_class`. If your renamed field uses a different name, add it to `_prediction_label()` in `future_risk.py`.

## Endpoints

After registering the router and restarting Uvicorn:

- `GET /api/future-risk/machines` — rank machines using real `public.predictions` rows
- `GET /api/future-risk/batches` — rank recurring batches with enough history
- `GET /api/future-risk/upcoming-batches` — score the entries in `upcoming_batches.csv` using machine and batch history
- `GET /api/future-risk/history.csv` — download the normalized history being analyzed

For local/demo CSV use, add `?source=csv` or `?source=sample`; the default `source=database` is the live Supabase history. The provided sample CSV has only four rows, so it correctly produces no machine ranking under the minimum of 10 inspections (and no batch ranking under the minimum of 5). It is an example format, not enough data for a meaningful ranking.

`upcoming_batches.csv` should use machine and batch *codes* matching `machines.machine_id` and `batches.batch_id`, rather than their UUID primary keys. Edit the schedule for your production scenario.

## Scoring logic

`risk_fraction = 0.50 * recent_defect_rate + 0.30 * historical_defect_rate + 0.20 * positive_trend_score`

- Recent window: latest 20 inspections per machine/batch
- Previous window: up to 20 inspections before the recent window
- Positive trend score: `min(max(recent_rate - previous_rate, 0) / 0.20, 1)`
- Minimum history: 10 inspections per machine; 5 per recurring batch
- Upcoming batch with its own eligible history: `60% machine risk + 40% batch risk`
- New/low-history batch: use eligible machine risk only
- If neither machine nor batch has enough evidence, the API returns `INSUFFICIENT_DATA`, not an invented percentage
- Most-likely defect is selected by confidence-weighted share among recent rows marked as defective

Risk levels are prototype bands: `LOW < 20%`, `MODERATE 20–<50%`, `HIGH >= 50%`.

## Important data limitation

The database ingestion endpoint derives `defect_present` from the saved predicted label (`no_defect`/normal-like labels are 0; other classes are 1). That means this is initially a *predicted-defect rate*, not an independently verified actual defect rate. For a scientifically stronger model, populate `defect_present` from inspected ground-truth/quality disposition when it becomes available.

## Test and start

```powershell
cd D:\Agnijnana_6\automotive-wheel-quality-intelligence\backend
.\venv\Scripts\python.exe -m py_compile app\api\future_risk.py app\services\future_risk_engine.py
$env:PYTHONPATH = (Get-Location).Path
.\venv\Scripts\python.exe -m unittest discover -s tests -v
.\venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` and look for the **Future Risk** endpoint group. These endpoints expose aggregated operational data; add your application's authentication/authorization before deploying them beyond a local/demo environment.
