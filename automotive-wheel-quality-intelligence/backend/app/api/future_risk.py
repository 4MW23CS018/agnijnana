"""API endpoints for future machine and upcoming-batch risk rankings.

Default source is the current public.predictions table in Supabase. The CSV
source is for offline/demo use only; the bundled CSV has only four sample rows,
which are below the minimum history thresholds and therefore won't produce
machine rankings yet.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import Response

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from app.services.future_risk_engine import (
    load_prediction_history,
    rank_batches,
    rank_machines,
    rank_upcoming_batches,
    validate_prediction_history,
)

logger = logging.getLogger(__name__)
router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if load_dotenv is not None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    load_dotenv(BACKEND_ROOT / ".env", override=False)

DATA_DIR = BACKEND_ROOT / "app" / "data"
HISTORY_CSV = Path(os.getenv("PREDICTION_HISTORY_CSV", str(DATA_DIR / "prediction_history.csv")))
SAMPLE_HISTORY_CSV = DATA_DIR / "prediction_history_sample.csv"
UPCOMING_CSV = Path(os.getenv("UPCOMING_BATCHES_CSV", str(DATA_DIR / "upcoming_batches.csv")))


def _supabase_credentials() -> tuple[str, str]:
    url = (os.getenv("SUPABASE_URL") or "").strip().rstrip("/")
    key = (
            os.getenv("SUPABASE_SECRET_KEY")
            or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_KEY")
            or ""
    ).strip()
    if not url or not key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Supabase is not configured. Set SUPABASE_URL and a server-side key in the repository-root .env file.",
        )
    if key.startswith("sb_publishable_"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Future-risk history requires backend-authorized read access. Configure a server-side Supabase secret/service-role key.",
        )
    return url, key


def _headers(key: str, *, range_header: str | None = None) -> dict[str, str]:
    result = {"apikey": key, "Accept": "application/json"}
    # Legacy anon/service-role keys are JWTs; modern sb_secret_ keys are not.
    if key.startswith("eyJ"):
        result["Authorization"] = f"Bearer {key}"
    if range_header:
        result["Range"] = range_header
    return result


async def _fetch_all(client: httpx.AsyncClient, url: str, key: str, table: str, select: str = "*") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    page_size = 1000
    while offset < 20000:
        response = await client.get(
            f"{url}/rest/v1/{table}",
            params={"select": select, "limit": page_size, "offset": offset, "order": "id.asc"},
            headers=_headers(key),
        )
        if response.is_error:
            logger.error("Future-risk read from public.%s failed: status=%s response=%s", table, response.status_code, response.text[:1000])
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not read public.{table} for future-risk analysis. Check the backend key, table privileges, RLS, and column names.",
            )
        try:
            page = response.json()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Supabase returned invalid JSON while reading public.{table}.") from exc
        if not isinstance(page, list):
            raise HTTPException(status_code=502, detail=f"Unexpected response while reading public.{table}.")
        records.extend(item for item in page if isinstance(item, dict))
        if len(page) < page_size:
            break
        offset += page_size
    return records


def _prediction_label(row: dict[str, Any]) -> str | None:
    for key in ("prediction_type", "defect_type", "predicted_class", "predicted_label", "prediction_class", "defect_class"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _is_predicted_defect(label: str | None) -> int:
    normalized = (label or "").strip().lower().replace("-", "_").replace(" ", "_")
    no_defect = {"", "unknown", "normal", "good", "ok", "pass", "no_defect", "no_defects", "defect_free", "background", "none", "null"}
    return 0 if normalized in no_defect or normalized.endswith("_unknown") else 1


def _parse_explanation(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, TypeError):
            return {}
    return {}


async def load_history_from_database() -> pd.DataFrame:
    """Convert actual saved predictions and their FK UUIDs into history rows."""
    url, key = _supabase_credentials()
    async with httpx.AsyncClient(timeout=httpx.Timeout(45.0, connect=10.0)) as client:
        predictions = await _fetch_all(client, url, key, "predictions")
        if not predictions:
            return pd.DataFrame(columns=[
                "inspection_id", "timestamp", "part_id", "batch_id", "machine_id",
                "predicted_defect", "model_confidence", "defect_present", "severity",
            ])
        machines = await _fetch_all(client, url, key, "machines", "id,machine_id")
        batches = await _fetch_all(client, url, key, "batches", "id,batch_id,machine_id")
        components = await _fetch_all(client, url, key, "components", "id,component_id")
        defects = await _fetch_all(client, url, key, "defects", "component_id,defect_type,severity,confidence")

    machine_codes = {str(item.get("id")): item.get("machine_id") for item in machines if item.get("id")}
    batch_codes = {str(item.get("id")): item.get("batch_id") for item in batches if item.get("id")}
    batch_machine_uuids = {str(item.get("id")): item.get("machine_id") for item in batches if item.get("id")}
    component_codes = {str(item.get("id")): item.get("component_id") for item in components if item.get("id")}

    defects_by_component: dict[str, list[dict[str, Any]]] = {}
    for item in defects:
        component_uuid = item.get("component_id")
        if component_uuid:
            defects_by_component.setdefault(str(component_uuid), []).append(item)

    history_rows: list[dict[str, Any]] = []
    for row in predictions:
        label = _prediction_label(row)
        if not label:
            continue
        timestamp = row.get("predicted_at") or row.get("created_at")
        if not timestamp:
            continue
        machine_uuid = str(row.get("machine_id")) if row.get("machine_id") else None
        batch_uuid = str(row.get("batch_id")) if row.get("batch_id") else None
        if not machine_uuid and batch_uuid:
            machine_uuid = str(batch_machine_uuids.get(batch_uuid)) if batch_machine_uuids.get(batch_uuid) else None
        component_uuid = str(row.get("component_id")) if row.get("component_id") else None
        explanation = _parse_explanation(row.get("explanation"))
        part_id = (
                explanation.get("wheel_id")
                or component_codes.get(component_uuid or "")
                or component_uuid
                or row.get("id")
        )
        mapped_defects = defects_by_component.get(component_uuid or "", [])
        selected_severity = None
        if mapped_defects:
            matching = [d for d in mapped_defects if str(d.get("defect_type", "")).strip().lower() == label.lower()]
            severity_source = matching or mapped_defects
            severity_source = sorted(severity_source, key=lambda d: float(d.get("confidence") or 0.0), reverse=True)
            selected_severity = severity_source[0].get("severity")

        history_rows.append({
            "inspection_id": row.get("id") or f"inspection-{len(history_rows) + 1}",
            "timestamp": timestamp,
            "part_id": part_id,
            "batch_id": batch_codes.get(batch_uuid or "") or batch_uuid or "",
            "machine_id": machine_codes.get(machine_uuid or "") or machine_uuid or "",
            "predicted_defect": label,
            "model_confidence": row.get("probability", 0.0),
            "defect_present": _is_predicted_defect(label),
            "severity": selected_severity,
        })

    history_df = pd.DataFrame(history_rows, columns=[
        "inspection_id", "timestamp", "part_id", "batch_id", "machine_id",
        "predicted_defect", "model_confidence", "defect_present", "severity",
    ])
    if history_df.empty:
        return history_df
    return validate_prediction_history(history_df, source="public.predictions")


def _get_csv_history(source: str) -> pd.DataFrame:
    path = SAMPLE_HISTORY_CSV if source == "sample" else HISTORY_CSV
    try:
        return load_prediction_history(path)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _get_history(source: str) -> tuple[pd.DataFrame, str]:
    if source == "database":
        try:
            return await load_history_from_database(), "public.predictions"
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Could not build future-risk history from Supabase")
            raise HTTPException(status_code=502, detail="Could not load prediction history from Supabase. See backend logs.") from exc
    if source in {"csv", "sample"}:
        return _get_csv_history(source), ("prediction_history_sample.csv" if source == "sample" else str(HISTORY_CSV))
    raise HTTPException(status_code=400, detail="source must be 'database', 'csv', or 'sample'.")


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    cleaned = df.astype(object).where(pd.notna(df), None)
    records = cleaned.to_dict(orient="records")
    # Timestamps / numpy scalar values are normalized for JSON serialization.
    for record in records:
        for key, value in list(record.items()):
            if hasattr(value, "isoformat"):
                record[key] = value.isoformat()
            elif hasattr(value, "item"):
                try:
                    record[key] = value.item()
                except Exception:
                    pass
    return records


@router.get("/machines", summary="Rank machines by historical future-defect risk")
async def get_machine_future_risk(source: str = Query(default="database")):
    history, source_name = await _get_history(source)
    ranked = rank_machines(history) if not history.empty else pd.DataFrame()
    return {
        "source": source_name,
        "history_rows": int(len(history)),
        "minimum_inspections": 10,
        "score_type": "prototype risk score; not a calibrated probability",
        "message": None if not ranked.empty else "No machine meets the minimum history threshold of 10 inspections yet.",
        "machines": _records(ranked),
    }


@router.get("/batches", summary="Rank batches by historical future-defect risk")
async def get_batch_historical_risk(source: str = Query(default="database")):
    history, source_name = await _get_history(source)
    ranked = rank_batches(history) if not history.empty else pd.DataFrame()
    return {
        "source": source_name,
        "history_rows": int(len(history)),
        "minimum_inspections": 5,
        "score_type": "prototype risk score; not a calibrated probability",
        "message": None if not ranked.empty else "No batch meets the minimum history threshold of 5 inspections yet.",
        "batches": _records(ranked),
    }


@router.get("/upcoming-batches", summary="Estimate risk for upcoming production batches")
async def get_upcoming_batch_risk(source: str = Query(default="database")):
    history, source_name = await _get_history(source)
    try:
        upcoming = pd.read_csv(UPCOMING_CSV)
    except (FileNotFoundError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=400, detail=f"Upcoming-batch CSV could not be loaded: {UPCOMING_CSV}") from exc
    try:
        ranked = rank_upcoming_batches(history, upcoming)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "history_source": source_name,
        "history_rows": int(len(history)),
        "upcoming_csv": str(UPCOMING_CSV),
        "score_type": "prototype risk score; not a calibrated probability",
        "message": None if any(pd.notna(risk) for risk in ranked.get("risk_percentage", [])) else "No upcoming batch has enough machine or batch history to estimate risk yet.",
        "upcoming_batches": _records(ranked),
    }


@router.get("/history.csv", summary="Download prediction history normalized from Supabase or CSV")
async def download_prediction_history(source: str = Query(default="database")):
    history, _ = await _get_history(source)
    return Response(
        content=history.to_csv(index=False),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="prediction_history.csv"'},
    )
