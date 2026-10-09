"""Explainable future defect-risk rankings based only on prediction history.

The score is a prototype risk indicator, not a calibrated probability of a
future defect. Validate it against later ground-truth production outcomes
before using it for operational decisions.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RECENT_WINDOW = 20
PREVIOUS_WINDOW = 20
MIN_MACHINE_INSPECTIONS = 10
MIN_BATCH_INSPECTIONS = 5
TREND_SCALE = 0.20

RISK_WEIGHTS = {
    "recent": 0.50,
    "historical": 0.30,
    "positive_trend": 0.20,
}

REQUIRED_HISTORY_COLUMNS = {
    "inspection_id",
    "timestamp",
    "part_id",
    "batch_id",
    "machine_id",
    "predicted_defect",
    "model_confidence",
    "defect_present",
}
UPCOMING_COLUMNS = {"batch_id", "machine_id", "scheduled_time"}

MACHINE_RESULT_COLUMNS = [
    "rank", "machine_id", "inspections", "historical_defect_rate",
    "recent_defect_rate", "trend_change", "risk_percentage", "risk_level",
    "most_likely_defect", "defect_share",
]
BATCH_RESULT_COLUMNS = [
    "rank", "batch_id", "machine_id", "inspections", "historical_defect_rate",
    "recent_defect_rate", "trend_change", "risk_percentage", "risk_level",
    "most_likely_defect", "defect_share",
]
UPCOMING_RESULT_COLUMNS = [
    "rank", "batch_id", "machine_id", "scheduled_time", "machine_risk",
    "batch_historical_risk", "risk_percentage", "risk_level",
    "most_likely_defect", "defect_share", "historical_batch_available",
    "batch_inspections",
]

_NO_DEFECT_LABELS = {
    "", "unknown", "normal", "good", "ok", "pass", "no_defect",
    "no_defects", "defect_free", "background", "none", "null",
}


def _normalize_label(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def _as_binary_defect(value: Any) -> float:
    """Normalize common bool / integer representations to 0 or 1."""
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if value is None or pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "defect", "present"}:
        return 1.0
    if text in {"0", "false", "no", "n", "no_defect", "absent"}:
        return 0.0
    try:
        number = float(text)
    except (TypeError, ValueError):
        return np.nan
    return number if number in (0.0, 1.0) else np.nan


def load_prediction_history(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate a CSV prediction-history dataset.

    The optional `severity` column is preserved when supplied, but is not used
    by the current risk score. Invalid timestamps/IDs or non-binary
    `defect_present` values are rejected with a helpful error.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Prediction history CSV not found: {path}")
    try:
        history = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Prediction history CSV is empty: {path}") from exc
    return validate_prediction_history(history, source=str(path))


def validate_prediction_history(history: pd.DataFrame, *, source: str = "history") -> pd.DataFrame:
    if not isinstance(history, pd.DataFrame):
        raise TypeError("Prediction history must be a pandas DataFrame")
    missing = sorted(REQUIRED_HISTORY_COLUMNS - set(history.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")

    df = history.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if df["timestamp"].isna().any():
        bad_rows = (df.index[df["timestamp"].isna()] + 2).tolist()[:10]
        raise ValueError(
            f"{source} contains invalid or blank timestamps at CSV/data rows: {bad_rows}"
        )

    for column in ("inspection_id", "part_id", "batch_id", "machine_id"):
        df[column] = df[column].astype("string").str.strip()
        df[column] = df[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    df["predicted_defect"] = df["predicted_defect"].map(_normalize_label)
    df["model_confidence"] = pd.to_numeric(df["model_confidence"], errors="coerce")
    # Treat missing/invalid confidence as 0 in confidence-weighted class shares;
    # it does not affect the defect-rate components of the risk score.
    df["model_confidence"] = df["model_confidence"].fillna(0.0).clip(0.0, 1.0)
    df["defect_present"] = df["defect_present"].map(_as_binary_defect)
    if df["defect_present"].isna().any():
        bad_rows = (df.index[df["defect_present"].isna()] + 2).tolist()[:10]
        raise ValueError(
            f"{source} has defect_present values other than 0/1 at CSV/data rows: {bad_rows}"
        )

    if "severity" not in df.columns:
        df["severity"] = pd.NA
    df["severity"] = df["severity"].astype("string")

    # Keep the latest version of duplicate inspection IDs, if any.
    df = df.drop_duplicates(subset=["inspection_id"], keep="last")
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    return df


def _is_defect_label(label: Any) -> bool:
    value = _normalize_label(label)
    return value not in _NO_DEFECT_LABELS and not value.endswith("_unknown")


def _confidence_weighted_top_defect(recent: pd.DataFrame) -> tuple[str | None, float]:
    defective = recent.loc[
        (recent["defect_present"] == 1)
        & recent["predicted_defect"].map(_is_defect_label)
        ].copy()
    if defective.empty:
        return None, 0.0

    by_class = defective.groupby("predicted_defect", dropna=False)["model_confidence"].sum()
    total_weight = float(by_class.sum())
    if total_weight <= 0:
        # When no usable confidences exist, fall back to frequency share.
        by_class = defective["predicted_defect"].value_counts().astype(float)
        total_weight = float(by_class.sum())
    if total_weight <= 0 or by_class.empty:
        return None, 0.0

    top_class = str(by_class.idxmax())
    share = float(by_class.max() / total_weight)
    return top_class, float(np.clip(share, 0.0, 1.0))


def _risk_level(risk_percentage: float | None) -> str:
    if risk_percentage is None or not np.isfinite(risk_percentage):
        return "INSUFFICIENT_DATA"
    if risk_percentage < 20.0:
        return "LOW"
    if risk_percentage < 50.0:
        return "MODERATE"
    return "HIGH"


def _entity_metrics(group: pd.DataFrame, entity_id: str) -> dict[str, Any]:
    ordered = group.sort_values("timestamp", kind="stable")
    count = int(len(ordered))
    recent = ordered.tail(RECENT_WINDOW)
    previous_end = max(0, count - len(recent))
    previous_start = max(0, previous_end - PREVIOUS_WINDOW)
    previous = ordered.iloc[previous_start:previous_end]

    historical_rate = float(ordered["defect_present"].mean()) if count else 0.0
    recent_rate = float(recent["defect_present"].mean()) if len(recent) else 0.0
    # No prior window means no evidence of a trend; do not invent one.
    previous_rate = float(previous["defect_present"].mean()) if len(previous) else recent_rate
    trend_change = recent_rate - previous_rate
    positive_trend = float(np.clip(max(0.0, trend_change) / TREND_SCALE, 0.0, 1.0))

    risk_fraction = (
            RISK_WEIGHTS["recent"] * recent_rate
            + RISK_WEIGHTS["historical"] * historical_rate
            + RISK_WEIGHTS["positive_trend"] * positive_trend
    )
    risk_percentage = float(np.clip(risk_fraction, 0.0, 1.0) * 100.0)
    most_likely_defect, defect_share = _confidence_weighted_top_defect(recent)

    return {
        "entity_id": str(entity_id),
        "inspections": count,
        "historical_defect_rate": historical_rate,
        "recent_defect_rate": recent_rate,
        "trend_change": trend_change,
        "risk_percentage": risk_percentage,
        "risk_level": _risk_level(risk_percentage),
        "most_likely_defect": most_likely_defect,
        "defect_share": defect_share,
    }


def rank_machines(history: pd.DataFrame) -> pd.DataFrame:
    """Rank machines with at least MIN_MACHINE_INSPECTIONS history rows."""
    df = validate_prediction_history(history, source="prediction history")
    records: list[dict[str, Any]] = []
    eligible = df.dropna(subset=["machine_id"])
    for machine_id, group in eligible.groupby("machine_id", sort=False):
        if len(group) < MIN_MACHINE_INSPECTIONS:
            continue
        metrics = _entity_metrics(group, str(machine_id))
        metrics["machine_id"] = metrics.pop("entity_id")
        records.append(metrics)

    if not records:
        return pd.DataFrame(columns=MACHINE_RESULT_COLUMNS)
    result = pd.DataFrame(records).sort_values(
        ["risk_percentage", "inspections"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=int))
    return result[MACHINE_RESULT_COLUMNS]


def rank_batches(history: pd.DataFrame) -> pd.DataFrame:
    """Rank recurring batches with at least MIN_BATCH_INSPECTIONS history rows."""
    df = validate_prediction_history(history, source="prediction history")
    records: list[dict[str, Any]] = []
    eligible = df.dropna(subset=["batch_id"])
    for batch_id, group in eligible.groupby("batch_id", sort=False):
        if len(group) < MIN_BATCH_INSPECTIONS:
            continue
        metrics = _entity_metrics(group, str(batch_id))
        modes = group["machine_id"].dropna().mode()
        metrics["machine_id"] = str(modes.iloc[0]) if not modes.empty else None
        metrics["batch_id"] = metrics.pop("entity_id")
        records.append(metrics)

    if not records:
        return pd.DataFrame(columns=BATCH_RESULT_COLUMNS)
    result = pd.DataFrame(records).sort_values(
        ["risk_percentage", "inspections"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
    result.insert(0, "rank", np.arange(1, len(result) + 1, dtype=int))
    return result[BATCH_RESULT_COLUMNS]


def rank_upcoming_batches(
        history: pd.DataFrame,
        upcoming: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate upcoming-batch risk from machine and recurring-batch history.

    A batch with enough previous inspections combines machine score (60%) and
    batch-specific score (40%). A new/low-history batch uses eligible machine
    risk alone. If neither has enough evidence, risk is marked unavailable.
    """
    df = validate_prediction_history(history, source="prediction history")
    missing = sorted(UPCOMING_COLUMNS - set(upcoming.columns))
    if missing:
        raise ValueError(f"Upcoming-batch data is missing required columns: {', '.join(missing)}")
    scheduled = upcoming.copy()
    scheduled["batch_id"] = scheduled["batch_id"].astype("string").str.strip()
    scheduled["machine_id"] = scheduled["machine_id"].astype("string").str.strip()
    scheduled["scheduled_time"] = pd.to_datetime(
        scheduled["scheduled_time"], errors="coerce", utc=True
    )
    if scheduled["scheduled_time"].isna().any():
        raise ValueError("Upcoming-batch data contains invalid or blank scheduled_time values")

    machine_risk = rank_machines(df).set_index("machine_id") if not rank_machines(df).empty else pd.DataFrame()
    batch_risk = rank_batches(df).set_index("batch_id") if not rank_batches(df).empty else pd.DataFrame()

    records: list[dict[str, Any]] = []
    for _, item in scheduled.iterrows():
        batch_id = str(item["batch_id"])
        machine_id = str(item["machine_id"])
        machine_metric = machine_risk.loc[machine_id] if not machine_risk.empty and machine_id in machine_risk.index else None
        batch_metric = batch_risk.loc[batch_id] if not batch_risk.empty and batch_id in batch_risk.index else None

        machine_score = float(machine_metric["risk_percentage"]) if machine_metric is not None else None
        batch_score = float(batch_metric["risk_percentage"]) if batch_metric is not None else None

        if machine_score is not None and batch_score is not None:
            combined_score = 0.60 * machine_score + 0.40 * batch_score
        elif batch_score is not None:
            combined_score = batch_score
        elif machine_score is not None:
            combined_score = machine_score
        else:
            combined_score = None

        if batch_metric is not None and batch_metric.get("most_likely_defect"):
            likely_defect = batch_metric["most_likely_defect"]
            defect_share = float(batch_metric["defect_share"])
        elif machine_metric is not None:
            likely_defect = machine_metric.get("most_likely_defect")
            defect_share = float(machine_metric.get("defect_share", 0.0))
        else:
            likely_defect, defect_share = None, 0.0

        batch_group = df.loc[df["batch_id"] == batch_id]
        records.append({
            "batch_id": batch_id,
            "machine_id": machine_id,
            "scheduled_time": item["scheduled_time"],
            "machine_risk": machine_score,
            "batch_historical_risk": batch_score,
            "risk_percentage": combined_score,
            "risk_level": _risk_level(combined_score),
            "most_likely_defect": likely_defect,
            "defect_share": defect_share,
            "historical_batch_available": bool(batch_metric is not None),
            "batch_inspections": int(len(batch_group)),
        })

    result = pd.DataFrame(records, columns=[c for c in UPCOMING_RESULT_COLUMNS if c != "rank"])
    if result.empty:
        result.insert(0, "rank", pd.Series(dtype="Int64"))
        return result[UPCOMING_RESULT_COLUMNS]

    # Known-risk upcoming batches first; insufficient-history rows remain listed
    # but receive no fabricated percentage and no numeric rank.
    result["_has_risk"] = result["risk_percentage"].notna()
    result = result.sort_values(
        ["_has_risk", "risk_percentage", "scheduled_time"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)
    result.insert(0, "rank", pd.array(
        [i + 1 if has_risk else pd.NA for i, has_risk in enumerate(result["_has_risk"])],
        dtype="Int64",
    ))
    result = result.drop(columns=["_has_risk"])
    return result[UPCOMING_RESULT_COLUMNS]
