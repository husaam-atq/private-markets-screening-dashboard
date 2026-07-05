from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_divide(numerator: float | int | None, denominator: float | int | None) -> float:
    if numerator is None or denominator in (None, 0):
        return np.nan
    try:
        if pd.isna(numerator) or pd.isna(denominator) or float(denominator) == 0:
            return np.nan
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return np.nan


def clip_score(value: float | int | None) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(max(0, min(100, value)))


def percentile_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    ranks = numeric.rank(pct=True, method="average")
    if not higher_is_better:
        ranks = 1 - ranks
    return (ranks * 100).fillna(0).clip(0, 100)


def weighted_sum(frame: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    total = sum(abs(v) for v in weights.values()) or 1.0
    result = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        if column in frame.columns:
            result += pd.to_numeric(frame[column], errors="coerce").fillna(0) * weight
    return (result / total).clip(0, 100)


def clean_text(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or " ")).strip()
    return value


def normalise_accession(accession: str) -> str:
    return re.sub(r"[^0-9]", "", accession or "")


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except EmptyDataError:
            return pd.DataFrame()
    return pd.DataFrame()


def upsert_skipped_tickers(rows: list[dict], path: Path) -> pd.DataFrame:
    columns = [
        "ticker",
        "company_name",
        "stage",
        "stage_failed",
        "reason",
        "reason_skipped",
        "detail",
        "logged_at",
        "timestamp",
    ]
    existing = read_csv_if_exists(path)
    new_rows = pd.DataFrame(rows)
    if new_rows.empty:
        for column in columns:
            if column not in existing.columns:
                existing[column] = ""
        if not existing.empty:
            existing["stage_failed"] = existing["stage_failed"].where(existing["stage_failed"].astype(str).str.len() > 0, existing.get("stage", ""))
            existing["reason_skipped"] = existing["reason_skipped"].where(existing["reason_skipped"].astype(str).str.len() > 0, existing.get("reason", ""))
            existing["timestamp"] = existing["timestamp"].where(existing["timestamp"].astype(str).str.len() > 0, existing.get("logged_at", ""))
        if existing.empty:
            existing = pd.DataFrame(columns=columns)
        write_csv(existing[columns], path)
        return existing
    for column in columns:
        if column not in new_rows.columns:
            new_rows[column] = ""
    new_rows["stage_failed"] = new_rows["stage_failed"].where(new_rows["stage_failed"].astype(str).str.len() > 0, new_rows["stage"])
    new_rows["reason_skipped"] = new_rows["reason_skipped"].where(new_rows["reason_skipped"].astype(str).str.len() > 0, new_rows["reason"])
    new_rows["timestamp"] = new_rows["timestamp"].where(new_rows["timestamp"].astype(str).str.len() > 0, new_rows["logged_at"])
    combined = pd.concat([existing, new_rows[columns]], ignore_index=True) if not existing.empty else new_rows[columns]
    for column in columns:
        if column not in combined.columns:
            combined[column] = ""
    combined = combined.drop_duplicates(["ticker", "stage_failed", "reason_skipped"], keep="last").sort_values(["stage_failed", "ticker"])
    write_csv(combined, path)
    return combined


def skipped_ticker_row(
    ticker: str,
    company_name: str = "",
    stage: str = "",
    reason: str = "",
    detail: str = "",
) -> dict:
    timestamp = utc_timestamp()
    return {
        "ticker": ticker,
        "company_name": company_name,
        "stage": stage,
        "stage_failed": stage,
        "reason": reason,
        "reason_skipped": reason,
        "detail": detail,
        "logged_at": timestamp,
        "timestamp": timestamp,
    }


def metric_value(frame: pd.DataFrame, metric: str, default: float = 0.0) -> float:
    if frame.empty or "metric" not in frame.columns or "value" not in frame.columns:
        return default
    row = frame[frame["metric"] == metric]
    return float(row["value"].iloc[0]) if not row.empty else default


def reset_skipped_tickers(path: Path) -> pd.DataFrame:
    columns = [
        "ticker",
        "company_name",
        "stage",
        "stage_failed",
        "reason",
        "reason_skipped",
        "detail",
        "logged_at",
        "timestamp",
    ]
    frame = pd.DataFrame(columns=columns)
    write_csv(frame, path)
    return frame


def non_null_ratio(row: pd.Series, columns: Iterable[str]) -> float:
    cols = [col for col in columns if col in row.index]
    if not cols:
        return 0.0
    present = sum(not pd.isna(row[col]) for col in cols)
    return present / len(cols)


def format_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def format_multiple(value: float | int | None) -> str:
    if value is None or pd.isna(value) or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.1f}x"


def format_money(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}bn"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}m"
    return f"${value:,.0f}"
