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
