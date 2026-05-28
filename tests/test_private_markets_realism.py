import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.categorisation import categorize_company
from src.config import PROCESSED_DIR
from src.scoring import calculate_scores
from src.utils import read_csv_if_exists, upsert_skipped_tickers


def test_mega_cap_quality_comp_not_pe_platform():
    row = pd.Series(
        {
            "public_quality_score": 88,
            "investment_screening_score": 70,
            "platform_candidate_score": 82,
            "platform_size_feasibility_score": 5,
            "public_to_private_feasibility_score": 25,
            "value_creation_potential_score": 45,
            "credit_risk_score": 15,
            "red_flag_score": 25,
            "data_quality_score": 90,
            "market_cap_band": "Mega-cap benchmark / public quality comp",
        }
    )
    category, tags, reason = categorize_company(row)
    assert category in {"Public Market Quality Compounder", "Benchmark Quality Comp"}
    assert "PE Platform Candidate" not in reason


def test_data_quality_score_falls_when_fields_missing(sample_outputs):
    screening = sample_outputs["screening"].copy()
    benchmarks = sample_outputs["benchmarks"].copy()
    baseline = calculate_scores(screening, benchmarks)["data_quality_score"].mean()
    missing = screening.copy()
    missing.loc[missing.index[:5], ["operating_cash_flow", "capex", "total_debt", "beta"]] = np.nan
    missing["sec_field_completeness_ratio"] = missing["sec_field_completeness_ratio"] * 0.65
    lower = calculate_scores(missing, benchmarks)["data_quality_score"].mean()
    assert lower < baseline


def test_source_type_and_skipped_ticker_outputs_exist():
    chunks = read_csv_if_exists(Path("data/interim/filing_chunks.csv"))
    assert "source_type" in chunks.columns
    upsert_skipped_tickers([], PROCESSED_DIR / "skipped_tickers.csv")
    skipped = read_csv_if_exists(PROCESSED_DIR / "skipped_tickers.csv")
    assert {"ticker", "stage", "reason", "stage_failed", "reason_skipped"}.issubset(skipped.columns)


def test_memo_avoids_buy_sell_language():
    memo_path = Path("outputs/reports/investment_screening_memo.md")
    if not memo_path.exists():
        return
    text = memo_path.read_text(encoding="utf-8").lower()
    assert "buy recommendation" not in text
    assert "sell recommendation" not in text
    assert "target to acquire" not in text


def test_dashboard_module_compiles():
    path = Path("dashboard/app.py")
    spec = importlib.util.spec_from_file_location("dashboard_app_static", path)
    assert spec is not None
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
