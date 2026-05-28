from __future__ import annotations

import argparse

import pandas as pd

from src.categorisation import run_categorisation
from src.config import PROCESSED_DIR, ensure_project_dirs
from src.feature_engineering import run_feature_engineering
from src.peer_benchmarking import run_peer_benchmarking
from src.scoring import run_scoring
from src.sec_client import load_sec_fundamentals
from src.universe import write_universe_outputs
from src.utils import read_csv_if_exists, write_csv
from src.yfinance_client import load_market_data


def build_target_deep_dive(scores: pd.DataFrame, categories: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    top = scores.sort_values("investment_screening_score", ascending=False).head(1)
    if top.empty:
        return pd.DataFrame()
    ticker = top.iloc[0]["ticker"]
    category = categories[categories["ticker"] == ticker]
    peer = benchmarks[benchmarks["ticker"] == ticker]
    merged = top.merge(category[["ticker", "primary_category", "category_tags", "why_this_category"]], on="ticker", how="left")
    merged = merged.merge(peer, on=["ticker", "company_name", "sector_theme"], how="left", suffixes=("", "_peer"))
    write_csv(merged, PROCESSED_DIR / "target_deep_dive.csv")
    return merged


def run_pipeline(mode: str = "sample") -> dict[str, pd.DataFrame]:
    ensure_project_dirs()
    write_universe_outputs()
    market = load_market_data(mode=mode)
    fundamentals = load_sec_fundamentals(mode=mode)
    screening = run_feature_engineering(mode=mode)
    benchmarks = run_peer_benchmarking()
    scores = run_scoring()
    categories = run_categorisation()
    target = build_target_deep_dive(scores, categories, benchmarks)
    return {
        "market": market,
        "fundamentals": fundamentals,
        "screening": screening,
        "benchmarks": benchmarks,
        "scores": scores,
        "categories": categories,
        "target": target,
    }


def load_pipeline_outputs() -> dict[str, pd.DataFrame]:
    return {
        "screening": read_csv_if_exists(PROCESSED_DIR / "screening_universe.csv"),
        "benchmarks": read_csv_if_exists(PROCESSED_DIR / "peer_benchmarks.csv"),
        "scores": read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv"),
        "categories": read_csv_if_exists(PROCESSED_DIR / "company_categories.csv"),
        "target": read_csv_if_exists(PROCESSED_DIR / "target_deep_dive.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the private markets screening pipeline.")
    parser.add_argument("--mode", choices=["sample", "online"], default="sample")
    args = parser.parse_args()
    outputs = run_pipeline(mode=args.mode)
    print(f"companies_screened={len(outputs['screening'])}")
    print(f"sectors_covered={outputs['screening']['sector_theme'].nunique()}")
    print(f"top_target={outputs['target']['ticker'].iloc[0] if not outputs['target'].empty else 'none'}")


if __name__ == "__main__":
    main()
