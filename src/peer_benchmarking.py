from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ensure_project_dirs
from src.feature_engineering import run_feature_engineering
from src.utils import percentile_score, read_csv_if_exists, safe_divide, write_csv


BENCHMARK_METRICS = [
    "revenue_growth_yoy",
    "revenue_cagr_3y",
    "gross_margin",
    "operating_margin",
    "fcf_conversion",
    "debt_to_ebit_proxy",
    "net_debt_to_ebit_proxy",
    "ev_to_sales",
    "ev_to_ebit_proxy",
    "drawdown_from_52w_high",
    "realised_volatility",
]


def build_peer_benchmarks(screening: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for sector, group in screening.groupby("sector_theme", dropna=False):
        medians = group[BENCHMARK_METRICS].median(numeric_only=True)
        q1 = group[BENCHMARK_METRICS].quantile(0.25, numeric_only=True)
        q3 = group[BENCHMARK_METRICS].quantile(0.75, numeric_only=True)
        percentiles = pd.DataFrame(index=group.index)
        for metric in BENCHMARK_METRICS:
            higher = metric not in {
                "debt_to_ebit_proxy",
                "net_debt_to_ebit_proxy",
                "ev_to_sales",
                "ev_to_ebit_proxy",
                "drawdown_from_52w_high",
                "realised_volatility",
            }
            percentiles[f"{metric}_peer_percentile"] = percentile_score(group[metric], higher_is_better=higher)
        for idx, company in group.iterrows():
            row = {
                "ticker": company["ticker"],
                "company_name": company["company_name"],
                "sector_theme": sector,
                "peer_count": len(group),
            }
            for metric in BENCHMARK_METRICS:
                value = company.get(metric, np.nan)
                median = medians.get(metric, np.nan)
                row[f"{metric}_peer_median"] = median
                row[f"{metric}_peer_q1"] = q1.get(metric, np.nan)
                row[f"{metric}_peer_q3"] = q3.get(metric, np.nan)
                row[f"{metric}_gap_to_median"] = value - median if not pd.isna(value) and not pd.isna(median) else np.nan
                row[f"{metric}_premium_discount_to_median"] = safe_divide(value - median, abs(median))
                row[f"{metric}_peer_percentile"] = percentiles.loc[idx, f"{metric}_peer_percentile"]
            rows.append(row)
    result = pd.DataFrame(rows)
    result["valuation_premium_discount_to_median"] = result["ev_to_sales_premium_discount_to_median"]
    result["margin_gap_vs_peers"] = result["operating_margin_gap_to_median"]
    result["growth_gap_vs_peers"] = result["revenue_growth_yoy_gap_to_median"]
    result["leverage_gap_vs_peers"] = result["debt_to_ebit_proxy_gap_to_median"]
    result["cash_conversion_gap_vs_peers"] = result["fcf_conversion_gap_to_median"]
    result["red_flag_comparison_vs_peers"] = np.nan
    return result


def run_peer_benchmarking() -> pd.DataFrame:
    ensure_project_dirs()
    screening = read_csv_if_exists(PROCESSED_DIR / "screening_universe.csv")
    if screening.empty:
        screening = run_feature_engineering()
    benchmarks = build_peer_benchmarks(screening)
    write_csv(benchmarks, PROCESSED_DIR / "peer_benchmarks.csv")
    return benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sector peer benchmarks.")
    parser.parse_args()
    frame = run_peer_benchmarking()
    print(f"peer_benchmark_rows={len(frame)}")


if __name__ == "__main__":
    main()
