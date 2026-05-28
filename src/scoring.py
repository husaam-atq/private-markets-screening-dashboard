from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR, ensure_project_dirs, load_config
from src.feature_engineering import run_feature_engineering
from src.peer_benchmarking import build_peer_benchmarks, run_peer_benchmarking
from src.utils import clip_score, percentile_score, read_csv_if_exists, safe_divide, weighted_sum, write_csv


SCORE_COLUMNS = [
    "public_quality_score",
    "investment_screening_score",
    "diligence_priority_score",
    "platform_candidate_score",
    "public_to_private_feasibility_score",
    "value_creation_potential_score",
    "credit_risk_score",
    "red_flag_score",
    "data_quality_score",
]


def _sweet_spot_score(value: float | int | None, bands: list[tuple[float, float, float]]) -> float:
    if value is None or pd.isna(value):
        return 0.0
    value = float(value)
    for lower, upper, score in bands:
        if lower <= value < upper:
            return score
    return 0.0


def _base_metric_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    directions = load_config("scoring_config.yaml")["metrics"]
    for metric in directions["higher_is_better"]:
        if metric in scored:
            scored[f"{metric}_score"] = percentile_score(scored[metric], True)
    for metric in directions["lower_is_better"]:
        if metric in scored:
            scored[f"{metric}_score"] = percentile_score(scored[metric], False)
    scored["market_cap_scale_score"] = percentile_score(np.log1p(pd.to_numeric(scored["market_cap"], errors="coerce")), True)
    ev = pd.to_numeric(scored["enterprise_value"], errors="coerce")
    scored["platform_size_feasibility_score"] = ev.apply(
        lambda value: _sweet_spot_score(
            value,
            [
                (0, 1_000_000_000, 35),
                (1_000_000_000, 5_000_000_000, 85),
                (5_000_000_000, 20_000_000_000, 100),
                (20_000_000_000, 50_000_000_000, 65),
                (50_000_000_000, 150_000_000_000, 20),
                (150_000_000_000, float("inf"), 5),
            ],
        )
    )
    scored["public_to_private_size_score"] = ev.apply(
        lambda value: _sweet_spot_score(
            value,
            [
                (0, 1_000_000_000, 20),
                (1_000_000_000, 5_000_000_000, 60),
                (5_000_000_000, 25_000_000_000, 100),
                (25_000_000_000, 50_000_000_000, 75),
                (50_000_000_000, 100_000_000_000, 35),
                (100_000_000_000, 150_000_000_000, 15),
                (150_000_000_000, float("inf"), 3),
            ],
        )
    )
    scored["data_quality_score"] = 0.0
    return scored


def _risk_scores(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    scored["debt_to_ebit_risk_score"] = percentile_score(scored["debt_to_ebit_proxy"], True)
    scored["net_debt_to_ebit_risk_score"] = percentile_score(scored["net_debt_to_ebit_proxy"], True)
    scored["fcf_conversion_risk_score"] = 100 - scored.get("fcf_conversion_score", 0)
    scored["cash_to_revenue_risk_score"] = 100 - scored.get("cash_to_revenue_score", 0)
    scored["margin_trend_risk_score"] = 100 - scored.get("margin_trend_score", 0)
    scored["drawdown_risk_score"] = percentile_score(scored["drawdown_from_52w_high"], True)
    scored["volatility_risk_score"] = percentile_score(scored["realised_volatility"], True)
    scored["negative_growth_risk_score"] = np.where(scored["revenue_growth_yoy"] < 0, 100, 100 - scored.get("revenue_growth_yoy_score", 0))
    scored["margin_deterioration_risk_score"] = np.where(scored["margin_trend"] < 0, 100, 100 - scored.get("margin_trend_score", 0))
    scored["weak_fcf_risk_score"] = np.where(scored["fcf_conversion"] < 0, 100, 100 - scored.get("fcf_conversion_score", 0))
    scored["leverage_risk_score"] = scored["debt_to_ebit_risk_score"]
    scored["liquidity_risk_score"] = 100 - scored.get("cash_to_revenue_score", 0)
    growth_support = pd.to_numeric(scored["revenue_growth_yoy"], errors="coerce").fillna(0)
    valuation = pd.to_numeric(scored["ev_to_sales"], errors="coerce")
    scored["valuation_without_growth_risk_score"] = np.where((valuation > valuation.quantile(0.70)) & (growth_support < growth_support.median()), 100, 100 - scored.get("ev_to_sales_score", 0))
    scored["data_gap_risk_score"] = 100 - scored["data_quality_score"]
    scored["red_flag_inverse_score"] = 100
    return scored


def _peer_adjustment_scores(scored: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    merged = scored.merge(
        benchmarks[
            [
                "ticker",
                "peer_count",
                "revenue_growth_yoy_peer_median",
                "operating_margin_peer_median",
                "fcf_conversion_peer_median",
                "debt_to_ebit_proxy_peer_median",
                "ev_to_sales_peer_median",
                "operating_margin_gap_to_median",
                "fcf_conversion_gap_to_median",
                "ev_to_sales_premium_discount_to_median",
            ]
        ],
        on="ticker",
        how="left",
    )
    margin_gap = -pd.to_numeric(merged["operating_margin_gap_to_median"], errors="coerce")
    cash_gap = -pd.to_numeric(merged["fcf_conversion_gap_to_median"], errors="coerce")
    valuation_discount = -pd.to_numeric(merged["ev_to_sales_premium_discount_to_median"], errors="coerce")
    merged["margin_gap_to_peer_score"] = percentile_score(margin_gap, True)
    merged["cash_conversion_gap_to_peer_score"] = percentile_score(cash_gap, True)
    merged["valuation_discount_to_peer_score"] = percentile_score(valuation_discount, True)
    return merged


def _filing_source_scores(scored: pd.DataFrame) -> pd.DataFrame:
    chunks = read_csv_if_exists(INTERIM_DIR / "filing_chunks.csv")
    scored = scored.copy()
    scored["filing_chunk_count"] = 0
    scored["real_filing_chunk_count"] = 0
    scored["sample_fallback_chunk_count"] = 0
    scored["filing_source_score"] = 35.0
    if chunks.empty or "ticker" not in chunks.columns:
        return scored
    source_counts = chunks.pivot_table(
        index="ticker",
        columns="source_type",
        values="chunk_id",
        aggfunc="count",
        fill_value=0,
    ).reset_index()
    for column in ["real_sec_filing", "sample_fallback", "cached_sec_snapshot"]:
        if column not in source_counts.columns:
            source_counts[column] = 0
    source_counts["filing_chunk_count"] = source_counts[["real_sec_filing", "sample_fallback", "cached_sec_snapshot"]].sum(axis=1)
    source_counts["real_filing_chunk_count"] = source_counts["real_sec_filing"]
    source_counts["sample_fallback_chunk_count"] = source_counts["sample_fallback"]
    source_counts["filing_source_score"] = np.select(
        [
            source_counts["real_sec_filing"] >= 25,
            source_counts["cached_sec_snapshot"] >= 10,
            source_counts["sample_fallback"] >= 5,
        ],
        [100, 82, 64],
        default=35,
    )
    keep = [
        "ticker",
        "filing_chunk_count",
        "real_filing_chunk_count",
        "sample_fallback_chunk_count",
        "filing_source_score",
    ]
    scored = scored.merge(source_counts[keep], on="ticker", how="left", suffixes=("", "_filing"))
    for column in keep[1:]:
        duplicate = f"{column}_filing"
        if duplicate in scored.columns:
            scored[column] = scored[duplicate].combine_first(scored[column])
            scored = scored.drop(columns=[duplicate])
        scored[column] = pd.to_numeric(scored[column], errors="coerce").fillna(0)
    return scored


def _data_quality_scores(scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored.copy()
    yfin = pd.to_numeric(scored.get("yfinance_field_completeness_ratio", 0), errors="coerce").fillna(0)
    sec = pd.to_numeric(scored.get("sec_field_completeness_ratio", 0), errors="coerce").fillna(0)
    days = pd.to_numeric(scored.get("filing_staleness_days", np.nan), errors="coerce")
    staleness = np.where(days.isna(), 45, np.where(days <= 540, 100, np.where(days <= 900, 75, 45)))
    peer = pd.to_numeric(scored.get("peer_count", 0), errors="coerce").fillna(0)
    peer_score = np.where(peer >= 5, 100, np.where(peer >= 3, 75, 45))
    source = scored.get("source_type", pd.Series("missing", index=scored.index)).fillna("missing")
    source_score = np.select(
        [
            source.eq("live_public_data"),
            source.eq("cached_sec_snapshot"),
            source.eq("sample_fallback"),
        ],
        [100, 88, 78],
        default=55,
    )
    proxy_penalty = pd.to_numeric(scored.get("proxy_usage_penalty", 0), errors="coerce").fillna(0)
    missing_cik_penalty = np.where(pd.to_numeric(scored.get("cik", np.nan), errors="coerce").isna(), 12, 0)
    filing_source = pd.to_numeric(scored.get("filing_source_score", 35), errors="coerce").fillna(35)
    fallback_penalty = np.where(pd.to_numeric(scored.get("sample_fallback_chunk_count", 0), errors="coerce").fillna(0) > 0, 5, 0)
    no_real_filing_penalty = np.where(pd.to_numeric(scored.get("real_filing_chunk_count", 0), errors="coerce").fillna(0) == 0, 6, 0)
    quality = (
        yfin * 100 * 0.24
        + sec * 100 * 0.30
        + staleness * 0.12
        + peer_score * 0.10
        + source_score * 0.08
        + filing_source * 0.10
        + 100 * 0.06
        - proxy_penalty
        - missing_cik_penalty
        - fallback_penalty
        - no_real_filing_penalty
    )
    scored["data_quality_score"] = pd.Series(quality, index=scored.index).clip(0, 100)
    scored["data_gap_risk_score"] = 100 - scored["data_quality_score"]
    scored["data_quality_explanation"] = scored.apply(
        lambda row: (
            f"Market fields {row.get('yfinance_field_completeness_ratio', 0):.0%}; "
            f"SEC fields {row.get('sec_field_completeness_ratio', 0):.0%}; "
            f"filing age {row.get('filing_staleness_days', np.nan):.0f} days; "
            f"peer count {row.get('peer_count', 0)}; source {row.get('source_type', 'unknown')}; "
            f"real filing chunks {row.get('real_filing_chunk_count', 0):.0f}; "
            "EBIT proxy used instead of EBITDA."
        ),
        axis=1,
    )
    return scored


def _score_explanation(row: pd.Series) -> str:
    strengths = []
    risks = []
    if row.get("revenue_growth_yoy", 0) > row.get("revenue_growth_yoy_peer_median", -999):
        strengths.append("above-peer latest revenue growth")
    if row.get("operating_margin", 0) > row.get("operating_margin_peer_median", -999):
        strengths.append("above-peer EBIT margin proxy")
    if row.get("fcf_conversion", 0) > 0.8:
        strengths.append("strong free cash flow conversion proxy")
    if row.get("debt_to_ebit_proxy", 999) > 3.5:
        risks.append("elevated debt to EBIT proxy")
    if row.get("drawdown_from_52w_high", 0) > 0.35:
        risks.append("material share price drawdown")
    if row.get("data_quality_score", 100) < 70:
        risks.append("incomplete public data coverage")
    strength_text = ", ".join(strengths[:3]) if strengths else "balanced but not standout structured metrics"
    risk_text = ", ".join(risks[:3]) if risks else "no major structured red flag from the cached screen"
    return f"Screen is supported by {strength_text}; key caution is {risk_text}."


def calculate_scores(screening: pd.DataFrame, benchmarks: pd.DataFrame | None = None) -> pd.DataFrame:
    cfg = load_config("scoring_config.yaml")["score_weights"]
    if benchmarks is None:
        benchmarks = build_peer_benchmarks(screening)
    scored = _base_metric_scores(screening)
    scored = _peer_adjustment_scores(scored, benchmarks)
    scored = _filing_source_scores(scored)
    scored = _data_quality_scores(scored)
    scored = _risk_scores(scored)
    scored["credit_risk_score"] = weighted_sum(scored, cfg["credit_risk_score"])
    scored["credit_risk_inverse_score"] = 100 - scored["credit_risk_score"]
    scored["red_flag_score"] = weighted_sum(scored, cfg["red_flag_score"])
    scored["red_flag_inverse_score"] = 100 - scored["red_flag_score"]
    scored["public_quality_score"] = weighted_sum(scored, cfg["public_quality_score"])
    scored["platform_candidate_score"] = weighted_sum(scored, cfg["platform_candidate_score"])
    scored["public_to_private_feasibility_score"] = weighted_sum(scored, cfg["public_to_private_feasibility_score"])
    scored["value_creation_potential_score"] = weighted_sum(scored, cfg["value_creation_potential_score"])
    scored["investment_screening_score"] = weighted_sum(scored, cfg["investment_screening_score"])
    scored["diligence_priority_score"] = weighted_sum(scored, cfg["diligence_priority_score"])
    for column in SCORE_COLUMNS:
        scored[column] = scored[column].apply(clip_score).round(1)
    scored["investment_screening_explanation"] = scored.apply(_score_explanation, axis=1)
    scored["platform_candidate_explanation"] = scored.apply(
        lambda row: "Platform screen reflects scale, margins, cash conversion and leverage discipline; it is not a transaction recommendation.",
        axis=1,
    )
    scored["value_creation_explanation"] = scored.apply(
        lambda row: "Value creation screen looks for peer margin or cash-conversion gaps with enough growth and risk control for further diligence.",
        axis=1,
    )
    scored["red_flag_explanation"] = scored.apply(
        lambda row: "Red flag screen combines leverage, weak cash conversion, liquidity, valuation support, volatility, trend and data-gap indicators.",
        axis=1,
    )
    return scored


def run_scoring() -> pd.DataFrame:
    ensure_project_dirs()
    screening = read_csv_if_exists(PROCESSED_DIR / "screening_universe.csv")
    if screening.empty:
        screening = run_feature_engineering()
    benchmarks = read_csv_if_exists(PROCESSED_DIR / "peer_benchmarks.csv")
    if benchmarks.empty:
        benchmarks = run_peer_benchmarking()
    scored = calculate_scores(screening, benchmarks)
    write_csv(scored, PROCESSED_DIR / "investment_scores.csv")
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate transparent screening scorecards.")
    parser.parse_args()
    frame = run_scoring()
    print(f"scored_companies={len(frame)}")
    print(f"top_investment_screening_score={frame['investment_screening_score'].max():.1f}")


if __name__ == "__main__":
    main()
