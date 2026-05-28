from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ensure_project_dirs, load_config
from src.feature_engineering import run_feature_engineering
from src.peer_benchmarking import build_peer_benchmarks, run_peer_benchmarking
from src.utils import clip_score, percentile_score, read_csv_if_exists, safe_divide, weighted_sum, write_csv


SCORE_COLUMNS = [
    "investment_screening_score",
    "diligence_priority_score",
    "platform_candidate_score",
    "value_creation_potential_score",
    "credit_risk_score",
    "red_flag_score",
    "data_quality_score",
]


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
    scored["data_quality_score"] = (pd.to_numeric(scored["data_completeness_ratio"], errors="coerce").fillna(0) * 100).clip(0, 100)
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
    scored = _risk_scores(scored)
    scored = _peer_adjustment_scores(scored, benchmarks)
    scored["credit_risk_score"] = weighted_sum(scored, cfg["credit_risk_score"])
    scored["red_flag_score"] = weighted_sum(scored, cfg["red_flag_score"])
    scored["red_flag_inverse_score"] = 100 - scored["red_flag_score"]
    scored["platform_candidate_score"] = weighted_sum(scored, cfg["platform_candidate_score"])
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
