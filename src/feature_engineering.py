from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR, ensure_project_dirs
from src.sec_client import load_sec_fundamentals
from src.utils import non_null_ratio, read_csv_if_exists, safe_divide, utc_timestamp, write_csv
from src.yfinance_client import load_market_data


METRIC_COLUMNS = [
    "revenue_growth_yoy",
    "revenue_cagr_3y",
    "gross_margin",
    "operating_margin",
    "net_margin",
    "operating_cash_flow_conversion",
    "free_cash_flow_proxy",
    "fcf_conversion",
    "capex_intensity",
    "cash_to_revenue",
    "debt_to_ebit_proxy",
    "net_debt_to_ebit_proxy",
    "ev_to_sales",
    "ev_to_ebit_proxy",
    "market_cap",
    "enterprise_value",
    "drawdown_from_52w_high",
    "realised_volatility",
    "margin_trend",
    "revenue_trend",
    "leverage_trend",
    "data_completeness_ratio",
    "yfinance_field_completeness_ratio",
    "sec_field_completeness_ratio",
    "filing_staleness_days",
    "proxy_usage_penalty",
    "market_cap_band",
]


def _latest_and_history(fundamentals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fundamentals = fundamentals.copy()
    fundamentals["fiscal_year"] = pd.to_numeric(fundamentals["fiscal_year"], errors="coerce")
    fundamentals = fundamentals.dropna(subset=["fiscal_year"])
    latest = fundamentals.sort_values(["ticker", "fiscal_year"]).groupby("ticker").tail(1)
    history = fundamentals.sort_values(["ticker", "fiscal_year"])
    return latest.reset_index(drop=True), history.reset_index(drop=True)


def _company_metrics(row: pd.Series, history: pd.DataFrame) -> dict:
    ticker = row["ticker"]
    hist = history[history["ticker"] == ticker].sort_values("fiscal_year")
    latest = hist.iloc[-1]
    prior = hist.iloc[-2] if len(hist) >= 2 else latest
    oldest = hist.iloc[0]
    revenue = latest.get("revenue")
    ebit = latest.get("operating_income")
    total_debt = latest.get("total_debt")
    cash = latest.get("cash_and_equivalents")
    ocf = latest.get("operating_cash_flow")
    capex = latest.get("capex")
    prior_revenue = prior.get("revenue")
    oldest_revenue = oldest.get("revenue")
    latest_year = latest.get("fiscal_year")
    oldest_year = oldest.get("fiscal_year")
    latest_op_margin = safe_divide(latest.get("operating_income"), latest.get("revenue"))
    oldest_op_margin = safe_divide(oldest.get("operating_income"), oldest.get("revenue"))
    latest_leverage = safe_divide(total_debt, ebit)
    oldest_leverage = safe_divide(oldest.get("total_debt"), oldest.get("operating_income"))
    year_span = max(1, int(latest_year - oldest_year)) if not pd.isna(latest_year) and not pd.isna(oldest_year) else 1
    fcf = ocf + capex if not pd.isna(ocf) and not pd.isna(capex) else np.nan
    revenue_cagr = (safe_divide(revenue, oldest_revenue) ** (1 / year_span) - 1) if safe_divide(revenue, oldest_revenue) > 0 else np.nan
    return {
        "ticker": ticker,
        "fiscal_year": int(latest_year) if not pd.isna(latest_year) else np.nan,
        "filing_date": latest.get("filing_date"),
        "cik": latest.get("cik"),
        "revenue": revenue,
        "gross_profit": latest.get("gross_profit"),
        "operating_income": ebit,
        "net_income": latest.get("net_income"),
        "cash_and_equivalents": cash,
        "total_debt": total_debt,
        "operating_cash_flow": ocf,
        "capex": capex,
        "revenue_growth_yoy": safe_divide(revenue, prior_revenue) - 1 if safe_divide(revenue, prior_revenue) else np.nan,
        "revenue_cagr_3y": revenue_cagr,
        "gross_margin": safe_divide(latest.get("gross_profit"), revenue),
        "operating_margin": latest_op_margin,
        "net_margin": safe_divide(latest.get("net_income"), revenue),
        "operating_cash_flow_conversion": safe_divide(ocf, ebit),
        "free_cash_flow_proxy": fcf,
        "fcf_conversion": safe_divide(fcf, ebit),
        "capex_intensity": safe_divide(abs(capex), revenue),
        "cash_to_revenue": safe_divide(cash, revenue),
        "debt_to_ebit_proxy": latest_leverage,
        "net_debt_to_ebit_proxy": safe_divide(total_debt - cash, ebit),
        "margin_trend": latest_op_margin - oldest_op_margin if not pd.isna(latest_op_margin) and not pd.isna(oldest_op_margin) else np.nan,
        "revenue_trend": revenue_cagr,
        "leverage_trend": latest_leverage - oldest_leverage if not pd.isna(latest_leverage) and not pd.isna(oldest_leverage) else np.nan,
        "fundamentals_source_type": latest.get("source_type", "unknown"),
        "fundamentals_data_mode": latest.get("data_mode", "unknown"),
        "uses_ebit_proxy": True,
        "ebitda_available": False,
    }


def market_cap_band(enterprise_value: float | int | None, red_flag_score: float | int | None = None) -> str:
    if enterprise_value is None or pd.isna(enterprise_value):
        return "Insufficient Data"
    ev = float(enterprise_value)
    if red_flag_score is not None and not pd.isna(red_flag_score) and float(red_flag_score) >= 75:
        return "Distressed/special situations watchlist"
    if ev >= 150_000_000_000:
        return "Mega-cap benchmark / public quality comp"
    if ev >= 50_000_000_000:
        return "Large-cap strategic / unlikely PE target"
    if ev >= 10_000_000_000:
        return "Mid-cap possible public-to-private screen"
    if ev >= 1_000_000_000:
        return "Small/mid-cap platform candidate"
    return "Distressed/special situations watchlist"


def build_screening_universe(market_data: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    latest, history = _latest_and_history(fundamentals)
    metric_rows = [_company_metrics(row, history) for _, row in latest.iterrows()]
    metrics = pd.DataFrame(metric_rows)
    combined = market_data.merge(metrics, on="ticker", how="left", suffixes=("", "_sec"))
    combined["ev_to_sales"] = combined.apply(lambda row: safe_divide(row["enterprise_value"], row["revenue"]), axis=1)
    combined["ev_to_ebit_proxy"] = combined.apply(lambda row: safe_divide(row["enterprise_value"], row["operating_income"]), axis=1)
    market_cols = [
        "market_cap",
        "enterprise_value",
        "latest_price",
        "price_52w_high",
        "price_52w_low",
        "realised_volatility",
        "beta",
        "ev_to_sales_yfinance",
        "trailing_pe",
    ]
    sec_cols = [
        "revenue",
        "gross_profit",
        "operating_income",
        "net_income",
        "cash_and_equivalents",
        "total_debt",
        "operating_cash_flow",
        "capex",
        "revenue_growth_yoy",
        "gross_margin",
        "operating_margin",
        "operating_cash_flow_conversion",
        "fcf_conversion",
        "debt_to_ebit_proxy",
        "ev_to_sales",
    ]
    combined["yfinance_field_completeness_ratio"] = combined.apply(lambda row: non_null_ratio(row, market_cols), axis=1)
    combined["sec_field_completeness_ratio"] = combined.apply(lambda row: non_null_ratio(row, sec_cols), axis=1)
    combined["data_completeness_ratio"] = (
        combined["yfinance_field_completeness_ratio"] * 0.40
        + combined["sec_field_completeness_ratio"] * 0.60
    )
    filing_dates = pd.to_datetime(combined["filing_date"], errors="coerce", utc=True)
    combined["filing_staleness_days"] = (pd.Timestamp.now(tz="UTC") - filing_dates).dt.days
    combined["proxy_usage_penalty"] = np.where(combined.get("uses_ebit_proxy", True), 8.0, 0.0)
    combined["market_cap_band"] = combined["enterprise_value"].apply(market_cap_band)
    if "source_type" in combined.columns:
        combined = combined.rename(columns={"source_type": "market_source_type", "data_mode": "market_data_mode"})
    if "fundamentals_source_type" not in combined.columns:
        combined["fundamentals_source_type"] = "missing"
    if "fundamentals_data_mode" not in combined.columns:
        combined["fundamentals_data_mode"] = "missing"
    combined["source_type"] = combined.apply(
        lambda row: "live_public_data"
        if row.get("market_source_type") == "live_yfinance" and row.get("fundamentals_source_type") == "live_sec_companyfacts"
        else ("cached_sec_snapshot" if row.get("fundamentals_source_type") == "live_sec_companyfacts" else "sample_fallback"),
        axis=1,
    )
    combined["metrics_refresh_timestamp"] = utc_timestamp()
    return combined


def run_feature_engineering(mode: str = "sample") -> pd.DataFrame:
    ensure_project_dirs()
    market = read_csv_if_exists(INTERIM_DIR / "market_data_snapshot.csv")
    if market.empty:
        market = load_market_data(mode=mode)
    fundamentals = read_csv_if_exists(INTERIM_DIR / "normalized_fundamentals.csv")
    if fundamentals.empty:
        fundamentals = load_sec_fundamentals(mode=mode)
    frame = build_screening_universe(market, fundamentals)
    write_csv(frame, PROCESSED_DIR / "screening_universe.csv")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structured screening metrics.")
    parser.add_argument("--mode", choices=["sample", "online"], default="online")
    args = parser.parse_args()
    frame = run_feature_engineering(mode=args.mode)
    print(f"screening_universe_rows={len(frame)}")
    print(f"metrics_calculated={len(METRIC_COLUMNS)}")


if __name__ == "__main__":
    main()
