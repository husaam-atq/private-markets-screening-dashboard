from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, SAMPLE_DIR, ensure_project_dirs
from src.universe import configured_universe, runtime_limit, sample_universe
from src.utils import (
    read_csv_if_exists,
    skipped_ticker_row,
    upsert_skipped_tickers,
    utc_timestamp,
    write_csv,
)


MARKET_COLUMNS = [
    "ticker",
    "company_name",
    "sector_theme",
    "sector",
    "industry",
    "market_cap",
    "enterprise_value",
    "latest_price",
    "price_52w_high",
    "price_52w_low",
    "drawdown_from_52w_high",
    "realised_volatility",
    "beta",
    "ev_to_sales_yfinance",
    "trailing_pe",
    "source_type",
    "data_mode",
    "data_source",
    "market_data_timestamp",
]


SECTOR_PROFILES = {
    "Software / SaaS": {"mc": 180, "evs": 9.0, "vol": 0.34, "beta": 1.12, "draw": 0.18},
    "Healthcare Services": {"mc": 95, "evs": 1.2, "vol": 0.21, "beta": 0.78, "draw": 0.12},
    "Business Services": {"mc": 70, "evs": 5.2, "vol": 0.22, "beta": 0.86, "draw": 0.13},
    "Industrials": {"mc": 62, "evs": 3.1, "vol": 0.25, "beta": 1.01, "draw": 0.15},
    "Consumer Services": {"mc": 58, "evs": 4.2, "vol": 0.28, "beta": 1.04, "draw": 0.17},
    "Energy Transition": {"mc": 44, "evs": 4.5, "vol": 0.46, "beta": 1.35, "draw": 0.32},
    "Digital Infrastructure / Telecom Infrastructure": {
        "mc": 78,
        "evs": 6.0,
        "vol": 0.24,
        "beta": 0.84,
        "draw": 0.16,
    },
    "Financial Technology": {"mc": 92, "evs": 6.3, "vol": 0.39, "beta": 1.25, "draw": 0.25},
    "Payments / Financial Services": {"mc": 92, "evs": 5.8, "vol": 0.34, "beta": 1.15, "draw": 0.20},
    "Alternative Asset Managers / Market Infrastructure": {"mc": 80, "evs": 4.6, "vol": 0.29, "beta": 1.05, "draw": 0.17},
    "Infrastructure / Utilities-Like Assets": {"mc": 40, "evs": 4.4, "vol": 0.18, "beta": 0.62, "draw": 0.10},
    "Real Estate / REITs": {"mc": 36, "evs": 7.5, "vol": 0.23, "beta": 0.78, "draw": 0.15},
    "Travel / Leisure / Consumer Platforms": {"mc": 48, "evs": 4.8, "vol": 0.35, "beta": 1.18, "draw": 0.22},
}

TICKER_ADJUSTMENTS = {
    "MSFT": (17.0, 0.08, -0.08),
    "ADBE": (5.0, 0.02, 0.01),
    "NOW": (3.0, 0.06, 0.01),
    "UNH": (5.5, -0.01, -0.02),
    "HCA": (1.6, 0.02, 0.00),
    "ADP": (2.5, 0.00, -0.03),
    "CTAS": (2.0, 0.01, -0.02),
    "ETN": (2.2, 0.05, -0.01),
    "PH": (1.4, 0.03, -0.01),
    "BKNG": (1.5, 0.01, 0.00),
    "MCD": (1.8, -0.02, -0.03),
    "FSLR": (1.1, 0.08, 0.06),
    "AES": (0.7, -0.03, 0.03),
    "TSLA": (6.5, 0.06, 0.04),
    "EQIX": (1.9, 0.01, -0.02),
    "DLR": (0.9, -0.01, 0.01),
    "V": (5.5, 0.02, -0.04),
    "MA": (5.0, 0.02, -0.04),
    "PYPL": (0.8, -0.04, 0.08),
    "SOFI": (0.25, 0.04, 0.13),
    "COIN": (0.45, 0.05, 0.18),
    "FIS": (0.8, -0.02, 0.05),
}


def _deterministic_noise(ticker: str, scale: float = 1.0) -> float:
    value = sum((idx + 1) * ord(char) for idx, char in enumerate(ticker))
    return ((value % 23) - 11) / 100 * scale


def build_sample_market_data(universe: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    universe = sample_universe() if universe is None else universe.copy()
    rows: list[dict] = []
    timestamp = utc_timestamp()
    for order, row in universe.reset_index(drop=True).iterrows():
        profile = SECTOR_PROFILES[row["sector_theme"]]
        scale, growth_bias, risk_bias = TICKER_ADJUSTMENTS.get(row["ticker"], (1.0, 0.0, 0.0))
        noise = _deterministic_noise(row["ticker"], 1.4)
        market_cap = max(1.2, profile["mc"] * scale * (1 + noise)) * 1_000_000_000
        ev_sales = max(0.4, profile["evs"] * (1 + growth_bias - risk_bias / 2 + noise / 2))
        enterprise_value = market_cap * (1.06 + max(risk_bias, -0.04))
        latest_price = max(4.0, 35 + order * 7.5 + market_cap / 50_000_000_000)
        drawdown = min(0.72, max(0.02, profile["draw"] + risk_bias + _deterministic_noise(row["ticker"], 0.6)))
        high = latest_price / (1 - drawdown)
        low = latest_price * (0.62 + max(0, 0.18 - drawdown / 2))
        volatility = min(0.85, max(0.10, profile["vol"] + risk_bias + _deterministic_noise(row["ticker"], 0.5)))
        beta = min(2.4, max(0.35, profile["beta"] + risk_bias + _deterministic_noise(row["ticker"], 0.7)))
        rows.append(
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "sector_theme": row["sector_theme"],
                "sector": row["sector"],
                "industry": row["industry"],
                "market_cap": market_cap,
                "enterprise_value": enterprise_value,
                "latest_price": latest_price,
                "price_52w_high": high,
                "price_52w_low": low,
                "drawdown_from_52w_high": drawdown,
                "realised_volatility": volatility,
                "beta": beta,
                "ev_to_sales_yfinance": ev_sales,
                "trailing_pe": max(5.0, 18 + ev_sales * 2 + growth_bias * 50),
                "source_type": "sample_fallback",
                "data_mode": "sample_fallback",
                "data_source": "cached_sample",
                "market_data_timestamp": timestamp,
            }
        )
    frame = pd.DataFrame(rows)[MARKET_COLUMNS]
    write_csv(frame, SAMPLE_DIR / "sample_market_data.csv")
    return frame


class YFinanceClient:
    def __init__(self, cache_dir: Path | None = None, sleep_seconds: float = 0.05) -> None:
        self.cache_dir = cache_dir or RAW_DIR / "yfinance"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds

    def fetch_ticker(self, ticker: str) -> dict:
        try:
            import yfinance as yf

            obj = yf.Ticker(ticker)
            info = obj.get_info()
            hist = obj.history(period="1y", auto_adjust=True)
            time.sleep(self.sleep_seconds)
            if hist.empty:
                raise ValueError("missing price history")
            latest_price = float(hist["Close"].iloc[-1])
            high = float(hist["Close"].max())
            low = float(hist["Close"].min())
            returns = hist["Close"].pct_change().dropna()
            vol = float(returns.std() * np.sqrt(252)) if len(returns) else np.nan
            return {
                "ticker": ticker,
                "company_name": info.get("longName") or info.get("shortName") or ticker,
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "latest_price": latest_price,
                "price_52w_high": high,
                "price_52w_low": low,
                "drawdown_from_52w_high": (high - latest_price) / high if high else np.nan,
                "realised_volatility": vol,
                "beta": info.get("beta"),
                "ev_to_sales_yfinance": info.get("enterpriseToRevenue"),
                "trailing_pe": info.get("trailingPE"),
                "source_type": "live_yfinance",
                "data_mode": "live_api",
                "data_source": "yfinance",
                "market_data_timestamp": utc_timestamp(),
            }
        except Exception as exc:  # pragma: no cover - network instability path
            return {
                "ticker": ticker,
                "source_type": "missing_yfinance",
                "data_mode": "live_api_failed",
                "data_source": f"yfinance_failed: {type(exc).__name__}",
                "skip_reason": type(exc).__name__,
            }

    def fetch_universe(self, universe: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
        rows: list[dict] = []
        frame = universe.head(limit) if limit else universe
        for _, base in frame.iterrows():
            fetched = self.fetch_ticker(str(base["ticker"]))
            merged = base.to_dict() | fetched
            rows.append(merged)
        result = pd.DataFrame(rows)
        for column in MARKET_COLUMNS:
            if column not in result.columns:
                result[column] = np.nan
        return result[MARKET_COLUMNS]


def load_market_data(mode: str = "sample", limit: int | None = None, runtime_mode: str | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    sample_path = SAMPLE_DIR / "sample_market_data.csv"
    if mode == "online":
        universe = configured_universe()
        limit = runtime_limit(runtime_mode, limit)
        online = YFinanceClient().fetch_universe(universe, limit=limit)
        required = ["market_cap", "enterprise_value", "latest_price", "price_52w_high", "price_52w_low"]
        valid_mask = online[required].notna().sum(axis=1) >= 3
        skipped_rows = []
        for _, row in online[~valid_mask].iterrows():
            skipped_rows.append(
                skipped_ticker_row(
                    ticker=row["ticker"],
                    company_name=row.get("company_name", ""),
                    stage="yfinance",
                    reason=row.get("skip_reason") or "missing_market_fields",
                    detail=row.get("data_source", ""),
                )
            )
        upsert_skipped_tickers(skipped_rows, PROCESSED_DIR / "skipped_tickers.csv")
        valid = online[valid_mask].copy()
        minimum_valid = 60 if limit is None or limit >= 100 else max(3, int((limit or len(online)) * 0.60))
        if len(valid) >= minimum_valid:
            write_csv(valid, INTERIM_DIR / "market_data_snapshot.csv")
            return valid
    cached = read_csv_if_exists(sample_path)
    expected_tickers = set(sample_universe()["ticker"])
    cached_tickers = set(cached["ticker"]) if not cached.empty and "ticker" in cached.columns else set()
    if cached.empty or cached_tickers != expected_tickers:
        cached = build_sample_market_data()
    cached["data_mode"] = "sample_fallback"
    cached["source_type"] = "sample_fallback"
    write_csv(cached, INTERIM_DIR / "market_data_snapshot.csv")
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or build market data snapshots.")
    parser.add_argument("--mode", choices=["sample", "online"], default="online")
    parser.add_argument("--runtime-mode", choices=["demo", "portfolio", "extended"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    frame = load_market_data(mode=args.mode, limit=args.limit, runtime_mode=args.runtime_mode)
    print(f"market_data_rows={len(frame)}")
    print(f"market_data_source={frame['data_source'].mode().iloc[0] if not frame.empty else 'none'}")


if __name__ == "__main__":
    main()
