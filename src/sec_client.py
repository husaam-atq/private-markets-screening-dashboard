from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, SAMPLE_DIR, ensure_project_dirs, load_config, sec_user_agent
from src.sec_xbrl_mapper import normalize_companyfacts_payload
from src.universe import configured_universe, runtime_mode_config, sample_universe
from src.utils import read_csv_if_exists, upsert_skipped_tickers, utc_timestamp, write_csv
from src.yfinance_client import build_sample_market_data


SAMPLE_CIKS = {
    "MSFT": 789019,
    "CRM": 1108524,
    "ADBE": 796343,
    "NOW": 1373715,
    "INTU": 896878,
    "UNH": 731766,
    "ELV": 1156039,
    "HCA": 860730,
    "DVA": 927066,
    "CVS": 64803,
    "ADP": 8670,
    "PAYX": 723531,
    "FIS": 1136893,
    "GPN": 1123360,
    "CTAS": 723254,
    "HON": 773840,
    "ETN": 1551182,
    "PH": 76334,
    "CARR": 1783180,
    "CAT": 18230,
    "MCD": 63908,
    "SBUX": 829224,
    "BKNG": 1075531,
    "MAR": 1048286,
    "HLT": 1585689,
    "NEE": 753308,
    "FSLR": 1274494,
    "AES": 874761,
    "TSLA": 1318605,
    "ALB": 915913,
    "AMT": 1053507,
    "EQIX": 1101239,
    "DLR": 1297996,
    "VZ": 732712,
    "TMUS": 1283699,
    "V": 1403161,
    "MA": 1141391,
    "PYPL": 1633917,
    "SOFI": 1818874,
    "COIN": 1679788,
}

FUNDAMENTAL_PROFILES = {
    "Software / SaaS": {"growth": 0.13, "gross": 0.72, "op": 0.30, "ocf": 0.36, "debt": 0.9, "capex": 0.05},
    "Healthcare Services": {"growth": 0.07, "gross": 0.28, "op": 0.075, "ocf": 0.095, "debt": 2.1, "capex": 0.025},
    "Business Services": {"growth": 0.08, "gross": 0.46, "op": 0.24, "ocf": 0.28, "debt": 1.8, "capex": 0.04},
    "Industrials": {"growth": 0.06, "gross": 0.36, "op": 0.18, "ocf": 0.20, "debt": 1.7, "capex": 0.045},
    "Consumer Services": {"growth": 0.07, "gross": 0.42, "op": 0.20, "ocf": 0.21, "debt": 2.4, "capex": 0.055},
    "Energy Transition": {"growth": 0.11, "gross": 0.30, "op": 0.09, "ocf": 0.10, "debt": 3.0, "capex": 0.12},
    "Digital Infrastructure / Telecom Infrastructure": {"growth": 0.06, "gross": 0.52, "op": 0.17, "ocf": 0.28, "debt": 4.4, "capex": 0.18},
    "Financial Technology": {"growth": 0.10, "gross": 0.58, "op": 0.24, "ocf": 0.27, "debt": 1.2, "capex": 0.035},
    "Payments / Financial Services": {"growth": 0.08, "gross": 0.54, "op": 0.22, "ocf": 0.25, "debt": 1.4, "capex": 0.035},
    "Alternative Asset Managers / Market Infrastructure": {"growth": 0.07, "gross": 0.50, "op": 0.25, "ocf": 0.27, "debt": 1.6, "capex": 0.025},
    "Real Estate / REITs": {"growth": 0.05, "gross": 0.62, "op": 0.23, "ocf": 0.34, "debt": 4.0, "capex": 0.11},
    "Travel / Leisure / Consumer Platforms": {"growth": 0.08, "gross": 0.42, "op": 0.16, "ocf": 0.20, "debt": 2.5, "capex": 0.06},
}

TICKER_FUNDAMENTAL_ADJUSTMENTS = {
    "MSFT": {"growth": 0.06, "op": 0.14, "ocf": 0.13, "debt": -0.6, "gross": 0.06},
    "CRM": {"growth": 0.02, "op": 0.03, "ocf": 0.07, "debt": -0.3},
    "NOW": {"growth": 0.09, "op": 0.02, "ocf": 0.08, "debt": -0.4},
    "UNH": {"growth": 0.03, "op": 0.01, "ocf": 0.02, "debt": -0.3},
    "HCA": {"growth": 0.02, "op": 0.02, "ocf": 0.03, "debt": 0.4},
    "DVA": {"growth": -0.02, "op": -0.01, "debt": 1.6, "ocf": -0.02},
    "ADP": {"growth": 0.02, "op": 0.06, "ocf": 0.05, "debt": -0.5, "gross": 0.03},
    "CTAS": {"growth": 0.03, "op": 0.05, "ocf": 0.04, "debt": -0.4},
    "ETN": {"growth": 0.04, "op": 0.05, "ocf": 0.03, "debt": -0.3},
    "PH": {"growth": 0.02, "op": 0.04, "ocf": 0.03},
    "CARR": {"growth": 0.01, "op": 0.02, "debt": 0.3},
    "BKNG": {"growth": 0.04, "op": 0.09, "ocf": 0.08, "debt": -0.2, "gross": 0.12},
    "MCD": {"growth": 0.00, "op": 0.14, "debt": 1.0, "ocf": 0.07},
    "FSLR": {"growth": 0.12, "gross": 0.08, "op": 0.06, "debt": -0.6, "capex": 0.04},
    "AES": {"growth": -0.03, "op": -0.02, "debt": 1.7, "capex": 0.05, "ocf": -0.02},
    "TSLA": {"growth": 0.07, "gross": -0.04, "op": -0.01, "debt": -0.4, "capex": 0.04},
    "ALB": {"growth": -0.06, "gross": -0.07, "op": -0.06, "debt": 0.5, "ocf": -0.04},
    "EQIX": {"growth": 0.04, "op": 0.04, "ocf": 0.08, "debt": -0.2, "capex": -0.03},
    "DLR": {"growth": 0.02, "op": -0.02, "debt": 0.7, "capex": 0.03},
    "VZ": {"growth": -0.01, "op": 0.02, "debt": 1.3, "ocf": 0.03},
    "TMUS": {"growth": 0.03, "op": 0.03, "debt": 0.8, "ocf": 0.02},
    "V": {"growth": 0.04, "gross": 0.20, "op": 0.34, "ocf": 0.30, "debt": -0.7, "capex": -0.02},
    "MA": {"growth": 0.04, "gross": 0.18, "op": 0.31, "ocf": 0.28, "debt": -0.7, "capex": -0.02},
    "PYPL": {"growth": -0.01, "op": -0.01, "ocf": 0.03, "debt": -0.2},
    "SOFI": {"growth": 0.10, "op": -0.14, "ocf": -0.08, "debt": 1.1},
    "COIN": {"growth": 0.06, "op": -0.06, "ocf": -0.02, "debt": 0.4},
}


def _noise(ticker: str, scale: float = 1.0) -> float:
    return (((sum(ord(c) for c in ticker) % 17) - 8) / 100) * scale


def build_sample_sec_fundamentals(universe: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    universe = sample_universe() if universe is None else universe.copy()
    market = read_csv_if_exists(SAMPLE_DIR / "sample_market_data.csv")
    if market.empty:
        market = build_sample_market_data(universe)
    market_lookup = market.set_index("ticker")
    rows: list[dict[str, Any]] = []
    for _, company in universe.iterrows():
        ticker = company["ticker"]
        profile = FUNDAMENTAL_PROFILES[company["sector_theme"]].copy()
        for key, value in TICKER_FUNDAMENTAL_ADJUSTMENTS.get(ticker, {}).items():
            profile[key] = profile.get(key, 0) + value
        market_row = market_lookup.loc[ticker]
        latest_revenue = float(market_row["enterprise_value"]) / max(0.35, float(market_row["ev_to_sales_yfinance"]))
        latest_revenue = max(800_000_000, latest_revenue)
        for year_offset, fiscal_year in enumerate([2022, 2023, 2024]):
            years_back = 2 - year_offset
            growth = max(-0.15, min(0.32, profile["growth"] + _noise(ticker, 0.8)))
            revenue = latest_revenue / ((1 + growth) ** years_back)
            gross_margin = max(0.05, min(0.92, profile["gross"] + _noise(ticker, 0.5) - years_back * 0.002))
            operating_margin = max(-0.20, min(0.72, profile["op"] + _noise(ticker, 0.4) + year_offset * 0.004))
            net_margin = operating_margin - 0.035 - max(0, profile["debt"]) * 0.004
            ocf_margin = max(-0.15, min(0.70, profile["ocf"] + _noise(ticker, 0.4) + year_offset * 0.003))
            debt_to_ebit = max(0.0, profile["debt"] + _noise(ticker, 2.0) + years_back * 0.05)
            operating_income = revenue * operating_margin
            gross_profit = revenue * gross_margin
            net_income = revenue * net_margin
            total_debt = max(0.0, operating_income * debt_to_ebit) if operating_income > 0 else revenue * 0.32
            cash = revenue * max(0.025, min(0.35, 0.11 - profile["debt"] * 0.008 + _noise(ticker, 0.4)))
            capex = -abs(revenue * max(0.01, profile["capex"] + _noise(ticker, 0.25)))
            ocf = revenue * ocf_margin
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company["company_name"],
                    "cik": SAMPLE_CIKS.get(ticker, 0),
                    "fiscal_year": fiscal_year,
                    "filing_type": "10-K",
                    "filing_date": f"{fiscal_year + 1}-02-15",
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "operating_income": operating_income,
                    "net_income": net_income,
                    "cash_and_equivalents": cash,
                    "total_assets": revenue * (1.15 + profile["debt"] * 0.10 + profile["capex"]),
                    "total_liabilities": revenue * (0.42 + profile["debt"] * 0.10),
                    "long_term_debt": total_debt * 0.86,
                    "current_debt": total_debt * 0.14,
                    "total_debt": total_debt,
                    "operating_cash_flow": ocf,
                    "capex": capex,
                    "shares_outstanding": max(20_000_000, float(market_row["market_cap"]) / float(market_row["latest_price"])),
                    "depreciation_amortization": abs(capex) * 0.65,
                    "interest_expense": total_debt * 0.045,
                    "source_type": "sample_fallback",
                    "data_mode": "sample_fallback",
                    "data_source": "cached_sec_xbrl_sample",
                    "fundamentals_refresh_timestamp": utc_timestamp(),
                }
            )
    frame = pd.DataFrame(rows)
    write_csv(frame, SAMPLE_DIR / "sample_sec_fundamentals.csv")
    return frame


class SECClient:
    def __init__(self, cache_dir: Path | None = None) -> None:
        self.config = load_config("sec_config.yaml")
        universe_cfg = load_config("universe_config.yaml")
        self.cache_dir = cache_dir or RAW_DIR
        self.headers = {
            "User-Agent": sec_user_agent(),
            "Accept-Encoding": "gzip, deflate",
        }
        self.timeout = int(self.config.get("request_timeout_seconds", 20))
        rate_limit = float(universe_cfg.get("sec_rate_limit_per_second", 8) or 8)
        configured_sleep = float(self.config.get("request_sleep_seconds", 0.12))
        self.sleep_seconds = max(configured_sleep, 1 / max(rate_limit, 1))

    @staticmethod
    def cik_padded(cik: int | str) -> str:
        return str(int(cik)).zfill(10)

    def _get_json(self, url: str, cache_path: Path) -> dict[str, Any]:
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        response = requests.get(url, headers=self.headers, timeout=self.timeout)
        response.raise_for_status()
        time.sleep(self.sleep_seconds)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(response.json(), handle)
        return response.json()

    def get_cik_mapping(self) -> pd.DataFrame:
        cache_path = RAW_DIR / "sec_submissions" / "company_tickers.json"
        try:
            payload = self._get_json(self.config["cik_mapping_url"], cache_path)
            rows = [
                {
                    "ticker": item["ticker"],
                    "company_name": item["title"],
                    "cik": int(item["cik_str"]),
                }
                for item in payload.values()
            ]
            return pd.DataFrame(rows)
        except Exception:  # pragma: no cover - network fallback
            return pd.DataFrame(
                [
                    {"ticker": ticker, "company_name": ticker, "cik": cik}
                    for ticker, cik in SAMPLE_CIKS.items()
                ]
            )

    def get_submissions(self, cik: int | str) -> dict[str, Any]:
        cik10 = self.cik_padded(cik)
        url = f"{self.config['base_url']}/submissions/CIK{cik10}.json"
        cache_path = RAW_DIR / "sec_submissions" / f"CIK{cik10}.json"
        return self._get_json(url, cache_path)

    def get_companyfacts(self, cik: int | str) -> dict[str, Any]:
        cik10 = self.cik_padded(cik)
        url = f"{self.config['base_url']}/api/xbrl/companyfacts/CIK{cik10}.json"
        cache_path = RAW_DIR / "sec_companyfacts" / f"CIK{cik10}.json"
        return self._get_json(url, cache_path)

    def fetch_fundamentals(self, universe: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
        mapping = self.get_cik_mapping()
        mapping_lookup = mapping.drop_duplicates("ticker").set_index("ticker")
        rows: list[pd.DataFrame] = []
        skipped: list[dict[str, Any]] = []
        frame = universe.head(limit) if limit else universe
        for _, company in frame.iterrows():
            ticker = str(company["ticker"])
            if ticker not in mapping_lookup.index:
                skipped.append(
                    {
                        "ticker": ticker,
                        "company_name": company.get("company_name", ""),
                        "stage": "sec_companyfacts",
                        "stage_failed": "sec_companyfacts",
                        "reason": "missing_cik_mapping",
                        "reason_skipped": "missing_cik_mapping",
                        "detail": "Ticker not found in SEC company_tickers mapping.",
                        "logged_at": utc_timestamp(),
                        "timestamp": utc_timestamp(),
                    }
                )
                continue
            cik = int(mapping_lookup.loc[ticker]["cik"])
            try:
                payload = self.get_companyfacts(cik)
                normalized = normalize_companyfacts_payload(payload, ticker, company["company_name"], cik)
                if not normalized.empty:
                    normalized["sector_theme"] = company["sector_theme"]
                    normalized["source_type"] = "live_sec_companyfacts"
                    normalized["data_mode"] = "live_api"
                    normalized["fundamentals_refresh_timestamp"] = utc_timestamp()
                    rows.append(normalized)
                else:
                    skipped.append(
                        {
                            "ticker": ticker,
                            "company_name": company.get("company_name", ""),
                            "stage": "sec_companyfacts",
                            "stage_failed": "sec_companyfacts",
                            "reason": "no_mappable_xbrl_facts",
                            "reason_skipped": "no_mappable_xbrl_facts",
                            "detail": f"CIK {cik} returned no mappable annual companyfacts.",
                            "logged_at": utc_timestamp(),
                            "timestamp": utc_timestamp(),
                        }
                    )
            except Exception:
                skipped.append(
                    {
                        "ticker": ticker,
                        "company_name": company.get("company_name", ""),
                        "stage": "sec_companyfacts",
                        "stage_failed": "sec_companyfacts",
                        "reason": "companyfacts_request_failed",
                        "reason_skipped": "companyfacts_request_failed",
                        "detail": f"CIK {cik}",
                        "logged_at": utc_timestamp(),
                        "timestamp": utc_timestamp(),
                    }
                )
                continue
        upsert_skipped_tickers(skipped, PROCESSED_DIR / "skipped_tickers.csv")
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _runtime_limit(runtime_mode: str | None, explicit_limit: int | None) -> int | None:
    if explicit_limit is not None:
        return explicit_limit
    config = runtime_mode_config(runtime_mode)
    max_companies = config.get("max_companies")
    return int(max_companies) if max_companies else None


def load_sec_fundamentals(mode: str = "sample", limit: int | None = None, runtime_mode: str | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    sample_path = SAMPLE_DIR / "sample_sec_fundamentals.csv"
    if mode == "online":
        limit = _runtime_limit(runtime_mode, limit)
        online = SECClient().fetch_fundamentals(configured_universe(), limit=limit)
        if not online.empty and online["ticker"].nunique() >= 5:
            write_csv(online, INTERIM_DIR / "normalized_fundamentals.csv")
            return online
    cached = read_csv_if_exists(sample_path)
    expected_tickers = set(sample_universe()["ticker"])
    cached_tickers = set(cached["ticker"]) if not cached.empty and "ticker" in cached.columns else set()
    if cached.empty or cached_tickers != expected_tickers:
        cached = build_sample_sec_fundamentals()
    if "source_type" not in cached.columns:
        cached["source_type"] = "sample_fallback"
    if "data_mode" not in cached.columns:
        cached["data_mode"] = "sample_fallback"
    write_csv(cached, INTERIM_DIR / "normalized_fundamentals.csv")
    return cached


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or build SEC fundamentals snapshots.")
    parser.add_argument("--mode", choices=["sample", "online"], default="online")
    parser.add_argument("--runtime-mode", choices=["demo", "portfolio", "extended"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    frame = load_sec_fundamentals(mode=args.mode, limit=args.limit, runtime_mode=args.runtime_mode)
    companies = frame["ticker"].nunique() if not frame.empty else 0
    print(f"sec_fundamental_rows={len(frame)}")
    print(f"sec_fundamental_companies={companies}")


if __name__ == "__main__":
    main()
