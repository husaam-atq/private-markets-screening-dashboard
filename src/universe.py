from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from src.config import INTERIM_DIR, SAMPLE_DIR, ensure_project_dirs, load_config
from src.utils import utc_timestamp, write_csv


SECTOR_METADATA = {
    "Software / SaaS": ("Information Technology", "Application Software"),
    "Healthcare Services": ("Health Care", "Managed Care and Services"),
    "Business Services": ("Industrials", "Professional and Commercial Services"),
    "Industrials": ("Industrials", "Industrial Products"),
    "Consumer Services": ("Consumer Discretionary", "Consumer Services"),
    "Energy Transition": ("Utilities / Energy", "Renewables and Electrification"),
    "Digital Infrastructure / Telecom Infrastructure": ("Real Estate / Communications", "Digital Infrastructure"),
    "Financial Technology": ("Financials", "Payments and Financial Technology"),
    "Infrastructure / Utilities-Like Assets": ("Utilities", "Regulated and Contracted Infrastructure"),
}


@dataclass(frozen=True)
class UniverseCompany:
    ticker: str
    company_name: str
    sector_theme: str
    sector: str
    industry: str


COMPANY_NAMES = {
    "MSFT": "Microsoft Corporation",
    "CRM": "Salesforce, Inc.",
    "ADBE": "Adobe Inc.",
    "NOW": "ServiceNow, Inc.",
    "INTU": "Intuit Inc.",
    "SNOW": "Snowflake Inc.",
    "DDOG": "Datadog, Inc.",
    "ZS": "Zscaler, Inc.",
    "MDB": "MongoDB, Inc.",
    "TEAM": "Atlassian Corporation",
    "NET": "Cloudflare, Inc.",
    "HUBS": "HubSpot, Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "ELV": "Elevance Health, Inc.",
    "CI": "The Cigna Group",
    "HUM": "Humana Inc.",
    "HCA": "HCA Healthcare, Inc.",
    "THC": "Tenet Healthcare Corporation",
    "DVA": "DaVita Inc.",
    "CVS": "CVS Health Corporation",
    "LH": "Labcorp Holdings Inc.",
    "DGX": "Quest Diagnostics Incorporated",
    "CNC": "Centene Corporation",
    "MOH": "Molina Healthcare, Inc.",
    "ADP": "Automatic Data Processing, Inc.",
    "PAYX": "Paychex, Inc.",
    "FI": "Fiserv, Inc.",
    "GPN": "Global Payments Inc.",
    "CTAS": "Cintas Corporation",
    "ROP": "Roper Technologies, Inc.",
    "WM": "Waste Management, Inc.",
    "RSG": "Republic Services, Inc.",
    "SPGI": "S&P Global Inc.",
    "MCO": "Moody's Corporation",
    "VRSK": "Verisk Analytics, Inc.",
    "TRI": "Thomson Reuters Corporation",
    "HON": "Honeywell International Inc.",
    "ETN": "Eaton Corporation plc",
    "EMR": "Emerson Electric Co.",
    "PH": "Parker-Hannifin Corporation",
    "ITW": "Illinois Tool Works Inc.",
    "MMM": "3M Company",
    "GE": "GE Aerospace",
    "CARR": "Carrier Global Corporation",
    "TT": "Trane Technologies plc",
    "IR": "Ingersoll Rand Inc.",
    "DE": "Deere & Company",
    "CAT": "Caterpillar Inc.",
    "MCD": "McDonald's Corporation",
    "SBUX": "Starbucks Corporation",
    "BKNG": "Booking Holdings Inc.",
    "MAR": "Marriott International, Inc.",
    "HLT": "Hilton Worldwide Holdings Inc.",
    "CMG": "Chipotle Mexican Grill, Inc.",
    "YUM": "Yum! Brands, Inc.",
    "RCL": "Royal Caribbean Cruises Ltd.",
    "CCL": "Carnival Corporation",
    "LYV": "Live Nation Entertainment, Inc.",
    "DPZ": "Domino's Pizza, Inc.",
    "ROST": "Ross Stores, Inc.",
    "NEE": "NextEra Energy, Inc.",
    "ENPH": "Enphase Energy, Inc.",
    "FSLR": "First Solar, Inc.",
    "SEDG": "SolarEdge Technologies, Inc.",
    "BE": "Bloom Energy Corporation",
    "PLUG": "Plug Power Inc.",
    "TSLA": "Tesla, Inc.",
    "AES": "The AES Corporation",
    "RUN": "Sunrun Inc.",
    "CSIQ": "Canadian Solar Inc.",
    "ALB": "Albemarle Corporation",
    "ORA": "Ormat Technologies, Inc.",
    "AMT": "American Tower Corporation",
    "CCI": "Crown Castle Inc.",
    "EQIX": "Equinix, Inc.",
    "DLR": "Digital Realty Trust, Inc.",
    "SBAC": "SBA Communications Corporation",
    "VZ": "Verizon Communications Inc.",
    "T": "AT&T Inc.",
    "TMUS": "T-Mobile US, Inc.",
    "LUMN": "Lumen Technologies, Inc.",
    "FYBR": "Frontier Communications Parent, Inc.",
    "UNIT": "Uniti Group Inc.",
    "IRM": "Iron Mountain Incorporated",
    "V": "Visa Inc.",
    "MA": "Mastercard Incorporated",
    "PYPL": "PayPal Holdings, Inc.",
    "XYZ": "Block, Inc.",
    "AXP": "American Express Company",
    "COF": "Capital One Financial Corporation",
    "AFRM": "Affirm Holdings, Inc.",
    "SOFI": "SoFi Technologies, Inc.",
    "HOOD": "Robinhood Markets, Inc.",
    "COIN": "Coinbase Global, Inc.",
    "MQ": "Marqeta, Inc.",
    "UPST": "Upstart Holdings, Inc.",
    "DUK": "Duke Energy Corporation",
    "SO": "The Southern Company",
    "AEP": "American Electric Power Company, Inc.",
    "EXC": "Exelon Corporation",
    "XEL": "Xcel Energy Inc.",
    "SRE": "Sempra",
    "PEG": "Public Service Enterprise Group Incorporated",
    "WEC": "WEC Energy Group, Inc.",
    "AWK": "American Water Works Company, Inc.",
    "CNP": "CenterPoint Energy, Inc.",
    "NRG": "NRG Energy, Inc.",
    "ETR": "Entergy Corporation",
}


def load_universe_config() -> dict:
    return load_config("universe_config.yaml")


def configured_universe() -> pd.DataFrame:
    cfg = load_universe_config()
    rows: list[dict[str, str]] = []
    for theme in cfg["themes"]:
        sector, industry = SECTOR_METADATA.get(theme["name"], ("Unknown", "Unknown"))
        for ticker in theme["tickers"]:
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": COMPANY_NAMES.get(ticker, ticker),
                    "sector_theme": theme["name"],
                    "sector": sector,
                    "industry": industry,
                    "universe_type": "default_online",
                }
            )
    return pd.DataFrame(rows).drop_duplicates("ticker").sort_values(["sector_theme", "ticker"])


def sample_universe() -> pd.DataFrame:
    cfg = load_universe_config()
    sample_tickers = set(cfg["sample_tickers"])
    frame = configured_universe()
    sample = frame[frame["ticker"].isin(sample_tickers)].copy()
    sample["universe_type"] = "cached_sample"
    return sample.sort_values(["sector_theme", "ticker"]).reset_index(drop=True)


def write_universe_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    default = configured_universe()
    sample = sample_universe()
    default["universe_refresh_timestamp"] = utc_timestamp()
    sample["universe_refresh_timestamp"] = utc_timestamp()
    write_csv(default, INTERIM_DIR / "default_universe.csv")
    write_csv(sample, SAMPLE_DIR / "sample_universe.csv")
    return default, sample


def main() -> None:
    parser = argparse.ArgumentParser(description="Build configured company universes.")
    parser.add_argument("--show", action="store_true", help="Print a short universe summary.")
    args = parser.parse_args()
    default, sample = write_universe_outputs()
    if args.show:
        print(f"default_universe_companies={len(default)}")
        print(f"sample_universe_companies={len(sample)}")
        print(f"sample_sectors={sample['sector_theme'].nunique()}")


if __name__ == "__main__":
    main()
