from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CHART_DIR = OUTPUT_DIR / "charts"
REPORT_DIR = OUTPUT_DIR / "reports"
EXCEL_DIR = OUTPUT_DIR / "excel"
EVAL_DIR = PROJECT_ROOT / "eval"


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.exists():
        logger.error("Config file not found: %s", resolved)
        raise FileNotFoundError(f"Config file not found: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except yaml.YAMLError as exc:
        logger.error("Failed to parse YAML config %s: %s", resolved, exc)
        raise ValueError(f"Invalid YAML in {resolved}: {exc}") from exc
    return data


def load_config(name: str) -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / name)


def get_env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)


def sec_user_agent() -> str:
    cfg = load_config("sec_config.yaml")
    configured = get_env("SEC_USER_AGENT") or cfg.get("sec_user_agent")
    if not configured or "example.com" in configured:
        return "private-markets-screening-dashboard contact@example.com"
    return str(configured)


def ensure_project_dirs() -> None:
    directories = [
        RAW_DIR / "yfinance",
        RAW_DIR / "sec_submissions",
        RAW_DIR / "sec_companyfacts",
        RAW_DIR / "filings",
        INTERIM_DIR / "retrieval_index",
        PROCESSED_DIR,
        SAMPLE_DIR,
        CHART_DIR,
        REPORT_DIR,
        EXCEL_DIR,
        EVAL_DIR,
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
