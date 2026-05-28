from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.config import RAW_DIR, ensure_project_dirs, load_config, sec_user_agent
from src.sec_client import SECClient
from src.utils import normalise_accession


def latest_filing_metadata(submissions: dict, forms: tuple[str, ...] = ("10-K", "10-Q")) -> pd.DataFrame:
    recent = submissions.get("filings", {}).get("recent", {})
    rows = []
    for idx, form in enumerate(recent.get("form", [])):
        if form not in forms:
            continue
        rows.append(
            {
                "form": form,
                "filing_date": recent.get("filingDate", [])[idx],
                "accession_number": recent.get("accessionNumber", [])[idx],
                "primary_document": recent.get("primaryDocument", [])[idx],
            }
        )
    return pd.DataFrame(rows)


def filing_url(cik: int | str, accession_number: str, primary_document: str) -> str:
    cfg = load_config("sec_config.yaml")
    cik_int = int(cik)
    accession = normalise_accession(accession_number)
    return f"{cfg['archive_base_url']}/{cik_int}/{accession}/{primary_document}"


def download_filing(cik: int | str, accession_number: str, primary_document: str) -> str:
    ensure_project_dirs()
    url = filing_url(cik, accession_number, primary_document)
    cache_path = RAW_DIR / "filings" / str(cik) / f"{normalise_accession(accession_number)}_{primary_document}"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8", errors="ignore")
    response = requests.get(url, headers={"User-Agent": sec_user_agent()}, timeout=25)
    response.raise_for_status()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def download_latest_filings(cik: int | str, forms: tuple[str, ...] = ("10-K", "10-Q"), limit: int = 2) -> list[dict]:
    client = SECClient()
    submissions = client.get_submissions(cik)
    metadata = latest_filing_metadata(submissions, forms=forms).head(limit)
    results = []
    for _, row in metadata.iterrows():
        text = download_filing(cik, row["accession_number"], row["primary_document"])
        results.append(row.to_dict() | {"text": text})
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Download latest SEC filings for a CIK.")
    parser.add_argument("--cik", type=str, default="")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not args.cik:
        print("Provide --cik to download live filings. Cached sample chunks are built by src.filing_parser.")
        return
    results = download_latest_filings(args.cik, limit=args.limit)
    print(f"downloaded_filings={len(results)}")


if __name__ == "__main__":
    main()
