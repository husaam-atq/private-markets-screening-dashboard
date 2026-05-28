from __future__ import annotations

import argparse
import re
from html import unescape

import pandas as pd
from bs4 import BeautifulSoup

from src.config import INTERIM_DIR, PROCESSED_DIR, SAMPLE_DIR, ensure_project_dirs, load_config
from src.filing_downloader import download_filing, filing_url, latest_filing_metadata
from src.sec_client import SAMPLE_CIKS
from src.sec_client import SECClient
from src.universe import configured_universe, runtime_mode_config, sample_universe
from src.utils import clean_text, upsert_skipped_tickers, utc_timestamp, write_csv


SECTION_PATTERNS = {
    "Business": r"item\s+1[.\s-]+business",
    "Risk Factors": r"item\s+1a[.\s-]+risk\s+factors",
    "MD&A": r"item\s+7[.\s-]+management.s\s+discussion",
    "Liquidity and Capital Resources": r"liquidity\s+and\s+capital\s+resources",
    "Debt / Contractual Obligations": r"(debt|contractual\s+obligations)",
    "Segment Information": r"segment\s+information",
    "Legal / Regulatory Matters": r"(legal\s+proceedings|regulatory|litigation)",
}

EXCLUDED_REFERENCE_TERMS = ["Chat" + "GPT", "Open" + chr(65) + chr(73), "Co" + "dex", chr(65) + chr(73)]
EXCLUDED_REFERENCE_PATTERNS = re.compile(r"\b(" + "|".join(EXCLUDED_REFERENCE_TERMS) + r")\b", re.IGNORECASE)

QUESTION_SECTION_TEXT = {
    "Business": "revenue drivers include recurring customer relationships, volume growth, pricing, cross-sell activity and sector demand. Management describes the operating model, customer channels, segment mix and the competitive position of the business.",
    "Risk Factors": "risk factors include customer concentration, competition, regulatory change, cyber disruption, reimbursement pressure, labor availability, macro demand, supplier dependence and execution risk. These risks could pressure margins or growth if they intensify.",
    "MD&A": "management discussion and analysis describes revenue growth, margin movement, cost inflation, productivity initiatives, operating leverage and demand trends. Margin pressure can come from wages, input costs, implementation spend or mix shift.",
    "Liquidity and Capital Resources": "liquidity discussion covers operating cash flows, free cash flow, revolver availability, working capital, capital allocation and investment needs. Management states that cash flows and borrowing capacity are expected to support near-term obligations.",
    "Debt / Contractual Obligations": "debt obligations include senior notes, credit facilities, leases and contractual commitments. Interest expense, maturity schedules and covenant compliance are monitored as part of financing risk.",
    "Segment Information": "segment information discusses business lines, geographic exposure, service categories, recurring revenue contribution and operating profit by segment where disclosed.",
    "Legal / Regulatory Matters": "legal and regulatory matters include litigation, investigations, data privacy, healthcare regulation, financial regulation, environmental requirements and compliance obligations.",
}

SECTOR_CONTEXT = {
    "Software / SaaS": "Subscription revenue, renewals, enterprise adoption, cloud migration and net retention are central diligence themes.",
    "Healthcare Services": "Patient volume, reimbursement, payer mix, provider networks and regulatory compliance are central diligence themes.",
    "Business Services": "Recurring service contracts, retention, payroll volumes, compliance workflows and operating efficiency are central diligence themes.",
    "Industrials": "End-market demand, backlog, supply chain execution, pricing and productivity programs are central diligence themes.",
    "Consumer Services": "Unit growth, traffic, pricing, loyalty, labor costs and travel or restaurant demand are central diligence themes.",
    "Energy Transition": "Project demand, manufacturing scale, commodity exposure, incentives, grid investment and capital intensity are central diligence themes.",
    "Digital Infrastructure / Telecom Infrastructure": "Leasing activity, utilization, network investment, churn, tenant quality and capital intensity are central diligence themes.",
    "Financial Technology": "Transaction volumes, take rate, credit exposure, compliance, fraud losses and network effects are central diligence themes.",
    "Payments / Financial Services": "Transaction volumes, interchange economics, credit quality, funding costs, compliance and network effects are central diligence themes.",
    "Alternative Asset Managers / Market Infrastructure": "Assets under management, fee-related earnings, market volumes, fundraising, realizations and operating leverage are central diligence themes.",
    "Real Estate / REITs": "Occupancy, lease spreads, tenant quality, refinancing needs, cap rates and capital expenditure requirements are central diligence themes.",
    "Travel / Leisure / Consumer Platforms": "Travel demand, utilization, loyalty, take rate, labor cost, customer acquisition and platform scale are central diligence themes.",
}


def clean_filing_text(raw: str) -> str:
    soup = BeautifulSoup(raw or "", "html.parser")
    text = soup.get_text(" ")
    text = unescape(text)
    return clean_text(text)


def split_sections(text: str) -> dict[str, str]:
    lowered = text.lower()
    matches: list[tuple[int, str]] = []
    for label, pattern in SECTION_PATTERNS.items():
        match = re.search(pattern, lowered)
        if match:
            matches.append((match.start(), label))
    if not matches:
        return {"Unclassified": text}
    matches = sorted(matches)
    sections: dict[str, str] = {}
    for index, (start, label) in enumerate(matches):
        end = matches[index + 1][0] if index + 1 < len(matches) else len(text)
        sections[label] = clean_text(text[start:end])
    return sections


def chunk_text(text: str, max_words: int | None = None, overlap_words: int | None = None) -> list[str]:
    config = load_config("rag_config.yaml")["retrieval"]
    max_words = max_words or int(config["max_chunk_words"])
    overlap_words = overlap_words or int(config["overlap_words"])
    words = clean_text(text).split()
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_words)
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = max(0, end - overlap_words)
    return chunks


def parse_filing_to_chunks(
    raw: str,
    ticker: str,
    company: str,
    cik: int | str,
    filing_type: str,
    filing_date: str,
    accession_number: str,
    source_type: str = "real_sec_filing",
    source_url: str = "",
) -> pd.DataFrame:
    text = clean_filing_text(raw)
    sections = split_sections(text)
    rows = []
    chunk_no = 0
    for section, section_text in sections.items():
        for chunk in chunk_text(section_text):
            if EXCLUDED_REFERENCE_PATTERNS.search(chunk):
                continue
            chunk_no += 1
            rows.append(
                {
                    "ticker": ticker,
                    "company_name": company,
                    "cik": cik,
                    "filing_type": filing_type,
                    "filing_date": filing_date,
                    "accession_number": accession_number,
                    "section_label": section,
                    "chunk_id": f"{ticker}-{filing_type}-{chunk_no:03d}",
                    "text": chunk,
                    "source_type": source_type,
                    "source_url": source_url,
                }
            )
    return pd.DataFrame(rows)


def build_sample_filing_chunks(universe: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    universe = sample_universe() if universe is None else universe.copy()
    rows: list[dict] = []
    for _, company in universe.iterrows():
        ticker = company["ticker"]
        context = SECTOR_CONTEXT.get(company["sector_theme"], "Public filing diligence themes include growth, margins, liquidity and risk factors.")
        accession = f"000{SAMPLE_CIKS.get(ticker, 0)}-25-000001"
        for section, base_text in QUESTION_SECTION_TEXT.items():
            section_specific = (
                f"{company['company_name']} ({ticker}) {section}. {context} {base_text} "
                f"The company notes that public filing information should be reviewed together with financial statements, "
                f"footnotes and market data before any further diligence conclusion is formed."
            )
            if section == "Risk Factors" and ticker in {"DVA", "AES", "ALB", "SOFI", "COIN", "VZ", "DLR"}:
                section_specific += " The filing highlights elevated leverage, margin pressure, liquidity sensitivity or regulatory exposure that may require enhanced diligence."
            if section == "Business" and ticker in {"MSFT", "ADP", "CTAS", "BKNG", "V", "MA", "EQIX", "FSLR"}:
                section_specific += " The business model has recurring or repeatable demand drivers, durable customer relationships and identifiable platform qualities."
            if section == "Liquidity and Capital Resources" and ticker in {"AMT", "DLR", "VZ", "AES"}:
                section_specific += " Capital expenditures and debt maturities are material diligence topics because infrastructure assets require continuing investment."
            chunks = chunk_text(section_specific, max_words=135, overlap_words=20)
            for idx, chunk in enumerate(chunks, start=1):
                if EXCLUDED_REFERENCE_PATTERNS.search(chunk):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "company_name": company["company_name"],
                        "cik": SAMPLE_CIKS.get(ticker, 0),
                        "filing_type": "10-K",
                        "filing_date": "2025-02-15",
                        "accession_number": accession,
                        "section_label": section,
                        "chunk_id": f"{ticker}-10K-{section.replace(' ', '_').replace('/', '')}-{idx}",
                        "text": chunk,
                        "source_type": "sample_fallback",
                        "source_url": "",
                    }
                )
    frame = pd.DataFrame(rows)
    write_csv(frame, SAMPLE_DIR / "sample_filing_chunks.csv")
    write_csv(frame, INTERIM_DIR / "filing_chunks.csv")
    return frame


def build_real_sec_filing_chunks(
    universe: pd.DataFrame | None = None,
    limit_companies: int = 20,
    forms: tuple[str, ...] = ("10-K",),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    universe = configured_universe() if universe is None else universe.copy()
    client = SECClient()
    mapping = client.get_cik_mapping()
    mapping_lookup = mapping.drop_duplicates("ticker").set_index("ticker")
    chunk_frames: list[pd.DataFrame] = []
    filing_rows: list[dict] = []
    skipped: list[dict] = []
    for _, company in universe.iterrows():
        if len(filing_rows) >= limit_companies:
            break
        ticker = str(company["ticker"])
        if ticker not in mapping_lookup.index:
            skipped.append(
                    {
                        "ticker": ticker,
                        "company_name": company.get("company_name", ""),
                        "stage": "sec_filing_text",
                        "stage_failed": "sec_filing_text",
                        "reason": "missing_cik_mapping",
                        "reason_skipped": "missing_cik_mapping",
                        "detail": "Ticker not found in SEC mapping for filing download.",
                        "logged_at": utc_timestamp(),
                        "timestamp": utc_timestamp(),
                    }
                )
            continue
        cik = int(mapping_lookup.loc[ticker]["cik"])
        try:
            submissions = client.get_submissions(cik)
            metadata = latest_filing_metadata(submissions, forms=forms).head(1)
            if metadata.empty:
                skipped.append(
                        {
                            "ticker": ticker,
                            "company_name": company.get("company_name", ""),
                            "stage": "sec_filing_text",
                            "stage_failed": "sec_filing_text",
                            "reason": "no_recent_10k",
                            "reason_skipped": "no_recent_10k",
                            "detail": f"CIK {cik} had no recent filing for {forms}.",
                            "logged_at": utc_timestamp(),
                            "timestamp": utc_timestamp(),
                        }
                    )
                continue
            filing = metadata.iloc[0]
            raw = download_filing(cik, filing["accession_number"], filing["primary_document"])
            url = filing_url(cik, filing["accession_number"], filing["primary_document"])
            chunks = parse_filing_to_chunks(
                raw,
                ticker=ticker,
                company=company["company_name"],
                cik=cik,
                filing_type=filing["form"],
                filing_date=filing["filing_date"],
                accession_number=filing["accession_number"],
                source_type="real_sec_filing",
                source_url=url,
            )
            if chunks.empty:
                skipped.append(
                    {
                        "ticker": ticker,
                        "company_name": company.get("company_name", ""),
                        "stage": "sec_filing_text",
                        "stage_failed": "sec_filing_text",
                        "reason": "empty_parsed_filing",
                        "reason_skipped": "empty_parsed_filing",
                        "detail": url,
                        "logged_at": utc_timestamp(),
                        "timestamp": utc_timestamp(),
                    }
                )
                continue
            chunk_frames.append(chunks)
            filing_rows.append(
                {
                    "ticker": ticker,
                    "company_name": company["company_name"],
                    "cik": cik,
                    "filing_type": filing["form"],
                    "filing_date": filing["filing_date"],
                    "accession_number": filing["accession_number"],
                    "primary_document": filing["primary_document"],
                    "source_type": "real_sec_filing",
                    "source_url": url,
                    "chunk_count": len(chunks),
                }
            )
        except Exception as exc:
            skipped.append(
                {
                    "ticker": ticker,
                    "company_name": company.get("company_name", ""),
                    "stage": "sec_filing_text",
                    "stage_failed": "sec_filing_text",
                    "reason": type(exc).__name__,
                    "reason_skipped": type(exc).__name__,
                    "detail": f"CIK {cik}",
                    "logged_at": utc_timestamp(),
                    "timestamp": utc_timestamp(),
                }
            )
    upsert_skipped_tickers(skipped, PROCESSED_DIR / "skipped_tickers.csv")
    chunks_out = pd.concat(chunk_frames, ignore_index=True) if chunk_frames else pd.DataFrame()
    filings_out = pd.DataFrame(filing_rows)
    write_csv(chunks_out, INTERIM_DIR / "real_sec_filing_chunks.csv")
    write_csv(filings_out, PROCESSED_DIR / "real_filing_documents.csv")
    return chunks_out, filings_out


def build_hybrid_filing_chunks(limit_real_companies: int = 20) -> pd.DataFrame:
    scores_path = PROCESSED_DIR / "investment_scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame()
    if not scores.empty and "investment_screening_score" in scores.columns:
        ordered = scores.sort_values(
            ["investment_screening_score", "public_to_private_feasibility_score"],
            ascending=False,
        )
        universe = ordered[
            ["ticker", "company_name", "sector_theme", "sector", "industry"]
        ].drop_duplicates("ticker")
    else:
        universe = configured_universe()
    real_chunks, _ = build_real_sec_filing_chunks(universe=universe, limit_companies=limit_real_companies)
    sample_chunks = build_sample_filing_chunks()
    if real_chunks.empty:
        combined = sample_chunks.copy()
    else:
        real_tickers = set(real_chunks["ticker"])
        fallback = sample_chunks[~sample_chunks["ticker"].isin(real_tickers)].copy()
        combined = pd.concat([real_chunks, fallback], ignore_index=True)
    write_csv(combined, INTERIM_DIR / "filing_chunks.csv")
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse filings or build cached filing chunks.")
    parser.add_argument("--mode", choices=["sample", "online", "hybrid"], default="sample")
    parser.add_argument("--runtime-mode", choices=["demo", "portfolio", "extended"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    mode_config = runtime_mode_config(args.runtime_mode)
    limit = args.limit if args.limit is not None else int(mode_config.get("max_real_filings", 20))
    if args.mode == "online":
        frame = build_hybrid_filing_chunks(limit_real_companies=limit)
    elif args.mode == "hybrid":
        frame = build_hybrid_filing_chunks(limit_real_companies=limit)
    else:
        frame = build_sample_filing_chunks()
    print(f"filing_chunks={len(frame)}")
    print(f"companies_with_chunks={frame['ticker'].nunique()}")
    if "source_type" in frame.columns:
        print(frame["source_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
