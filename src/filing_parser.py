from __future__ import annotations

import argparse
import re
from html import unescape

import pandas as pd
from bs4 import BeautifulSoup

from src.config import INTERIM_DIR, SAMPLE_DIR, ensure_project_dirs, load_config
from src.sec_client import SAMPLE_CIKS
from src.universe import sample_universe
from src.utils import clean_text, write_csv


SECTION_PATTERNS = {
    "Business": r"item\s+1[.\s-]+business",
    "Risk Factors": r"item\s+1a[.\s-]+risk\s+factors",
    "MD&A": r"item\s+7[.\s-]+management.s\s+discussion",
    "Liquidity and Capital Resources": r"liquidity\s+and\s+capital\s+resources",
    "Debt / Contractual Obligations": r"(debt|contractual\s+obligations)",
    "Segment Information": r"segment\s+information",
    "Legal / Regulatory Matters": r"(legal\s+proceedings|regulatory|litigation)",
}

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
) -> pd.DataFrame:
    text = clean_filing_text(raw)
    sections = split_sections(text)
    rows = []
    chunk_no = 0
    for section, section_text in sections.items():
        for chunk in chunk_text(section_text):
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
                    }
                )
    frame = pd.DataFrame(rows)
    write_csv(frame, SAMPLE_DIR / "sample_filing_chunks.csv")
    write_csv(frame, INTERIM_DIR / "filing_chunks.csv")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse filings or build cached filing chunks.")
    parser.parse_args()
    frame = build_sample_filing_chunks()
    print(f"filing_chunks={len(frame)}")
    print(f"companies_with_chunks={frame['ticker'].nunique()}")


if __name__ == "__main__":
    main()
