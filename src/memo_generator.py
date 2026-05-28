from __future__ import annotations

import argparse

import pandas as pd

from src.config import PROCESSED_DIR, REPORT_DIR, ensure_project_dirs
from src.filing_rag import run_retrieval_for_companies
from src.screening_pipeline import run_pipeline
from src.utils import format_money, format_multiple, format_pct, read_csv_if_exists, write_csv


def _evidence_reference(row: pd.Series) -> str:
    return (
        f"{row['company_name']} | {row['ticker']} | {row['filing_type']} | "
        f"{row['filing_date']} | {row['section_label']} | {row['chunk_id']} | "
        f"{row.get('source_type', 'unknown')}"
    )


def _evidence_lines(evidence: pd.DataFrame, question_id: str, limit: int = 2) -> list[str]:
    subset = evidence[evidence["question_id"] == question_id].sort_values("rank").head(limit)
    if subset.empty:
        return ["- insufficient filing evidence found"]
    lines = []
    for _, row in subset.iterrows():
        snippet = str(row["text"])[:420].strip()
        lines.append(f"- {snippet} Source: {_evidence_reference(row)}.")
    return lines


def generate_memo(target_ticker: str | None = None) -> tuple[str, pd.DataFrame]:
    ensure_project_dirs()
    scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
    categories = read_csv_if_exists(PROCESSED_DIR / "company_categories.csv")
    benchmarks = read_csv_if_exists(PROCESSED_DIR / "peer_benchmarks.csv")
    questions = read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv")
    evidence = read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv")
    if scores.empty or categories.empty or benchmarks.empty:
        run_pipeline(mode="online")
        scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
        categories = read_csv_if_exists(PROCESSED_DIR / "company_categories.csv")
        benchmarks = read_csv_if_exists(PROCESSED_DIR / "peer_benchmarks.csv")
    if evidence.empty or questions.empty:
        selected = [target_ticker] if target_ticker else scores.sort_values("investment_screening_score", ascending=False).head(8)["ticker"].tolist()
        run_retrieval_for_companies(tickers=selected, top_k=5)
        questions = read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv")
        evidence = read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv")
    target_ticker = target_ticker or scores.sort_values("investment_screening_score", ascending=False).iloc[0]["ticker"]
    company = scores[scores["ticker"] == target_ticker].iloc[0]
    category = categories[categories["ticker"] == target_ticker].iloc[0]
    peer = benchmarks[benchmarks["ticker"] == target_ticker].iloc[0]
    company_questions = questions[questions["ticker"] == target_ticker]
    company_evidence = evidence[evidence["ticker"] == target_ticker] if not evidence.empty else pd.DataFrame()

    lines = [
        f"# Investment Screening Memo: {company['company_name']} ({company['ticker']})",
        "",
        (
            f"Data mode: structured data source is {company.get('source_type', 'unknown')}; "
            "filing evidence rows are labelled by source type."
        ),
        "",
        "## 1. Executive Summary",
        (
            f"{company['company_name']} screens as **{category['primary_category']}** with an "
            f"Investment Screening Score of {company['investment_screening_score']:.1f}, "
            f"Public-to-Private Feasibility Score of {company.get('public_to_private_feasibility_score', 0):.1f}, "
            f"Platform Candidate Score of {company['platform_candidate_score']:.1f} and "
            f"Red Flag Score of {company['red_flag_score']:.1f}. This memo frames a public-data "
            "screening view for further diligence and does not provide investment advice."
        ),
        "",
        "## 2. Why This Company Screened",
        f"{company['investment_screening_explanation']}",
        f"Category rationale: {category['why_this_category']}",
        "",
        "## 3. Sector and Peer Context",
        (
            f"The company is mapped to {company['sector_theme']}. Relative to the selected peer set, "
            f"latest revenue growth is {format_pct(company['revenue_growth_yoy'])} versus a peer "
            f"median of {format_pct(peer['revenue_growth_yoy_peer_median'])}; EBIT margin proxy is "
            f"{format_pct(company['operating_margin'])} versus peer median "
            f"{format_pct(peer['operating_margin_peer_median'])}."
        ),
        "",
        "## 3A. Private-Markets Feasibility Classification",
        (
            f"Market-cap band: {company.get('market_cap_band', 'n/a')}. Public Quality Score is "
            f"{company.get('public_quality_score', 0):.1f}, while Public-to-Private Feasibility Score is "
            f"{company.get('public_to_private_feasibility_score', 0):.1f}. Mega-cap names can be useful "
            "benchmark quality comps, but the feasibility score penalises enterprise values that are unlikely "
            "to fit a typical PE platform or public-to-private mandate."
        ),
        "",
        "## 4. Financial Profile",
        (
            f"Latest revenue is {format_money(company['revenue'])}, gross margin is "
            f"{format_pct(company['gross_margin'])}, EBIT margin proxy is "
            f"{format_pct(company['operating_margin'])}, and free cash flow conversion proxy is "
            f"{format_pct(company['fcf_conversion'])}."
        ),
        "",
        "## 5. Valuation Snapshot",
        (
            f"Market capitalization is {format_money(company['market_cap'])}; enterprise value is "
            f"{format_money(company['enterprise_value'])}; EV / sales is {format_multiple(company['ev_to_sales'])}; "
            f"EV / EBIT proxy is {format_multiple(company['ev_to_ebit_proxy'])}."
        ),
        "",
        "## 6. Leverage and Cash Conversion",
        (
            f"Debt / EBIT proxy is {format_multiple(company['debt_to_ebit_proxy'])}, net debt / EBIT proxy is "
            f"{format_multiple(company['net_debt_to_ebit_proxy'])}, cash / revenue is "
            f"{format_pct(company['cash_to_revenue'])}, and capex intensity is "
            f"{format_pct(company['capex_intensity'])}."
        ),
        "",
        "## 7. Investment Screening Scorecard",
        "| Scorecard | Score |",
        "|---|---:|",
        f"| Public Quality Score | {company.get('public_quality_score', 0):.1f} |",
        f"| Investment Screening Score | {company['investment_screening_score']:.1f} |",
        f"| Diligence Priority Score | {company['diligence_priority_score']:.1f} |",
        f"| Platform Candidate Score | {company['platform_candidate_score']:.1f} |",
        f"| Public-to-Private Feasibility Score | {company.get('public_to_private_feasibility_score', 0):.1f} |",
        f"| Value Creation Potential Score | {company['value_creation_potential_score']:.1f} |",
        f"| Credit Risk Score | {company['credit_risk_score']:.1f} |",
        f"| Red Flag Score | {company['red_flag_score']:.1f} |",
        f"| Data Quality Score | {company['data_quality_score']:.1f} |",
        "",
        "## 8. Key Filing Evidence",
    ]
    for question_id in ["revenue_drivers", "liquidity_risks", "margin_pressure", "debt_obligations", "competition"]:
        question_row = company_questions[company_questions["question_id"] == question_id]
        question_text = question_row["question"].iloc[0] if not question_row.empty else question_id
        lines.append(f"**{question_text}**")
        lines.extend(_evidence_lines(company_evidence, question_id))
        lines.append("")
    lines.extend(
        [
            "## 9. Diligence Questions",
            "- What portion of growth is volume, pricing, retention, mix or acquisition-driven?",
            "- How resilient is cash conversion after normalising working capital and capex?",
            "- Which peer margin gaps are structural versus addressable through operating initiatives?",
            "- Are debt maturities, covenants or capital requirements limiting strategic flexibility?",
            "- Which filing risks require specialist legal, regulatory or commercial diligence?",
            "",
            "## 10. Red Flags and Risks",
            (
                f"Structured red flags include a Red Flag Score of {company['red_flag_score']:.1f}, "
                f"Credit Risk Score of {company['credit_risk_score']:.1f}, drawdown from 52-week high of "
                f"{format_pct(company['drawdown_from_52w_high'])}, and realised volatility of "
                f"{format_pct(company['realised_volatility'])}."
            ),
            "",
            "## 11. Value Creation Angles",
            (
                f"Value Creation Potential Score is {company['value_creation_potential_score']:.1f}. "
                "Potential angles should be diligence-tested through peer margin benchmarking, cash conversion "
                "normalisation, capex planning and operating KPI review."
            ),
            "",
            "## 12. Recommendation for Further Diligence",
            f"Proceed as a {category['primary_category']} screen, with emphasis on validating the public-data signals.",
            "",
            "## 13. Data and Methodology Limitations",
            (
                "This memo uses public market data, SEC filing fundamentals and source-backed filing retrieval. "
                "Rows labelled sample_fallback are included only where real filing chunks were not available. "
                "It is a screening workflow demonstration, not investment advice or a substitute "
                "for confidential diligence, legal review, management meetings or full quality-of-earnings analysis."
            ),
        ]
    )
    memo = "\n".join(lines)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "investment_screening_memo.md").write_text(memo, encoding="utf-8")
    memo_rows = pd.DataFrame(
        [
            {
                "ticker": target_ticker,
                "company_name": company["company_name"],
                "section": "Investment Screening Memo",
                "primary_category": category["primary_category"],
                "public_quality_score": company.get("public_quality_score"),
                "investment_screening_score": company["investment_screening_score"],
                "public_to_private_feasibility_score": company.get("public_to_private_feasibility_score"),
                "citation_coverage": 1.0 if not company_evidence.empty else 0.0,
            }
        ]
    )
    write_csv(memo_rows, PROCESSED_DIR / "memo_outputs.csv")
    return memo, memo_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic investment screening memo.")
    parser.add_argument("--ticker", type=str, default=None)
    args = parser.parse_args()
    memo, _ = generate_memo(target_ticker=args.ticker)
    print(f"memo_characters={len(memo)}")


if __name__ == "__main__":
    main()
