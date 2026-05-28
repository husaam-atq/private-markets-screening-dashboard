from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.charts import generate_all_charts
from src.config import EXCEL_DIR, INTERIM_DIR, PROCESSED_DIR, REPORT_DIR, SAMPLE_DIR, ensure_project_dirs
from src.memo_generator import generate_memo
from src.retrieval_eval import run_retrieval_evaluation
from src.screening_pipeline import run_pipeline
from src.utils import format_pct, read_csv_if_exists


def _metric(eval_results: pd.DataFrame, name: str, default: float = 0.0) -> float:
    row = eval_results[eval_results["metric"] == name]
    return float(row["value"].iloc[0]) if not row.empty else default


def generate_filing_evidence_report() -> str:
    evidence = read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv")
    questions = read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv")
    lines = ["# Filing Evidence Report", ""]
    if questions.empty:
        lines.append("No filing questions were generated.")
    for _, question in questions.iterrows():
        lines.append(f"## {question['company_name']} ({question['ticker']}) - {question['question']}")
        lines.append(f"Summary: {question['summary_answer']}")
        subset = evidence[
            (evidence["ticker"] == question["ticker"]) & (evidence["question_id"] == question["question_id"])
        ].sort_values("rank")
        if subset.empty:
            lines.append("- insufficient filing evidence found")
        else:
            for _, row in subset.head(3).iterrows():
                lines.append(
                    f"- {str(row['text'])[:360]} Source: {row['company_name']} | {row['filing_type']} | "
                    f"{row['filing_date']} | {row['section_label']} | {row['chunk_id']}."
                )
        lines.append("")
    report = "\n".join(lines)
    (REPORT_DIR / "filing_evidence_report.md").write_text(report, encoding="utf-8")
    return report


def generate_rag_evaluation_report() -> str:
    eval_results = read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv")
    details = read_csv_if_exists(PROCESSED_DIR / "rag_eval_details.csv")
    if eval_results.empty:
        eval_results, details = run_retrieval_evaluation()
    lines = [
        "# RAG Evaluation Report",
        "",
        "This report evaluates whether the filing retrieval layer returns source-backed evidence for diligence-style questions.",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for _, row in eval_results.iterrows():
        value = row["value"]
        formatted = f"{float(value):.1f}" if row["metric"].endswith("questions") or row["metric"] == "questions_evaluated" else format_pct(float(value))
        lines.append(f"| {row['metric']} | {formatted} |")
    weak = details[(details["expected_no_answer"] == False) & (details["hit_at_5"] == 0)].head(5) if not details.empty else pd.DataFrame()
    lines.extend(["", "## Weak Retrieval Cases", ""])
    if weak.empty:
        lines.append("No answerable gold questions missed at top 5 in the cached sample evaluation.")
    else:
        for _, row in weak.iterrows():
            lines.append(f"- {row['ticker']} {row['question_id']} missed at top 5; top score {row['top_score']:.3f}.")
    report = "\n".join(lines)
    (REPORT_DIR / "rag_evaluation_report.md").write_text(report, encoding="utf-8")
    return report


def generate_excel_workbook() -> Path:
    workbook = EXCEL_DIR / "private_markets_screening_outputs.xlsx"
    sheets = {
        "Executive Summary": pd.DataFrame([summary_metrics()]),
        "Universe": read_csv_if_exists(SAMPLE_DIR / "sample_universe.csv"),
        "Market Data": read_csv_if_exists(INTERIM_DIR / "market_data_snapshot.csv"),
        "SEC Fundamentals": read_csv_if_exists(INTERIM_DIR / "normalized_fundamentals.csv"),
        "Screening Scores": read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv"),
        "Company Categories": read_csv_if_exists(PROCESSED_DIR / "company_categories.csv"),
        "Peer Benchmarks": read_csv_if_exists(PROCESSED_DIR / "peer_benchmarks.csv"),
        "Target Deep Dive": read_csv_if_exists(PROCESSED_DIR / "target_deep_dive.csv"),
        "Filing Evidence": read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv"),
        "Diligence Questions": read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv"),
        "RAG Evaluation": read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv"),
        "Methodology": pd.DataFrame(
            [
                {
                    "item": "Methodology",
                    "description": "Transparent percentile scorecards using public market data, SEC-style cached sample fundamentals and filing evidence retrieval.",
                }
            ]
        ),
    }
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])
            worksheet = writer.sheets[sheet_name[:31]]
            for col in worksheet.columns:
                worksheet.column_dimensions[col[0].column_letter].width = min(max(len(str(col[0].value or "")) + 4, 14), 40)
    return workbook


def summary_metrics() -> dict:
    scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
    screening = read_csv_if_exists(PROCESSED_DIR / "screening_universe.csv")
    fundamentals = read_csv_if_exists(INTERIM_DIR / "normalized_fundamentals.csv")
    chunks = read_csv_if_exists(INTERIM_DIR / "filing_chunks.csv")
    eval_results = read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv")
    categories = read_csv_if_exists(PROCESSED_DIR / "company_categories.csv")
    return {
        "companies_screened": len(screening),
        "sectors_covered": screening["sector_theme"].nunique() if not screening.empty else 0,
        "metrics_calculated": 22,
        "companies_with_sec_fundamentals": fundamentals["ticker"].nunique() if not fundamentals.empty else 0,
        "companies_with_filing_chunks": chunks["ticker"].nunique() if not chunks.empty else 0,
        "average_data_quality_score": scores["data_quality_score"].mean() if not scores.empty else 0,
        "high_priority_diligence_candidates": int((categories["primary_category"] == "High Priority for Further Diligence").sum()) if not categories.empty else 0,
        "rag_hit_at_5": _metric(eval_results, "hit_rate_at_5"),
        "rag_precision_at_5": _metric(eval_results, "precision_at_5"),
        "citation_coverage": _metric(eval_results, "citation_coverage"),
        "unsupported_claim_rate": _metric(eval_results, "unsupported_claim_rate"),
    }


def run_reporting() -> dict[str, Path | str]:
    ensure_project_dirs()
    if read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv").empty:
        run_pipeline(mode="sample")
    if read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv").empty:
        run_retrieval_evaluation()
    generate_memo()
    filing_report = generate_filing_evidence_report()
    rag_report = generate_rag_evaluation_report()
    generate_all_charts()
    workbook = generate_excel_workbook()
    return {
        "workbook": workbook,
        "filing_report": str(REPORT_DIR / "filing_evidence_report.md"),
        "rag_report": str(REPORT_DIR / "rag_evaluation_report.md"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reports, charts and Excel workbook.")
    parser.parse_args()
    outputs = run_reporting()
    print(f"excel_workbook={outputs['workbook']}")
    print("reports_generated=true")


if __name__ == "__main__":
    main()
