from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.charts import generate_all_charts
from src.config import EXCEL_DIR, INTERIM_DIR, PROCESSED_DIR, REPORT_DIR, SAMPLE_DIR, ensure_project_dirs
from src.feature_engineering import METRIC_COLUMNS
from src.grounded_generation_eval import run_grounded_generation_evaluation
from src.memo_generator import generate_memo
from src.rag_generator import run_generation
from src.retrieval_eval import run_retrieval_evaluation
from src.screening_pipeline import run_pipeline
from src.utils import format_pct, metric_value, read_csv_if_exists


def generate_filing_evidence_report() -> str:
    evidence = read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv")
    questions = read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv")
    generated = read_csv_if_exists(PROCESSED_DIR / "generated_rag_answers.csv")
    lines = [
        "# Filing Evidence Report",
        "",
        "Evidence rows are labelled by `source_type`. Real SEC filing chunks are preferred; sample fallback chunks are used only where real filing text was not indexed.",
        "Generated answers are deterministic by default. Optional local Ollama answers are shown only when available and remain constrained to retrieved evidence.",
        "",
    ]
    if questions.empty:
        lines.append("No filing questions were generated.")
    for _, question in questions.iterrows():
        lines.append(f"## {question['company_name']} ({question['ticker']}) - {question['question']}")
        lines.append(f"Summary: {question['summary_answer']}")
        generated_row = generated[
            (generated.get("ticker", pd.Series(dtype=str)) == question["ticker"])
            & (generated.get("question_id", pd.Series(dtype=str)) == question["question_id"])
        ]
        if not generated_row.empty:
            row = generated_row.iloc[0]
            lines.append(
                f"Generated answer ({row.get('generation_mode', 'deterministic')}): {row.get('answer', '')} "
                f"Cited chunks: {row.get('cited_chunks', '')}. Unsupported warning: {row.get('unsupported_claim_warning', False)}."
            )
        subset = evidence[
            (evidence["ticker"] == question["ticker"]) & (evidence["question_id"] == question["question_id"])
        ].sort_values("rank")
        if subset.empty:
            lines.append("- insufficient filing evidence found")
        else:
            for _, row in subset.head(3).iterrows():
                lines.append(
                    f"- {str(row.get('evidence_snippet', row['text']))[:420]} Source: {row['company_name']} | {row['filing_type']} | "
                    f"{row['filing_date']} | {row['section_label']} | {row['chunk_id']} | "
                    f"{row.get('source_type', 'unknown')} | section boost {float(row.get('section_boost', 0)):.3f}."
                )
        lines.append("")
    report = "\n".join(lines)
    (REPORT_DIR / "filing_evidence_report.md").write_text(report, encoding="utf-8")
    return report


def generate_rag_evaluation_report() -> str:
    eval_results = read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv")
    details = read_csv_if_exists(PROCESSED_DIR / "rag_eval_details.csv")
    examples = read_csv_if_exists(PROCESSED_DIR / "rag_eval_examples.csv")
    grounded = read_csv_if_exists(PROCESSED_DIR / "grounded_generation_eval.csv")
    if eval_results.empty:
        eval_results, details = run_retrieval_evaluation()
    lines = [
        "# RAG Evaluation Report",
        "",
        "This report evaluates whether the filing retrieval layer returns source-backed evidence for diligence-style questions. The gold set includes section-confuser and no-answer questions, so perfect-looking metrics should not be assumed in live refreshes.",
        "Retrieval metrics are separate from optional local language-model generation metrics.",
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
    lines.extend(["", "## Strong and Weak Retrieval Examples", ""])
    if not examples.empty:
        for _, row in examples.head(12).iterrows():
            lines.append(
                f"- {row['example_type']}: {row['ticker']} {row['question_id']} "
                f"Precision@5 {row.get('precision_at_5', 0):.2f}, top score {row.get('top_score', 0):.3f}, "
                f"top source {row.get('top_source_type', 'unknown')}."
            )
    weak = details[(details["expected_no_answer"] == False) & (details["hit_at_5"] == 0)].head(5) if not details.empty else pd.DataFrame()
    lines.extend(["", "## Missed Answerable Questions", ""])
    if weak.empty:
        lines.append("No answerable gold questions missed at top 5 in the current evaluation, but Precision@5 still shows that some retrieved chunks are adjacent rather than directly responsive.")
    else:
        for _, row in weak.iterrows():
            lines.append(f"- {row['ticker']} {row['question_id']} missed at top 5; top score {row['top_score']:.3f}.")
    lines.extend(["", "## Grounded Generation Context", ""])
    if grounded.empty:
        lines.append("Grounded generation evaluation has not been run yet.")
    else:
        for _, row in grounded.iterrows():
            lines.append(f"- {row['metric']}: {row['value']:.3f}")
    lines.extend(
        [
            "",
            "## Limitations and Roadmap",
            "",
            "- Hit@5 measures whether at least one relevant passage appears; Precision@5 is more important for user-facing evidence quality.",
            "- Section-confuser questions can retrieve adjacent sections where the filing discusses the same economic issue in different wording.",
            "- Real filing coverage is intentionally capped for commit-safe outputs and can be expanded locally.",
            "- The retrieval benchmark is not a substitute for legal, financial or investment diligence.",
        ]
    )
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
        "RAG Eval Details": read_csv_if_exists(PROCESSED_DIR / "rag_eval_details.csv"),
        "RAG Eval Examples": read_csv_if_exists(PROCESSED_DIR / "rag_eval_examples.csv"),
        "Generated RAG Answers": read_csv_if_exists(PROCESSED_DIR / "generated_rag_answers.csv"),
        "Grounded Gen Eval": read_csv_if_exists(PROCESSED_DIR / "grounded_generation_eval.csv"),
        "Grounded Gen Details": read_csv_if_exists(PROCESSED_DIR / "grounded_generation_eval_details.csv"),
        "Skipped Tickers": read_csv_if_exists(PROCESSED_DIR / "skipped_tickers.csv"),
        "Real Filing Docs": read_csv_if_exists(PROCESSED_DIR / "real_filing_documents.csv"),
        "Methodology": pd.DataFrame(
            [
                {
                    "item": "Methodology",
                    "description": "Transparent percentile scorecards using public market data, SEC companyfacts where available, source-labelled filing evidence retrieval and explicit fallback warnings.",
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
    market = read_csv_if_exists(INTERIM_DIR / "market_data_snapshot.csv")
    skipped = read_csv_if_exists(PROCESSED_DIR / "skipped_tickers.csv")
    real_filings = read_csv_if_exists(PROCESSED_DIR / "real_filing_documents.csv")
    grounded = read_csv_if_exists(PROCESSED_DIR / "grounded_generation_eval.csv")
    return {
        "companies_screened": len(screening),
        "sectors_covered": screening["sector_theme"].nunique() if not screening.empty else 0,
        "metrics_calculated": len(METRIC_COLUMNS),
        "companies_with_yfinance_data": market["ticker"].nunique() if not market.empty else 0,
        "companies_with_sec_fundamentals": fundamentals["ticker"].nunique() if not fundamentals.empty else 0,
        "companies_with_usable_sec_fundamentals": screening["revenue"].notna().sum() if not screening.empty and "revenue" in screening else 0,
        "companies_with_filing_chunks": chunks["ticker"].nunique() if not chunks.empty else 0,
        "real_filing_documents_parsed": real_filings["ticker"].nunique() if not real_filings.empty else 0,
        "real_filing_chunks": int((chunks["source_type"] == "real_sec_filing").sum()) if not chunks.empty and "source_type" in chunks else 0,
        "sample_fallback_chunks": int((chunks["source_type"] == "sample_fallback").sum()) if not chunks.empty and "source_type" in chunks else 0,
        "skipped_ticker_count": len(skipped),
        "average_data_quality_score": scores["data_quality_score"].mean() if not scores.empty else 0,
        "public_to_private_candidates": int((categories["primary_category"] == "Public-to-Private Candidate").sum()) if not categories.empty else 0,
        "pe_platform_candidates": int((categories["primary_category"] == "PE Platform Candidate").sum()) if not categories.empty else 0,
        "rag_hit_at_5": metric_value(eval_results, "hit_rate_at_5"),
        "rag_precision_at_5": metric_value(eval_results, "precision_at_5"),
        "citation_coverage": metric_value(eval_results, "citation_coverage"),
        "unsupported_claim_rate": metric_value(eval_results, "unsupported_claim_rate"),
        "grounded_generation_score": metric_value(grounded, "groundedness_score"),
        "grounded_generation_citation_coverage": metric_value(grounded, "citation_coverage"),
    }


def run_reporting() -> dict[str, Path | str]:
    ensure_project_dirs()
    if read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv").empty:
        run_pipeline(mode="online")
    if read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv").empty:
        run_retrieval_evaluation()
    if read_csv_if_exists(PROCESSED_DIR / "generated_rag_answers.csv").empty:
        run_generation()
    run_grounded_generation_evaluation()
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
