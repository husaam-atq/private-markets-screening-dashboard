from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CHART_DIR, PROCESSED_DIR, REPORT_DIR, ensure_project_dirs
from src.rag_generator import run_generation
from src.utils import format_pct, read_csv_if_exists, write_csv


def evaluate_generated_answers(answers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if answers.empty:
        return pd.DataFrame(), pd.DataFrame()
    details = answers.copy()
    details["citation_pass"] = pd.to_numeric(details.get("citation_coverage", 0), errors="coerce").fillna(0) >= 1
    details["overlap_pass"] = pd.to_numeric(details.get("evidence_overlap_score", 0), errors="coerce").fillna(0) >= 0.15
    details["banned_language_pass"] = pd.to_numeric(details.get("banned_language_count", 0), errors="coerce").fillna(0) == 0
    details["unsupported_claim_pass"] = ~details.get("unsupported_claim_warning", pd.Series(False, index=details.index)).astype(bool)
    details["no_answer_accuracy_flag"] = np.where(
        details.get("no_answer_flag", pd.Series(False, index=details.index)).astype(bool),
        details["answer"].astype(str).str.contains("insufficient evidence found", case=False, na=False),
        True,
    )
    details["numeric_consistency_pass"] = ~details["answer"].astype(str).str.contains(r"\b\d+(?:\.\d+)?\s?%", regex=True, na=False)
    details["answer_relevance_proxy"] = pd.to_numeric(details.get("evidence_overlap_score", 0), errors="coerce").fillna(0).clip(0, 1)
    details["groundedness_score"] = (
        details["citation_pass"].astype(float) * 0.28
        + details["overlap_pass"].astype(float) * 0.24
        + details["banned_language_pass"].astype(float) * 0.18
        + details["unsupported_claim_pass"].astype(float) * 0.18
        + details["numeric_consistency_pass"].astype(float) * 0.12
    )
    summary = pd.DataFrame(
        [
            {"metric": "answers_evaluated", "value": len(details), "metric_type": "coverage"},
            {"metric": "citation_coverage", "value": details["citation_pass"].mean(), "metric_type": "groundedness"},
            {"metric": "unsupported_claim_rate", "value": 1 - details["unsupported_claim_pass"].mean(), "metric_type": "groundedness"},
            {"metric": "evidence_overlap_score", "value": details["answer_relevance_proxy"].mean(), "metric_type": "groundedness"},
            {"metric": "banned_language_count", "value": details["banned_language_count"].sum(), "metric_type": "guardrail"},
            {"metric": "no_answer_accuracy", "value": details["no_answer_accuracy_flag"].mean(), "metric_type": "groundedness"},
            {"metric": "numeric_consistency_pass_rate", "value": details["numeric_consistency_pass"].mean(), "metric_type": "groundedness"},
            {"metric": "groundedness_score", "value": details["groundedness_score"].mean(), "metric_type": "groundedness"},
        ]
    )
    return summary, details


def generate_groundedness_chart(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    wanted = [
        "citation_coverage",
        "unsupported_claim_rate",
        "evidence_overlap_score",
        "no_answer_accuracy",
        "numeric_consistency_pass_rate",
        "groundedness_score",
    ]
    subset = summary[summary["metric"].isin(wanted)].copy()
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = ["#2f6f8f" if metric != "unsupported_claim_rate" else "#9a4d41" for metric in subset["metric"]]
    ax.bar(subset["metric"].str.replace("_", "\n"), subset["value"].astype(float), color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate / Score")
    ax.set_title("Grounded Generation Evaluation")
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHART_DIR / "grounded_generation_eval_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_groundedness_report(summary: pd.DataFrame, details: pd.DataFrame) -> str:
    lines = [
        "# Grounded Generation Evaluation Report",
        "",
        "This report evaluates generated filing answers for citation coverage, evidence overlap, banned recommendation language and unsupported-claim warnings. The default project path is deterministic; local Ollama generation is optional.",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for _, row in summary.iterrows():
        value = float(row["value"])
        formatted = f"{value:.0f}" if row["metric"] in {"answers_evaluated", "banned_language_count"} else format_pct(value)
        lines.append(f"| {row['metric']} | {formatted} |")
    lines.extend(["", "## Generation Modes", ""])
    if not details.empty:
        for mode, count in details["generation_mode"].value_counts().items():
            lines.append(f"- {mode}: {count} answers")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Evidence overlap is a practical lexical proxy, not a legal or diligence completeness test.",
            "- Local model output, when enabled, can vary by model and should be reviewed against source filings.",
            "- The guardrails are designed to flag unsupported or recommendation-like wording, not to replace human review.",
        ]
    )
    report = "\n".join(lines)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "grounded_generation_eval_report.md").write_text(report, encoding="utf-8")
    return report


def run_grounded_generation_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    answers = read_csv_if_exists(PROCESSED_DIR / "generated_rag_answers.csv")
    if answers.empty:
        answers = run_generation()
    summary, details = evaluate_generated_answers(answers)
    write_csv(summary, PROCESSED_DIR / "grounded_generation_eval.csv")
    write_csv(details, PROCESSED_DIR / "grounded_generation_eval_details.csv")
    generate_groundedness_chart(summary)
    generate_groundedness_report(summary, details)
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated filing answers for grounding and guardrails.")
    parser.parse_args()
    summary, _ = run_grounded_generation_evaluation()
    if summary.empty:
        print("grounded_answers_evaluated=0")
    else:
        print(f"grounded_answers_evaluated={int(summary.loc[summary['metric'] == 'answers_evaluated', 'value'].iloc[0])}")
        print(f"groundedness_score={summary.loc[summary['metric'] == 'groundedness_score', 'value'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
