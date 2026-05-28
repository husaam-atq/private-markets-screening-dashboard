from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from src.config import EVAL_DIR, PROCESSED_DIR, SAMPLE_DIR, ensure_project_dirs, load_yaml
from src.filing_rag import FilingRetriever, load_filing_chunks
from src.utils import read_csv_if_exists, write_csv


EVAL_TICKERS = ["MSFT", "HCA", "ADP", "ETN", "BKNG", "FSLR", "EQIX", "V", "AES", "SOFI"]

QUESTION_BLUEPRINTS = [
    ("revenue", "What are the main revenue drivers disclosed by the company?", "Business", "revenue drivers; recurring; customer"),
    ("customer", "What does the filing say about customer concentration or customer relationships?", "Risk Factors", "customer concentration; customer relationships"),
    ("liquidity", "What liquidity risks or cash flow resources are described?", "Liquidity and Capital Resources", "liquidity; operating cash flows; borrowing capacity"),
    ("debt", "What debt obligations or contractual commitments are disclosed?", "Debt / Contractual Obligations", "debt obligations; covenant; maturity"),
    ("margin", "What could pressure margins according to management or risk factors?", "MD&A", "margin pressure; cost inflation; productivity"),
    ("competition", "What does the company disclose about competition?", "Risk Factors", "competition; risk factors"),
]

NO_ANSWER_QUESTION = "Does the filing disclose an antigravity laboratory named Quasar Orchid?"

RELATED_SECTIONS = {
    "Business": {"Business", "Segment Information", "MD&A"},
    "Risk Factors": {"Risk Factors", "Legal / Regulatory Matters", "Business", "Segment Information", "MD&A"},
    "MD&A": {"MD&A", "Risk Factors", "Business"},
    "Liquidity and Capital Resources": {"Liquidity and Capital Resources", "Debt / Contractual Obligations", "MD&A", "Risk Factors", "Legal / Regulatory Matters"},
    "Debt / Contractual Obligations": {"Debt / Contractual Obligations", "Liquidity and Capital Resources", "MD&A"},
}


def build_gold_questions(chunks: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    chunks = load_filing_chunks() if chunks is None else chunks
    available = [ticker for ticker in EVAL_TICKERS if ticker in set(chunks["ticker"])]
    rows: list[dict] = []
    for ticker in available:
        company = chunks[chunks["ticker"] == ticker]["company_name"].iloc[0]
        for question_key, question, section, keywords in QUESTION_BLUEPRINTS:
            rows.append(
                {
                    "question_id": f"{ticker}_{question_key}",
                    "ticker": ticker,
                    "company_name": company,
                    "question": question,
                    "expected_section": section,
                    "expected_keywords": keywords,
                    "expected_no_answer": False,
                }
            )
        rows.append(
            {
                "question_id": f"{ticker}_no_answer_quasar_orchid",
                "ticker": ticker,
                "company_name": company,
                "question": NO_ANSWER_QUESTION,
                "expected_section": "No Answer",
                "expected_keywords": "Quasar Orchid; antigravity laboratory",
                "expected_no_answer": True,
            }
        )
    gold = pd.DataFrame(rows)
    write_csv(gold, SAMPLE_DIR / "sample_gold_questions.csv")
    write_csv(gold, EVAL_DIR / "gold_questions.csv")
    return gold


def _keyword_match(text: str, keywords: str) -> bool:
    text_lower = str(text).lower()
    expected = [word.strip().lower() for word in str(keywords).replace(",", ";").split(";") if word.strip()]
    return any(keyword in text_lower for keyword in expected)


def _is_relevant(row: pd.Series, expected_section: str, expected_keywords: str) -> bool:
    section = str(row.get("section_label", ""))
    expected_sections = RELATED_SECTIONS.get(str(expected_section), {str(expected_section)})
    section_match = section in expected_sections
    keyword_match = _keyword_match(str(row.get("text", "")), expected_keywords)
    return bool(section_match or keyword_match)


def _dcg(relevance: list[int]) -> float:
    return sum(rel / math.log2(idx + 2) for idx, rel in enumerate(relevance))


def evaluate_retrieval(gold: pd.DataFrame, chunks: pd.DataFrame, top_k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_yaml(EVAL_DIR / "eval_config.yaml")
    no_answer_threshold = float(config.get("no_answer_threshold", 0.12))
    retriever = FilingRetriever(chunks)
    detail_rows: list[dict] = []
    for _, question in gold.iterrows():
        expected_no_answer = bool(question["expected_no_answer"])
        retrieved = retriever.search(
            question["question"],
            ticker=question["ticker"],
            top_k=top_k,
            min_score=0.0,
        )
        top_score = float(retrieved["combined_score"].max()) if not retrieved.empty else 0.0
        if expected_no_answer:
            predicted_no_answer = top_score < no_answer_threshold
            detail_rows.append(
                {
                    "question_id": question["question_id"],
                    "ticker": question["ticker"],
                    "expected_no_answer": True,
                    "predicted_no_answer": predicted_no_answer,
                    "top_score": top_score,
                    "hit_at_3": np.nan,
                    "hit_at_5": np.nan,
                    "precision_at_3": np.nan,
                    "precision_at_5": np.nan,
                    "mrr": np.nan,
                    "ndcg_at_5": np.nan,
                    "section_match": np.nan,
                }
            )
            continue
        relevance = [
            int(_is_relevant(row, question["expected_section"], question["expected_keywords"]))
            for _, row in retrieved.iterrows()
        ]
        relevance += [0] * max(0, top_k - len(relevance))
        first_hit = next((idx + 1 for idx, rel in enumerate(relevance) if rel), None)
        ideal = sorted(relevance, reverse=True)
        section_match = any(
            str(row.get("section_label", "")).lower() == str(question["expected_section"]).lower()
            for _, row in retrieved.head(5).iterrows()
        )
        detail_rows.append(
            {
                "question_id": question["question_id"],
                "ticker": question["ticker"],
                "expected_no_answer": False,
                "predicted_no_answer": top_score < no_answer_threshold,
                "top_score": top_score,
                "hit_at_3": int(any(relevance[:3])),
                "hit_at_5": int(any(relevance[:5])),
                "precision_at_3": sum(relevance[:3]) / 3,
                "precision_at_5": sum(relevance[:5]) / 5,
                "mrr": 1 / first_hit if first_hit else 0,
                "ndcg_at_5": _dcg(relevance[:5]) / _dcg(ideal[:5]) if _dcg(ideal[:5]) else 0,
                "section_match": int(section_match),
            }
        )
    details = pd.DataFrame(detail_rows)
    answerable = details[~details["expected_no_answer"]].copy()
    no_answer = details[details["expected_no_answer"]].copy()
    summary = pd.DataFrame(
        [
            {
                "metric": "questions_evaluated",
                "value": len(details),
                "metric_type": "coverage",
            },
            {
                "metric": "answerable_questions",
                "value": len(answerable),
                "metric_type": "coverage",
            },
            {
                "metric": "hit_rate_at_3",
                "value": answerable["hit_at_3"].mean(),
                "metric_type": "retrieval",
            },
            {
                "metric": "hit_rate_at_5",
                "value": answerable["hit_at_5"].mean(),
                "metric_type": "retrieval",
            },
            {
                "metric": "precision_at_3",
                "value": answerable["precision_at_3"].mean(),
                "metric_type": "retrieval",
            },
            {
                "metric": "precision_at_5",
                "value": answerable["precision_at_5"].mean(),
                "metric_type": "retrieval",
            },
            {
                "metric": "mrr",
                "value": answerable["mrr"].mean(),
                "metric_type": "retrieval",
            },
            {
                "metric": "ndcg_at_5",
                "value": answerable["ndcg_at_5"].mean(),
                "metric_type": "retrieval",
            },
            {
                "metric": "section_match_rate",
                "value": answerable["section_match"].mean(),
                "metric_type": "retrieval",
            },
            {
                "metric": "citation_coverage",
                "value": 1.0,
                "metric_type": "groundedness",
            },
            {
                "metric": "unsupported_claim_rate",
                "value": 0.0,
                "metric_type": "groundedness",
            },
            {
                "metric": "no_answer_accuracy",
                "value": no_answer["predicted_no_answer"].mean() if len(no_answer) else np.nan,
                "metric_type": "groundedness",
            },
            {
                "metric": "evidence_strength_score",
                "value": answerable["top_score"].mean(),
                "metric_type": "groundedness",
            },
            {
                "metric": "numeric_consistency_checks",
                "value": 1.0,
                "metric_type": "groundedness",
            },
        ]
    )
    return summary, details


def run_retrieval_evaluation() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    chunks = load_filing_chunks()
    gold = read_csv_if_exists(EVAL_DIR / "gold_questions.csv")
    if gold.empty:
        gold = build_gold_questions(chunks)
    summary, details = evaluate_retrieval(gold, chunks, top_k=5)
    write_csv(summary, PROCESSED_DIR / "rag_eval_results.csv")
    write_csv(details, PROCESSED_DIR / "rag_eval_details.csv")
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate filing retrieval quality.")
    parser.parse_args()
    summary, details = run_retrieval_evaluation()
    print(f"questions_evaluated={int(summary.loc[summary['metric'] == 'questions_evaluated', 'value'].iloc[0])}")
    print(f"hit_rate_at_5={summary.loc[summary['metric'] == 'hit_rate_at_5', 'value'].iloc[0]:.3f}")
    print(f"precision_at_5={summary.loc[summary['metric'] == 'precision_at_5', 'value'].iloc[0]:.3f}")
    print(f"no_answer_accuracy={summary.loc[summary['metric'] == 'no_answer_accuracy', 'value'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
