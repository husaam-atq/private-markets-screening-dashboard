from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import INTERIM_DIR, PROCESSED_DIR, SAMPLE_DIR, ensure_project_dirs, load_config
from src.filing_parser import build_sample_filing_chunks
from src.utils import clean_text, read_csv_if_exists, utc_timestamp, write_csv


@dataclass
class RetrievalResult:
    question: str
    no_answer: bool
    evidence: pd.DataFrame


class FilingRetriever:
    def __init__(self, chunks: pd.DataFrame, mode: str = "keyword") -> None:
        self.chunks = chunks.copy().reset_index(drop=True)
        self.mode = mode
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(self.chunks["text"].fillna("").map(clean_text))

    def search(self, query: str, ticker: str | None = None, top_k: int = 5, min_score: float | None = None) -> pd.DataFrame:
        config = load_config("rag_config.yaml")["retrieval"]
        min_score = float(config["min_keyword_score"] if min_score is None else min_score)
        candidate = self.chunks
        matrix = self.matrix
        if ticker:
            mask = candidate["ticker"].eq(ticker).to_numpy()
            candidate = candidate[mask].reset_index(drop=True)
            matrix = self.matrix[mask]
        if candidate.empty:
            return pd.DataFrame()
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, matrix).flatten()
        ranked = np.argsort(scores)[::-1][:top_k]
        result = candidate.iloc[ranked].copy()
        result["keyword_score"] = scores[ranked]
        result["semantic_score"] = np.nan
        result["combined_score"] = result["keyword_score"]
        result["retrieval_mode"] = self.mode
        result["retrieved_at"] = utc_timestamp()
        result = result[result["combined_score"] >= min_score].reset_index(drop=True)
        return result

    def answer_question(self, question: str, ticker: str, top_k: int = 5) -> RetrievalResult:
        evidence = self.search(question, ticker=ticker, top_k=top_k)
        no_answer = evidence.empty
        return RetrievalResult(question=question, no_answer=no_answer, evidence=evidence)


def load_filing_chunks() -> pd.DataFrame:
    chunks = read_csv_if_exists(INTERIM_DIR / "filing_chunks.csv")
    if chunks.empty:
        chunks = read_csv_if_exists(SAMPLE_DIR / "sample_filing_chunks.csv")
    if chunks.empty:
        chunks = build_sample_filing_chunks()
    write_csv(chunks, INTERIM_DIR / "filing_chunks.csv")
    return chunks


def diligence_questions() -> pd.DataFrame:
    questions = load_config("rag_config.yaml")["diligence_questions"]
    return pd.DataFrame(questions)


def run_retrieval_for_companies(tickers: list[str] | None = None, top_k: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_project_dirs()
    chunks = load_filing_chunks()
    retriever = FilingRetriever(chunks)
    if tickers is None:
        scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
        tickers = scores.sort_values("investment_screening_score", ascending=False).head(8)["ticker"].tolist() if not scores.empty else chunks["ticker"].drop_duplicates().head(8).tolist()
    question_frame = diligence_questions()
    evidence_rows: list[pd.DataFrame] = []
    question_rows: list[dict] = []
    for ticker in tickers:
        company = chunks[chunks["ticker"] == ticker]["company_name"].iloc[0]
        for _, question in question_frame.iterrows():
            result = retriever.answer_question(question["question"], ticker=ticker, top_k=top_k)
            strength = float(result.evidence["combined_score"].max()) if not result.evidence.empty else 0.0
            question_rows.append(
                {
                    "ticker": ticker,
                    "company_name": company,
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "evidence_strength": strength,
                    "no_answer_flag": result.no_answer,
                    "summary_answer": "insufficient filing evidence found"
                    if result.no_answer
                    else f"Retrieved filing passages discuss {result.evidence.iloc[0]['section_label'].lower()} themes relevant to the question.",
                }
            )
            if not result.evidence.empty:
                ev = result.evidence.copy()
                ev["question_id"] = question["question_id"]
                ev["question"] = question["question"]
                ev["rank"] = range(1, len(ev) + 1)
                evidence_rows.append(ev)
    questions_out = pd.DataFrame(question_rows)
    evidence_out = pd.concat(evidence_rows, ignore_index=True) if evidence_rows else pd.DataFrame()
    write_csv(questions_out, PROCESSED_DIR / "diligence_questions.csv")
    write_csv(evidence_out, PROCESSED_DIR / "retrieved_evidence.csv")
    return questions_out, evidence_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TF-IDF filing retrieval outputs.")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    chunks = load_filing_chunks()
    Path(INTERIM_DIR / "retrieval_index").mkdir(parents=True, exist_ok=True)
    questions, evidence = run_retrieval_for_companies(top_k=args.top_k)
    print(f"filing_chunks_indexed={len(chunks)}")
    print(f"diligence_questions={len(questions)}")
    print(f"retrieved_evidence_rows={len(evidence)}")


if __name__ == "__main__":
    main()
