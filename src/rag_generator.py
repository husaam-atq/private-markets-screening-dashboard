from __future__ import annotations

import argparse
import ast
import json
import re
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, ensure_project_dirs, get_env, load_config
from src.filing_rag import run_retrieval_for_companies
from src.llm_client import OllamaClient, is_llm_enabled
from src.prompt_templates import diligence_answer_prompt, insufficient_evidence_response
from src.utils import clean_text, read_csv_if_exists, utc_timestamp, write_csv


BANNED_RECOMMENDATION_PATTERNS = [
    r"\bbuy recommendation\b",
    r"\bsell recommendation\b",
    r"\brecommend (?:buy|sell)\b",
    r"\bshould (?:buy|sell)\b",
    r"\bbuy the stock\b",
    r"\bsell the stock\b",
    r"\binvest now\b",
    r"\bshould invest\b",
    r"\bguaranteed\b",
    r"\brisk-free\b",
    r"\bcertain upside\b",
    r"\brecommendation to purchase\b",
    r"\bprice target\b",
]


def banned_language_count(text: str) -> int:
    value = str(text).lower()
    return sum(1 for pattern in BANNED_RECOMMENDATION_PATTERNS if re.search(pattern, value))


def evidence_overlap_score(answer: str, evidence_text: str) -> float:
    answer_tokens = {token for token in re.findall(r"[a-zA-Z][a-zA-Z]{3,}", str(answer).lower())}
    evidence_tokens = {token for token in re.findall(r"[a-zA-Z][a-zA-Z]{3,}", str(evidence_text).lower())}
    if not answer_tokens:
        return 0.0
    return len(answer_tokens & evidence_tokens) / len(answer_tokens)


def extract_cited_chunks(text: str, evidence: pd.DataFrame) -> list[str]:
    known = set(evidence.get("chunk_id", pd.Series(dtype=str)).dropna().astype(str))
    cited = [chunk for chunk in known if chunk and chunk in str(text)]
    return sorted(cited)


def _safe_json_parse(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    candidates = [text.strip()]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(candidate)
                return parsed if isinstance(parsed, dict) else None
            except (SyntaxError, ValueError):
                continue
    return None


def deterministic_answer(question: str, evidence: pd.DataFrame) -> dict[str, Any]:
    if evidence.empty or float(evidence.get("combined_score", pd.Series([0])).max()) < 0.045:
        return insufficient_evidence_response(question)
    top = evidence.sort_values("rank" if "rank" in evidence.columns else "combined_score", ascending=True).head(5)
    cited = top["chunk_id"].dropna().astype(str).tolist()
    sections = ", ".join(top["section_label"].dropna().astype(str).drop_duplicates().head(3).tolist())
    source_types = ", ".join(top["source_type"].dropna().astype(str).drop_duplicates().tolist())
    snippets = top["evidence_snippet" if "evidence_snippet" in top.columns else "text"].fillna("").astype(str).head(3).tolist()
    answer = (
        f"Retrieved evidence indicates that this diligence question is most directly supported by {sections} filing passages. "
        f"The strongest passages should be reviewed in the source filing before drawing a diligence conclusion. "
        f"Cited chunks: {', '.join(cited[:5])}."
    )
    return {
        "answer": answer,
        "evidence_summary": " ".join(snippets)[:1200],
        "key_points": [snippet[:220] for snippet in snippets[:3]],
        "risks": ["Evidence is public filing text and may not cover all diligence topics."],
        "diligence_follow_ups": [
            "Review cited filing sections in full.",
            "Tie filing statements back to structured financial metrics and peer benchmarks.",
        ],
        "cited_chunks": cited,
        "insufficient_evidence": False,
        "source_types": source_types,
    }


def apply_generation_guardrails(answer_payload: dict[str, Any], evidence: pd.DataFrame) -> dict[str, Any]:
    answer_text = clean_text(" ".join(str(answer_payload.get(key, "")) for key in ["answer", "evidence_summary", "key_points", "risks"]))
    evidence_text = clean_text(" ".join(evidence.get("text", pd.Series(dtype=str)).fillna("").astype(str).tolist()))
    cited = answer_payload.get("cited_chunks", [])
    if isinstance(cited, str):
        cited = [item.strip() for item in cited.split(",") if item.strip()]
    known = set(evidence.get("chunk_id", pd.Series(dtype=str)).dropna().astype(str))
    citation_coverage = 1.0 if cited and all(chunk in known for chunk in cited) else 0.0
    overlap = evidence_overlap_score(answer_text, evidence_text)
    banned_count = banned_language_count(answer_text)
    insufficient = bool(answer_payload.get("insufficient_evidence", False))
    unsupported_warning = citation_coverage < 1.0 or overlap < 0.15 or banned_count > 0
    if evidence.empty and not insufficient:
        unsupported_warning = True
    return {
        "cited_chunks": cited,
        "citation_coverage": citation_coverage,
        "evidence_overlap_score": overlap,
        "banned_language_count": banned_count,
        "unsupported_claim_warning": unsupported_warning,
    }


class RAGAnswerGenerator:
    def __init__(
        self,
        llm_enabled: bool | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        cfg = load_config("rag_config.yaml").get("generation", {})
        self.llm_enabled = is_llm_enabled() if llm_enabled is None else llm_enabled
        self.model_name = model_name or get_env("OLLAMA_MODEL") or cfg.get("ollama_model", "qwen3")
        self.temperature = float(temperature if temperature is not None else get_env("LLM_TEMPERATURE", cfg.get("temperature", 0.0)))
        self.max_tokens = int(max_tokens if max_tokens is not None else get_env("LLM_MAX_TOKENS", cfg.get("max_tokens", 700)))
        self.client = OllamaClient(model_name=self.model_name)

    def generate_answer(self, ticker: str, company_name: str, question: str, evidence: pd.DataFrame) -> dict[str, Any]:
        fallback = deterministic_answer(question, evidence)
        generation_mode = "deterministic"
        model_used = ""
        warning = ""
        payload = fallback
        if self.llm_enabled and not evidence.empty:
            prompt = diligence_answer_prompt(company_name, ticker, question, evidence)
            response = self.client.generate(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
            warning = response.warning
            if response.used_llm:
                parsed = _safe_json_parse(response.text)
                if parsed:
                    payload = parsed
                    generation_mode = "local_llm"
                    model_used = response.model_name
                else:
                    payload = {
                        "answer": response.text,
                        "evidence_summary": response.text[:1000],
                        "key_points": [],
                        "risks": [],
                        "diligence_follow_ups": ["Review local model output against cited filing chunks."],
                        "cited_chunks": extract_cited_chunks(response.text, evidence),
                        "insufficient_evidence": "insufficient evidence" in response.text.lower(),
                    }
                    generation_mode = "local_llm"
                    model_used = response.model_name
                    warning = "Local model returned non-JSON text; guardrails were applied to the raw answer."
        guardrails = apply_generation_guardrails(payload, evidence)
        payload = payload | guardrails
        return {
            "ticker": ticker,
            "company_name": company_name,
            "question": question,
            "answer": clean_text(payload.get("answer", "")),
            "evidence_summary": clean_text(payload.get("evidence_summary", "")),
            "key_points": json.dumps(payload.get("key_points", [])),
            "risks": json.dumps(payload.get("risks", [])),
            "diligence_follow_ups": json.dumps(payload.get("diligence_follow_ups", [])),
            "cited_chunks": "; ".join(payload.get("cited_chunks", [])),
            "evidence_strength": float(evidence["combined_score"].max()) if not evidence.empty else 0.0,
            "generation_mode": generation_mode,
            "model_name": model_used,
            "llm_warning": warning,
            "unsupported_claim_warning": bool(payload["unsupported_claim_warning"]),
            "citation_coverage": float(payload["citation_coverage"]),
            "evidence_overlap_score": float(payload["evidence_overlap_score"]),
            "banned_language_count": int(payload["banned_language_count"]),
            "no_answer_flag": evidence.empty or bool(payload.get("insufficient_evidence", False)),
            "generated_at": utc_timestamp(),
        }


def run_generation(tickers: list[str] | None = None, llm_enabled: bool | None = None) -> pd.DataFrame:
    ensure_project_dirs()
    evidence = read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv")
    questions = read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv")
    if evidence.empty or questions.empty:
        run_retrieval_for_companies(tickers=tickers, top_k=5)
        evidence = read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv")
        questions = read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv")
    if tickers:
        questions = questions[questions["ticker"].isin(tickers)].copy()
    generator = RAGAnswerGenerator(llm_enabled=llm_enabled)
    rows: list[dict[str, Any]] = []
    for _, question in questions.iterrows():
        subset = evidence[
            (evidence["ticker"] == question["ticker"])
            & (evidence["question_id"] == question["question_id"])
        ].sort_values("rank")
        rows.append(
            generator.generate_answer(
                ticker=str(question["ticker"]),
                company_name=str(question["company_name"]),
                question=str(question["question"]),
                evidence=subset,
            )
            | {"question_id": question["question_id"]}
        )
    output = pd.DataFrame(rows)
    write_csv(output, PROCESSED_DIR / "generated_rag_answers.csv")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate source-grounded filing answers with deterministic or local Ollama mode.")
    parser.add_argument("--ticker", action="append", default=None)
    parser.add_argument("--llm", action="store_true", help="Attempt local Ollama generation.")
    args = parser.parse_args()
    answers = run_generation(tickers=args.ticker, llm_enabled=args.llm if args.llm else None)
    modes = answers["generation_mode"].value_counts().to_dict() if not answers.empty else {}
    print(f"generated_answers={len(answers)}")
    print(f"generation_modes={modes}")


if __name__ == "__main__":
    main()
