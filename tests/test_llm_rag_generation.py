import pandas as pd
import requests

from src.filing_rag import FilingRetriever
from src.grounded_generation_eval import evaluate_generated_answers
from src.llm_client import OllamaClient
from src.rag_generator import (
    RAGAnswerGenerator,
    apply_generation_guardrails,
    banned_language_count,
    deterministic_answer,
)


def _chunks():
    return pd.DataFrame(
        [
            {
                "ticker": "HUBS",
                "company_name": "HubSpot, Inc.",
                "cik": 1404655,
                "filing_type": "10-K",
                "filing_date": "2026-02-11",
                "accession_number": "000-test",
                "section_label": "Liquidity and Capital Resources",
                "chunk_id": "HUBS-LIQ-1",
                "text": "Deferred revenue and cash flows are discussed in liquidity resources.",
                "source_type": "real_sec_filing",
                "source_url": "https://www.sec.gov/example",
            },
            {
                "ticker": "HUBS",
                "company_name": "HubSpot, Inc.",
                "cik": 1404655,
                "filing_type": "10-K",
                "filing_date": "2026-02-11",
                "accession_number": "000-test",
                "section_label": "MD&A",
                "chunk_id": "HUBS-MDA-1",
                "text": "Revenue growth is driven by subscription customers, product adoption and retention.",
                "source_type": "real_sec_filing",
                "source_url": "https://www.sec.gov/example",
            },
        ]
    )


def test_section_boost_routes_revenue_question_to_mda():
    retriever = FilingRetriever(_chunks())
    result = retriever.search("What are the company's main revenue drivers?", ticker="HUBS", top_k=2)
    assert result.iloc[0]["section_label"] == "MD&A"
    assert result.iloc[0]["section_boost"] > 0


def test_ollama_unavailable_falls_back(monkeypatch):
    def raise_connection(*args, **kwargs):
        raise requests.ConnectionError("offline")

    monkeypatch.setattr(requests, "get", raise_connection)
    client = OllamaClient(base_url="http://localhost:11434", model_name="missing")
    assert client.is_ollama_available() is False
    response = client.generate("test")
    assert response.used_llm is False
    assert "fallback" in response.warning


def test_rag_generator_deterministic_includes_cited_chunks():
    evidence = FilingRetriever(_chunks()).search("What are the company's main revenue drivers?", ticker="HUBS", top_k=2)
    evidence["rank"] = range(1, len(evidence) + 1)
    answer = RAGAnswerGenerator(llm_enabled=False).generate_answer(
        "HUBS",
        "HubSpot, Inc.",
        "What are the company's main revenue drivers?",
        evidence,
    )
    assert answer["generation_mode"] == "deterministic"
    assert "HUBS-MDA-1" in answer["cited_chunks"]
    assert answer["banned_language_count"] == 0


def test_guardrails_detect_banned_language_and_unsupported_claim():
    evidence = _chunks().head(1)
    payload = {
        "answer": "This is a buy with guaranteed certain upside.",
        "evidence_summary": "Unsupported phrase.",
        "cited_chunks": [],
    }
    result = apply_generation_guardrails(payload, evidence)
    assert result["banned_language_count"] >= 2
    assert result["unsupported_claim_warning"] is True
    assert banned_language_count("price target and recommendation to purchase") == 2


def test_grounded_generation_eval_outputs_metrics():
    answer = deterministic_answer("What are the company's main revenue drivers?", _chunks())
    rows = pd.DataFrame(
        [
            {
                "ticker": "HUBS",
                "question": "What are the company's main revenue drivers?",
                "answer": answer["answer"],
                "citation_coverage": 1.0,
                "evidence_overlap_score": 0.5,
                "banned_language_count": 0,
                "unsupported_claim_warning": False,
                "no_answer_flag": False,
                "generation_mode": "deterministic",
            }
        ]
    )
    summary, details = evaluate_generated_answers(rows)
    assert "groundedness_score" in set(summary["metric"])
    assert details["groundedness_score"].iloc[0] > 0.8
