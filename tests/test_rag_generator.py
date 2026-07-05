from __future__ import annotations

import json

import pandas as pd

from src import rag_generator
from src.llm_client import LLMResponse
from src.rag_generator import (
    RAGAnswerGenerator,
    _safe_json_parse,
    deterministic_answer,
    evidence_overlap_score,
    extract_cited_chunks,
    run_generation,
)


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "HUBS",
                "question_id": "revenue_drivers",
                "company_name": "HubSpot, Inc.",
                "chunk_id": "HUBS-MDA-1",
                "section_label": "MD&A",
                "source_type": "real_sec_filing",
                "text": "Revenue growth is driven by subscription customers and product adoption.",
                "evidence_snippet": "Revenue growth is driven by subscription customers.",
                "combined_score": 0.42,
                "rank": 1,
            },
            {
                "ticker": "HUBS",
                "question_id": "revenue_drivers",
                "company_name": "HubSpot, Inc.",
                "chunk_id": "HUBS-LIQ-1",
                "section_label": "Liquidity",
                "source_type": "real_sec_filing",
                "text": "Deferred revenue and cash flows are discussed in liquidity resources.",
                "evidence_snippet": "Deferred revenue and cash flows.",
                "combined_score": 0.30,
                "rank": 2,
            },
        ]
    )


def test_safe_json_parse_plain_object():
    assert _safe_json_parse('{"a": 1}') == {"a": 1}


def test_safe_json_parse_embedded_object():
    text = "Here is the answer:\n{\"answer\": \"ok\"}\nThanks"
    assert _safe_json_parse(text) == {"answer": "ok"}


def test_safe_json_parse_python_literal_fallback():
    assert _safe_json_parse("{'answer': 'ok'}") == {"answer": "ok"}


def test_safe_json_parse_returns_none_for_non_dict_and_empty():
    assert _safe_json_parse("") is None
    assert _safe_json_parse("[1, 2, 3]") is None
    assert _safe_json_parse("not json at all") is None


def test_extract_cited_chunks_finds_known_ids():
    evidence = _evidence()
    cited = extract_cited_chunks("See HUBS-MDA-1 and HUBS-LIQ-1 for detail.", evidence)
    assert cited == ["HUBS-LIQ-1", "HUBS-MDA-1"]


def test_evidence_overlap_score_bounds():
    assert evidence_overlap_score("", "anything") == 0.0
    score = evidence_overlap_score("revenue growth subscription", "revenue growth adoption")
    assert 0.0 < score <= 1.0


def test_deterministic_answer_insufficient_for_weak_evidence():
    weak = _evidence().copy()
    weak["combined_score"] = 0.001
    answer = deterministic_answer("What drives revenue?", weak)
    assert answer["insufficient_evidence"] is True


def test_deterministic_answer_cites_chunks_for_strong_evidence():
    answer = deterministic_answer("What drives revenue?", _evidence())
    assert answer["insufficient_evidence"] is False
    assert "HUBS-MDA-1" in answer["cited_chunks"]


def test_generate_answer_local_llm_json(monkeypatch):
    generator = RAGAnswerGenerator(llm_enabled=True)
    payload = {
        "answer": "Subscription revenue drivers per HUBS-MDA-1.",
        "evidence_summary": "Revenue growth.",
        "cited_chunks": ["HUBS-MDA-1"],
        "insufficient_evidence": False,
    }

    class FakeClient:
        def generate(self, prompt, temperature=0.0, max_tokens=700):
            return LLMResponse(json.dumps(payload), "qwen3:latest", True)

    generator.client = FakeClient()
    result = generator.generate_answer("HUBS", "HubSpot, Inc.", "What drives revenue?", _evidence())
    assert result["generation_mode"] == "local_llm"
    assert result["model_name"] == "qwen3:latest"
    assert result["citation_coverage"] == 1.0


def test_generate_answer_local_llm_non_json_text(monkeypatch):
    generator = RAGAnswerGenerator(llm_enabled=True)

    class FakeClient:
        def generate(self, prompt, temperature=0.0, max_tokens=700):
            return LLMResponse("Free-form answer mentioning HUBS-MDA-1.", "qwen3:latest", True)

    generator.client = FakeClient()
    result = generator.generate_answer("HUBS", "HubSpot, Inc.", "What drives revenue?", _evidence())
    assert result["generation_mode"] == "local_llm"
    assert "non-JSON" in result["llm_warning"]
    assert "HUBS-MDA-1" in result["cited_chunks"]


def test_generate_answer_deterministic_when_llm_disabled():
    generator = RAGAnswerGenerator(llm_enabled=False)
    result = generator.generate_answer("HUBS", "HubSpot, Inc.", "What drives revenue?", _evidence())
    assert result["generation_mode"] == "deterministic"
    assert result["no_answer_flag"] is False


def test_run_generation_uses_cached_frames(monkeypatch):
    questions = pd.DataFrame(
        [
            {
                "ticker": "HUBS",
                "question_id": "revenue_drivers",
                "company_name": "HubSpot, Inc.",
                "question": "What drives revenue?",
            }
        ]
    )

    def fake_read(path):
        name = str(path)
        if "retrieved_evidence" in name:
            return _evidence()
        if "diligence_questions" in name:
            return questions
        return pd.DataFrame()

    monkeypatch.setattr(rag_generator, "read_csv_if_exists", fake_read)
    monkeypatch.setattr(rag_generator, "write_csv", lambda frame, path: None)

    def fail_retrieval(*a, **k):  # pragma: no cover - must not run when cache present
        raise AssertionError("retrieval should be skipped")

    monkeypatch.setattr(rag_generator, "run_retrieval_for_companies", fail_retrieval)

    output = run_generation(llm_enabled=False)
    assert len(output) == 1
    assert output.iloc[0]["ticker"] == "HUBS"
    assert output.iloc[0]["question_id"] == "revenue_drivers"
