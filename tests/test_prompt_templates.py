from __future__ import annotations

import pandas as pd

from src.prompt_templates import (
    BASE_INSTRUCTIONS,
    diligence_answer_prompt,
    diligence_questions_prompt,
    filing_evidence_summary_prompt,
    format_evidence_block,
    insufficient_evidence_response,
    memo_section_prompt,
    red_flag_summary_prompt,
)


def _evidence(n: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "chunk_id": f"HUBS-{i}",
                "company_name": "HubSpot, Inc.",
                "ticker": "HUBS",
                "filing_type": "10-K",
                "filing_date": "2026-02-11",
                "section_label": "MD&A",
                "source_type": "real_sec_filing",
                "evidence_snippet": f"Evidence snippet number {i}.",
            }
            for i in range(n)
        ]
    )


def test_format_evidence_block_empty():
    assert format_evidence_block(pd.DataFrame()) == "No retrieved evidence."


def test_format_evidence_block_includes_fields():
    block = format_evidence_block(_evidence(1))
    assert "Chunk: HUBS-0" in block
    assert "HubSpot, Inc. (HUBS)" in block
    assert "Section: MD&A" in block
    assert "Evidence snippet number 0." in block


def test_format_evidence_block_respects_limit():
    block = format_evidence_block(_evidence(5), limit=2)
    assert block.count("Chunk:") == 2


def test_format_evidence_block_falls_back_to_text_column():
    frame = pd.DataFrame([{"chunk_id": "X", "text": "raw text body"}])
    assert "raw text body" in format_evidence_block(frame)


def test_prompt_functions_embed_base_instructions_and_company():
    evidence = _evidence(1)
    prompts = [
        diligence_answer_prompt("HubSpot", "HUBS", "What drives revenue?", evidence),
        filing_evidence_summary_prompt("HubSpot", "HUBS", evidence),
        red_flag_summary_prompt("HubSpot", "HUBS", evidence),
        memo_section_prompt("HubSpot", "HUBS", "Business Overview", evidence),
        diligence_questions_prompt("HubSpot", "HUBS", evidence),
    ]
    for prompt in prompts:
        assert BASE_INSTRUCTIONS.strip() in prompt
        assert "HubSpot" in prompt
        assert "HUBS" in prompt


def test_diligence_answer_prompt_contains_question_and_json_keys():
    prompt = diligence_answer_prompt("HubSpot", "HUBS", "What drives revenue?", _evidence(1))
    assert "What drives revenue?" in prompt
    assert "cited_chunks" in prompt
    assert "insufficient_evidence" in prompt


def test_memo_section_prompt_names_section():
    prompt = memo_section_prompt("HubSpot", "HUBS", "Liquidity", _evidence(1))
    assert "Liquidity" in prompt


def test_insufficient_evidence_response_shape():
    response = insufficient_evidence_response("What are the liquidity risks?")
    assert response["insufficient_evidence"] is True
    assert response["answer"] == "insufficient evidence found"
    assert response["cited_chunks"] == []
    assert "What are the liquidity risks?" in response["diligence_follow_ups"][0]
