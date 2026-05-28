from __future__ import annotations

import pandas as pd


BASE_INSTRUCTIONS = """
You are supporting a private markets screening workflow. Answer only from the retrieved public filing evidence.
If the answer is not supported by the evidence, state that insufficient evidence was found.
Do not provide investment advice or buy/sell recommendations.
Use further-diligence language only. Cite chunk IDs, filing sections and filing dates.
Separate factual evidence from analyst interpretation. Keep the answer professional and concise.
"""


def format_evidence_block(evidence: pd.DataFrame, limit: int = 5) -> str:
    if evidence.empty:
        return "No retrieved evidence."
    rows = []
    for _, row in evidence.head(limit).iterrows():
        rows.append(
            "\n".join(
                [
                    f"Chunk: {row.get('chunk_id', '')}",
                    f"Company: {row.get('company_name', '')} ({row.get('ticker', '')})",
                    f"Filing: {row.get('filing_type', '')} filed {row.get('filing_date', '')}",
                    f"Section: {row.get('section_label', '')}",
                    f"Source type: {row.get('source_type', '')}",
                    f"Evidence: {str(row.get('evidence_snippet', row.get('text', '')))[:900]}",
                ]
            )
        )
    return "\n\n".join(rows)


def diligence_answer_prompt(company_name: str, ticker: str, question: str, evidence: pd.DataFrame) -> str:
    return f"""{BASE_INSTRUCTIONS}

Company: {company_name} ({ticker})
Question: {question}

Retrieved evidence:
{format_evidence_block(evidence)}

Return a concise JSON object with keys:
answer, evidence_summary, key_points, risks, diligence_follow_ups, cited_chunks, insufficient_evidence.
"""


def filing_evidence_summary_prompt(company_name: str, ticker: str, evidence: pd.DataFrame) -> str:
    return f"""{BASE_INSTRUCTIONS}

Summarise the filing evidence for {company_name} ({ticker}) using only these chunks:
{format_evidence_block(evidence)}
"""


def red_flag_summary_prompt(company_name: str, ticker: str, evidence: pd.DataFrame) -> str:
    return f"""{BASE_INSTRUCTIONS}

Identify source-backed red flags for {company_name} ({ticker}) using only these chunks.
If the chunks do not support a red flag, say insufficient evidence found.

{format_evidence_block(evidence)}
"""


def memo_section_prompt(company_name: str, ticker: str, memo_section: str, evidence: pd.DataFrame) -> str:
    return f"""{BASE_INSTRUCTIONS}

Draft the {memo_section} section for {company_name} ({ticker}) using only these chunks:
{format_evidence_block(evidence)}
"""


def diligence_questions_prompt(company_name: str, ticker: str, evidence: pd.DataFrame) -> str:
    return f"""{BASE_INSTRUCTIONS}

Generate diligence follow-up questions for {company_name} ({ticker}) using only these chunks:
{format_evidence_block(evidence)}
"""


def insufficient_evidence_response(question: str) -> dict:
    return {
        "answer": "insufficient evidence found",
        "evidence_summary": "No retrieved filing passages were strong enough to support an answer.",
        "key_points": [],
        "risks": ["Evidence gap requires source filing review."],
        "diligence_follow_ups": [f"Review source filings manually for: {question}"],
        "cited_chunks": [],
        "insufficient_evidence": True,
    }
