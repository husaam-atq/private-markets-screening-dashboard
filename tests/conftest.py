from __future__ import annotations

import pytest

from src.filing_parser import build_sample_filing_chunks
from src.retrieval_eval import build_gold_questions
from src.screening_pipeline import run_pipeline
from src.yfinance_client import build_sample_market_data
from src.sec_client import build_sample_sec_fundamentals


@pytest.fixture(scope="session")
def sample_outputs():
    build_sample_market_data()
    build_sample_sec_fundamentals()
    outputs = run_pipeline(mode="sample")
    chunks = build_sample_filing_chunks()
    gold = build_gold_questions(chunks)
    return outputs | {"chunks": chunks, "gold": gold}
