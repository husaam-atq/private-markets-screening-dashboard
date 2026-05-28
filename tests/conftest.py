from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.config import DATA_DIR, OUTPUT_DIR
from src.filing_parser import build_sample_filing_chunks
from src.retrieval_eval import build_gold_questions
from src.screening_pipeline import run_pipeline
from src.yfinance_client import build_sample_market_data
from src.sec_client import build_sample_sec_fundamentals


@pytest.fixture(scope="session", autouse=True)
def preserve_generated_artifacts(tmp_path_factory):
    backup_root = tmp_path_factory.mktemp("artifact_backup")
    targets = [DATA_DIR / "interim", DATA_DIR / "processed", DATA_DIR / "sample", OUTPUT_DIR]
    backups: list[tuple[Path, Path]] = []
    for target in targets:
        if target.exists():
            backup = backup_root / target.name
            shutil.copytree(target, backup)
            backups.append((target, backup))
    yield
    for target, backup in backups:
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(backup, target)


@pytest.fixture(scope="session")
def sample_outputs():
    build_sample_market_data()
    build_sample_sec_fundamentals()
    outputs = run_pipeline(mode="sample")
    chunks = build_sample_filing_chunks()
    gold = build_gold_questions(chunks)
    return outputs | {"chunks": chunks, "gold": gold}
