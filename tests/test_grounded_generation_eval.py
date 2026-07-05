from __future__ import annotations

import pandas as pd

from src import grounded_generation_eval as gge
from src.config import CHART_DIR, REPORT_DIR
from src.grounded_generation_eval import (
    evaluate_generated_answers,
    generate_groundedness_chart,
    generate_groundedness_report,
    run_grounded_generation_evaluation,
)


def _answers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "HUBS",
                "question": "What drives revenue?",
                "answer": "Retrieved evidence indicates subscription revenue drivers. Cited chunks: HUBS-1.",
                "citation_coverage": 1.0,
                "evidence_overlap_score": 0.5,
                "banned_language_count": 0,
                "unsupported_claim_warning": False,
                "no_answer_flag": False,
                "generation_mode": "deterministic",
            },
            {
                "ticker": "ROKU",
                "question": "Any answer?",
                "answer": "insufficient evidence found",
                "citation_coverage": 0.0,
                "evidence_overlap_score": 0.0,
                "banned_language_count": 0,
                "unsupported_claim_warning": True,
                "no_answer_flag": True,
                "generation_mode": "deterministic",
            },
        ]
    )


def test_evaluate_generated_answers_empty_returns_empty_frames():
    summary, details = evaluate_generated_answers(pd.DataFrame())
    assert summary.empty
    assert details.empty


def test_evaluate_generated_answers_metrics_and_flags():
    summary, details = evaluate_generated_answers(_answers())
    metrics = dict(zip(summary["metric"], summary["value"]))
    assert metrics["answers_evaluated"] == 2
    assert metrics["citation_coverage"] == 0.5
    assert metrics["no_answer_accuracy"] == 1.0
    # both answers have no percentage tokens -> numeric consistency all pass
    assert metrics["numeric_consistency_pass_rate"] == 1.0
    assert set(details["citation_pass"]) == {True, False}
    assert details["groundedness_score"].iloc[0] > details["groundedness_score"].iloc[1]


def test_numeric_consistency_flags_percentage_in_answer():
    rows = _answers().head(1).copy()
    rows.loc[0, "answer"] = "Margins expanded 12.5% year over year."
    _summary, details = evaluate_generated_answers(rows)
    assert details["numeric_consistency_pass"].iloc[0] is False or bool(
        details["numeric_consistency_pass"].iloc[0]
    ) is False


def test_no_answer_accuracy_flag_false_when_missing_disclaimer():
    rows = _answers().tail(1).copy()
    rows.loc[rows.index[0], "answer"] = "This is a confident answer without the disclaimer."
    _summary, details = evaluate_generated_answers(rows)
    assert bool(details["no_answer_accuracy_flag"].iloc[0]) is False


def test_generate_groundedness_chart_writes_png():
    summary, _details = evaluate_generated_answers(_answers())
    target = CHART_DIR / "grounded_generation_eval_summary.png"
    if target.exists():
        target.unlink()
    generate_groundedness_chart(summary)
    assert target.exists()


def test_generate_groundedness_chart_noop_on_empty():
    # Should not raise
    generate_groundedness_chart(pd.DataFrame())


def test_generate_groundedness_report_writes_markdown():
    summary, details = evaluate_generated_answers(_answers())
    report = generate_groundedness_report(summary, details)
    assert "# Grounded Generation Evaluation Report" in report
    assert "deterministic: 2 answers" in report
    assert (REPORT_DIR / "grounded_generation_eval_report.md").exists()


def test_run_grounded_generation_evaluation_uses_cached_answers(monkeypatch):
    monkeypatch.setattr(gge, "read_csv_if_exists", lambda path: _answers())

    def fail_generation():  # pragma: no cover - must not be called
        raise AssertionError("run_generation should be skipped when answers exist")

    monkeypatch.setattr(gge, "run_generation", fail_generation)
    summary, details = run_grounded_generation_evaluation()
    assert not summary.empty
    assert len(details) == 2


def test_run_grounded_generation_evaluation_falls_back_to_generation(monkeypatch):
    monkeypatch.setattr(gge, "read_csv_if_exists", lambda path: pd.DataFrame())
    monkeypatch.setattr(gge, "run_generation", _answers)
    summary, _details = run_grounded_generation_evaluation()
    metrics = dict(zip(summary["metric"], summary["value"]))
    assert metrics["answers_evaluated"] == 2
