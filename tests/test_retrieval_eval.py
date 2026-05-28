from src.retrieval_eval import evaluate_retrieval


def test_retrieval_eval_metrics(sample_outputs):
    summary, details = evaluate_retrieval(sample_outputs["gold"], sample_outputs["chunks"], top_k=5)
    assert "hit_rate_at_5" in set(summary["metric"])
    assert summary.loc[summary["metric"] == "questions_evaluated", "value"].iloc[0] >= 40
    assert details["question_id"].notna().all()
