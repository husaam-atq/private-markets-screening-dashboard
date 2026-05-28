from src.scoring import SCORE_COLUMNS, calculate_scores


def test_scores_bounded(sample_outputs):
    scores = sample_outputs["scores"]
    for column in SCORE_COLUMNS:
        assert scores[column].between(0, 100).all()
    assert scores["investment_screening_explanation"].str.len().min() > 20
