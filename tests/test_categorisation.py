from src.categorisation import CATEGORY_ORDER, categorize_companies


def test_category_rules(sample_outputs):
    categories = sample_outputs["categories"]
    assert categories["primary_category"].isin(CATEGORY_ORDER).all()
    assert categories["why_this_category"].str.contains("investment score").all()
