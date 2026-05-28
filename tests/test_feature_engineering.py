from src.feature_engineering import METRIC_COLUMNS, build_screening_universe


def test_feature_engineering_calculations(sample_outputs):
    frame = sample_outputs["screening"]
    assert len(frame) >= 30
    assert set(["revenue_growth_yoy", "gross_margin", "fcf_conversion", "ev_to_sales"]).issubset(frame.columns)
    assert len(METRIC_COLUMNS) >= 20
    assert frame["data_completeness_ratio"].between(0, 1).all()
