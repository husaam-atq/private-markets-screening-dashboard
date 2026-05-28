from src.peer_benchmarking import build_peer_benchmarks


def test_peer_benchmarking_calculations(sample_outputs):
    benchmarks = sample_outputs["benchmarks"]
    assert "valuation_premium_discount_to_median" in benchmarks.columns
    assert benchmarks["peer_count"].min() >= 3
    assert benchmarks["revenue_growth_yoy_peer_percentile"].between(0, 100).all()
