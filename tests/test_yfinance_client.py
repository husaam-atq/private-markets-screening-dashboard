from src.yfinance_client import build_sample_market_data, load_market_data


def test_yfinance_client_sample_fallback():
    sample = build_sample_market_data()
    loaded = load_market_data(mode="sample")
    assert len(sample) >= 30
    assert len(loaded) == len(sample)
    assert loaded["data_source"].eq("cached_sample").all()
    assert loaded["drawdown_from_52w_high"].between(0, 1).all()
