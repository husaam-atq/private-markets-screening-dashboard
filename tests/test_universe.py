from src.universe import configured_universe, sample_universe


def test_universe_loading():
    default = configured_universe()
    sample = sample_universe()
    assert len(default) >= 80
    assert len(sample) >= 30
    assert sample["sector_theme"].nunique() >= 5
    assert sample["ticker"].is_unique
