from src.sec_client import SECClient, SAMPLE_CIKS


def test_sec_cik_mapping_helpers():
    assert SECClient.cik_padded(SAMPLE_CIKS["MSFT"]) == "0000789019"
    mapping = SECClient().get_cik_mapping()
    assert {"ticker", "cik"}.issubset(mapping.columns)
