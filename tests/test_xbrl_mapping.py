from src.sec_xbrl_mapper import normalize_companyfacts_payload


def test_xbrl_mapping_basic_payload():
    payload = {
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": [{"fy": 2024, "form": "10-K", "filed": "2025-02-01", "val": 1000}]}},
                "OperatingIncomeLoss": {"units": {"USD": [{"fy": 2024, "form": "10-K", "filed": "2025-02-01", "val": 200}]}},
                "Assets": {"units": {"USD": [{"fy": 2024, "form": "10-K", "filed": "2025-02-01", "val": 1500}]}},
            }
        }
    }
    frame = normalize_companyfacts_payload(payload, "ABC", "ABC Corp", 123)
    assert frame.loc[0, "revenue"] == 1000
    assert frame.loc[0, "operating_income"] == 200
