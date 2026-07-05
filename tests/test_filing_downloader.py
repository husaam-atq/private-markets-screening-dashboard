from __future__ import annotations

import pytest
import requests

from src import filing_downloader
from src.filing_downloader import (
    download_filing,
    download_latest_filings,
    filing_url,
    latest_filing_metadata,
)


def _submissions() -> dict:
    return {
        "filings": {
            "recent": {
                "form": ["10-K", "8-K", "10-Q", "10-K"],
                "filingDate": ["2026-02-01", "2026-01-15", "2025-11-01", "2025-02-01"],
                "accessionNumber": [
                    "0001234567-26-000001",
                    "0001234567-26-000002",
                    "0001234567-25-000010",
                    "0001234567-25-000001",
                ],
                "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm"],
            }
        }
    }


def test_latest_filing_metadata_filters_forms():
    metadata = latest_filing_metadata(_submissions(), forms=("10-K",))
    assert list(metadata["form"]) == ["10-K", "10-K"]
    assert list(metadata["accession_number"]) == [
        "0001234567-26-000001",
        "0001234567-25-000001",
    ]
    assert set(metadata.columns) == {
        "form",
        "filing_date",
        "accession_number",
        "primary_document",
    }


def test_latest_filing_metadata_empty_when_no_filings():
    assert latest_filing_metadata({}).empty


def test_filing_url_normalises_accession_and_cik():
    url = filing_url("0000320193", "0001234567-26-000001", "a.htm")
    assert url.endswith("/320193/000123456726000001/a.htm")
    assert url.startswith("https://www.sec.gov/Archives/edgar/data")


def test_download_filing_reads_from_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(filing_downloader, "RAW_DIR", tmp_path)

    def fail_get(*args, **kwargs):  # pragma: no cover - must not be called
        raise AssertionError("network should not be used on cache hit")

    monkeypatch.setattr(filing_downloader.requests, "get", fail_get)

    cache_path = tmp_path / "filings" / "320193" / "000123456726000001_a.htm"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("cached filing text", encoding="utf-8")

    text = download_filing("320193", "0001234567-26-000001", "a.htm")
    assert text == "cached filing text"


def test_download_filing_fetches_and_writes_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(filing_downloader, "RAW_DIR", tmp_path)

    class FakeResponse:
        text = "downloaded filing body"

        def raise_for_status(self):
            return None

    captured = {}

    def fake_get(url, headers=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        return FakeResponse()

    monkeypatch.setattr(filing_downloader.requests, "get", fake_get)

    text = download_filing("320193", "0001234567-26-000001", "a.htm")
    assert text == "downloaded filing body"
    assert "User-Agent" in captured["headers"]
    cache_path = tmp_path / "filings" / "320193" / "000123456726000001_a.htm"
    assert cache_path.read_text(encoding="utf-8") == "downloaded filing body"


def test_download_filing_raises_on_http_error(monkeypatch, tmp_path):
    monkeypatch.setattr(filing_downloader, "RAW_DIR", tmp_path)

    class FakeResponse:
        text = ""

        def raise_for_status(self):
            raise requests.HTTPError("404")

    monkeypatch.setattr(filing_downloader.requests, "get", lambda *a, **k: FakeResponse())

    with pytest.raises(requests.HTTPError):
        download_filing("320193", "0001234567-26-000001", "missing.htm")


def test_download_latest_filings_combines_metadata_and_text(monkeypatch):
    class FakeClient:
        def get_submissions(self, cik):
            return _submissions()

    monkeypatch.setattr(filing_downloader, "SECClient", FakeClient)
    monkeypatch.setattr(
        filing_downloader,
        "download_filing",
        lambda cik, accession, document: f"text::{accession}",
    )

    results = download_latest_filings(320193, forms=("10-K",), limit=1)
    assert len(results) == 1
    assert results[0]["form"] == "10-K"
    assert results[0]["text"] == "text::0001234567-26-000001"


def test_main_without_cik_prints_hint(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["filing_downloader", "--cik", ""])
    filing_downloader.main()
    assert "Provide --cik" in capsys.readouterr().out


def test_main_with_cik_downloads(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["filing_downloader", "--cik", "320193", "--limit", "2"])
    monkeypatch.setattr(
        filing_downloader,
        "download_latest_filings",
        lambda cik, limit=2: [{"text": "a"}, {"text": "b"}],
    )
    filing_downloader.main()
    assert "downloaded_filings=2" in capsys.readouterr().out
