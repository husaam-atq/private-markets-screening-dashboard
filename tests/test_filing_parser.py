from src.filing_parser import chunk_text, parse_filing_to_chunks


def test_filing_chunking():
    raw = "<html>Item 1. Business Revenue drivers and customers. Item 1A. Risk Factors Competition and regulation.</html>"
    frame = parse_filing_to_chunks(raw, "ABC", "ABC Corp", 123, "10-K", "2025-02-01", "000123")
    assert not frame.empty
    assert {"ticker", "section_label", "chunk_id", "text"}.issubset(frame.columns)
    assert chunk_text(" ".join(["word"] * 250), max_words=100, overlap_words=10)
