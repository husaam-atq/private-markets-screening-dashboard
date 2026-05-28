from src.filing_rag import FilingRetriever


def test_retrieval_returns_metadata(sample_outputs):
    retriever = FilingRetriever(sample_outputs["chunks"])
    result = retriever.search("main revenue drivers recurring customer relationships", ticker="MSFT", top_k=5)
    assert not result.empty
    assert {"ticker", "section_label", "chunk_id", "combined_score"}.issubset(result.columns)


def test_no_answer_handling(sample_outputs):
    retriever = FilingRetriever(sample_outputs["chunks"])
    result = retriever.answer_question("Project Atlas binding acquisition plan", ticker="MSFT", top_k=5)
    assert result.no_answer or result.evidence["combined_score"].max() < 0.12
