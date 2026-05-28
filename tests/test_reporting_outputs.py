from pathlib import Path

from src.charts import generate_all_charts
from src.config import CHART_DIR, EXCEL_DIR, REPORT_DIR
from src.filing_rag import run_retrieval_for_companies
from src.memo_generator import generate_memo
from src.reporting import generate_excel_workbook, generate_filing_evidence_report, generate_rag_evaluation_report
from src.retrieval_eval import run_retrieval_evaluation


def test_memo_chart_excel_report_outputs(sample_outputs):
    run_retrieval_for_companies(top_k=5)
    run_retrieval_evaluation()
    memo, memo_rows = generate_memo("MSFT")
    assert "Investment Screening Memo" in memo
    assert not memo_rows.empty
    generate_filing_evidence_report()
    generate_rag_evaluation_report()
    generate_all_charts()
    workbook = generate_excel_workbook()
    assert workbook.exists()
    assert (REPORT_DIR / "investment_screening_memo.md").exists()
    assert (REPORT_DIR / "filing_evidence_report.md").exists()
    assert (REPORT_DIR / "rag_evaluation_report.md").exists()
    assert (CHART_DIR / "top_screened_companies.png").exists()
