# Private Markets Investment Screening & Filing Intelligence Dashboard

## Overview

This project is a Python and Streamlit workflow for public-company screening, peer benchmarking and SEC filing evidence retrieval. It is designed as a recruiter-facing portfolio project for private markets investment analytics, comparable-company analysis, filing review and diligence automation roles.

The system ranks listed-company comparables for **further diligence**, not for buy/sell decisions. Structured scores are deterministic and calculated from public market data, SEC-style fundamentals and peer-relative metrics. The filing intelligence layer retrieves source-backed passages to support or challenge the structured screen.

## Why I Built This

Private markets teams often begin with public comparables before moving into confidential diligence. This project mirrors that early-stage workflow:

1. Select a sector or theme.
2. Pull market data and filing fundamentals.
3. Calculate growth, margin, leverage, valuation, cash-conversion and risk metrics.
4. Rank companies with transparent scorecards.
5. Categorise diligence archetypes.
6. Retrieve filing evidence for business model, risk, liquidity, margin and capital allocation questions.
7. Produce memo, Excel, chart and dashboard outputs.

## What Makes This Distinct

- Not a generic stock screener or backtest.
- Not a standalone LBO or valuation model.
- Uses deterministic structured scoring, with configurable weights in `config/scoring_config.yaml`.
- Separates screening scores from filing evidence retrieval.
- Includes a retrieval evaluation harness with no-answer checks and citation coverage.
- Generates recruiter-friendly outputs: Streamlit dashboard, Excel workbook, charts and memo reports.

## Data Sources

- `yfinance` is used for ticker metadata, market cap, price history, valuation-style fields, latest price, 52-week range, drawdown and realised volatility. It is an unofficial open-source library using Yahoo Finance publicly available interfaces and should be treated as research and educational data.
- SEC EDGAR endpoints are implemented for CIK mapping, submissions metadata and XBRL companyfacts. A compliant `SEC_USER_AGENT` is configurable in `.env.example` and `config/sec_config.yaml`.
- Cached sample mode is included for reproducibility. The sample mode provides representative public-company records, SEC-style fundamentals and filing chunks so the project works offline and during API failures.

## Screening Methodology

The structured pipeline calculates 22 metrics across:

- growth: latest revenue growth and 3-year revenue CAGR
- profitability: gross margin, EBIT margin proxy and net margin
- cash conversion: operating cash flow conversion, free cash flow proxy and FCF conversion
- capital intensity: capex intensity and cash / revenue
- leverage: debt / EBIT proxy and net debt / EBIT proxy
- valuation: EV / sales and EV / EBIT proxy
- market risk: drawdown from 52-week high and realised volatility
- trends and quality: margin trend, revenue trend, leverage trend and data completeness

EBIT is labelled as an EBIT proxy. The project does not silently call EBIT EBITDA.

## Scorecards

The scoring engine creates:

- Investment Screening Score
- Diligence Priority Score
- Platform Candidate Score
- Value Creation Potential Score
- Credit Risk Score
- Red Flag Score
- Data Quality Score

Every score is bounded from 0 to 100 and uses transparent weighted percentile ranks. The weights can be edited in `config/scoring_config.yaml`.

## Company Categories

Companies receive one primary category and multiple tags:

- High Priority for Further Diligence
- Quality Compounder
- Platform Candidate
- Value / Re-rating Candidate
- Operational Improvement Candidate
- Leveraged Credit Watchlist
- Distressed / Special Situations Watchlist
- Low Priority / Reject
- Insufficient Data

The system never outputs buy, sell or invest recommendations.

## SEC Filing Intelligence / RAG Layer

The filing layer:

- loads filing chunks with ticker, company, CIK, filing type, filing date, section and chunk ID metadata
- uses TF-IDF retrieval as the default dependable method
- returns top-k evidence chunks with source metadata
- flags weak retrieval as `insufficient filing evidence found`
- generates a source-backed evidence report and deterministic memo

The filing retrieval layer supports diligence questions on revenue drivers, customer concentration, liquidity, debt obligations, margin pressure, capex, competition, regulation, risk factors, cash flow and capital allocation.

## RAG Evaluation

The evaluation harness uses a labelled gold set with answerable and no-answer diligence questions. Relevance is checked through expected section labels and expected keywords. The cached run generated:

- 70 labelled evaluation questions
- 60 answerable questions
- Hit@3: 100.0%
- Hit@5: 100.0%
- Precision@3: 78.9%
- Precision@5: 71.3%
- MRR: 100.0%
- nDCG@5: 95.9%
- citation coverage: 100.0%
- unsupported-claim rate: 0.0%
- no-answer accuracy: 100.0%

## Dashboard Preview

Run:

```bash
streamlit run dashboard/app.py
```

The dashboard includes tabs for overview, sector screening, rankings, peer benchmarking, target deep dive, filing evidence, memo preview, RAG evaluation and methodology.

## Example Outputs

- `outputs/excel/private_markets_screening_outputs.xlsx`
- `outputs/reports/investment_screening_memo.md`
- `outputs/reports/filing_evidence_report.md`
- `outputs/reports/rag_evaluation_report.md`
- `outputs/charts/top_screened_companies.png`
- `outputs/charts/valuation_vs_growth.png`
- `outputs/charts/peer_percentile_radar.png`
- `outputs/charts/rag_eval_summary.png`

## Headline Results

- Screened 40 public companies across 8 sectors/themes in cached sample mode.
- Configured a 108-company default online universe.
- Calculated 22 financial, valuation, leverage, cash-conversion and risk metrics.
- Included SEC-style fundamentals for 40 companies.
- Indexed 280 filing chunks across 40 companies.
- Built explainable investment screening, diligence priority, platform candidate, value creation, credit risk, red flag and data quality scorecards.
- Categorised companies into 3 high-priority diligence candidates, 4 quality compounders, 1 platform candidate, 1 operational improvement candidate, 7 leveraged credit watchlist names, 18 value / re-rating candidates and 6 low-priority screens.
- Retrieved filing evidence for 80 company-question pairs.
- RAG evaluation achieved Hit@5 of 100.0%, Precision@5 of 71.3%, citation coverage of 100.0% and unsupported-claim rate of 0.0%.
- Generated dashboard, Excel workbook, charts and source-backed investment screening memo outputs.

## Project Structure

```text
config/       universe, SEC, scoring and retrieval settings
data/         sample, interim and processed datasets
dashboard/    Streamlit dashboard
eval/         gold questions and evaluation config
notebooks/    walkthrough notebook
outputs/      charts, Excel workbook and Markdown reports
src/          ingestion, scoring, retrieval, evaluation and reporting code
tests/        pytest coverage for core workflows
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full cached sample workflow:

```bash
python -m src.universe
python -m src.yfinance_client
python -m src.sec_client
python -m src.screening_pipeline
python -m src.filing_rag
python -m src.retrieval_eval
python -m src.reporting
pytest
streamlit run dashboard/app.py
```

Windows launcher fallback:

```powershell
py -m src.universe
py -m src.yfinance_client
py -m src.sec_client
py -m src.screening_pipeline
py -m src.filing_rag
py -m src.retrieval_eval
py -m src.reporting
py -m pytest
py -m streamlit run dashboard/app.py
```

Optional limited online smoke tests:

```bash
python -m src.yfinance_client --mode online --limit 10
python -m src.sec_client --mode online --limit 10
```

If online calls fail or return incomplete data, the clients fall back to cached sample mode.

## Tests

The test suite covers:

- universe loading
- yfinance fallback behavior
- SEC CIK mapping
- SEC XBRL mapping
- feature engineering calculations
- score bounds
- category rules
- peer benchmarking
- filing chunking
- retrieval metadata
- no-answer handling
- RAG evaluation metrics
- memo, chart, Excel and report outputs

Latest local run: `13 passed`.

## Data Limitations

- yfinance data is unofficial and should be treated as research and educational data.
- SEC EDGAR APIs provide official filing and fundamental data, but XBRL concepts vary by company and require careful mapping.
- Cached sample mode is designed for reproducibility and offline demonstration. It is not a substitute for a full live-data diligence process.
- Filing section parsing can be imperfect because SEC filing formats vary across issuers and years.
- The RAG layer retrieves evidence but does not guarantee legal, accounting or investment completeness.

## Investment Advice Disclaimer

This project is a public-data research and screening workflow for educational and portfolio demonstration purposes. It does not provide investment advice, buy/sell recommendations or analysis of private confidential deal materials. “High Priority for Further Diligence” means the company screened well against selected public-data criteria and should be reviewed further before any decision.

## Future Enhancements

- Increase the live online universe to 150 to 200 companies after validating ticker coverage.
- Add richer live SEC filing extraction for latest 10-K and 10-Q sections.
- Add issuer-specific XBRL tag diagnostics and confidence flags.
- Expand gold questions across more companies and filing sections.
- Add scenario filters for infrastructure, secondaries, co-investment and private credit use cases.
- Add persistence for historical refresh comparisons.

## CV Bullet

Detailed version:

**Private Markets Investment Screening & Filing Intelligence Dashboard | Python, pandas, SEC EDGAR, yfinance, Streamlit, RAG**

- Built a Python/Streamlit investment screening and filing intelligence tool to rank listed-company comparables across selectable sectors using public market data, SEC XBRL fundamentals and filing text. Created explainable investment screening, platform candidate, value creation, credit risk, red-flag and data-quality scorecards, peer benchmarking, target categorisation and source-backed diligence memo outputs using a RAG-style SEC filing retrieval layer.

Metrics version:

- Screened 40 public companies across 8 sectors, calculating 22 financial, valuation and risk metrics; generated target categories, peer benchmarking and source-backed filing evidence, with RAG evaluation achieving Hit@5 of 100.0%, Precision@5 of 71.3% and citation coverage of 100.0%.
