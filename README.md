# Private Markets Investment Screening & Filing Intelligence Dashboard

## Overview

This project is a public-data private markets screening and SEC filing intelligence workflow built with Python, pandas, yfinance, SEC EDGAR APIs, scikit-learn TF-IDF retrieval and Streamlit.

It screens listed-company comparables for further diligence. It does not make buy/sell calls, does not use private confidential deal materials and does not present an acquisition recommendation.

## Why I Built This

Private equity, co-investment, secondaries, infrastructure, real assets and private credit teams often start with public comparables before moving into proprietary diligence. This project mirrors that early workflow:

1. Define a sector or theme universe.
2. Pull public market data and SEC fundamentals.
3. Calculate growth, margin, leverage, valuation, cash-conversion and risk metrics.
4. Separate public-market quality from private-markets feasibility.
5. Rank companies using transparent scorecards.
6. Retrieve filing evidence for diligence questions.
7. Generate a memo, evidence report, evaluation report, Excel workbook, charts and dashboard.

## What Makes This Distinct

- It is not a stock backtest.
- It is not a generic stock screener.
- It is not a standalone LBO model.
- It separates "excellent public company" from "realistic private-markets diligence candidate."
- It includes live API mode, cached real-data snapshot mode and sample fallback mode.
- It evaluates the filing retrieval layer with labelled questions, no-answer checks and weak retrieval examples.

## Data Modes

### Live API Mode

The online pipeline attempts to refresh:

- yfinance market data, price history, valuation fields and volatility
- SEC CIK mapping
- SEC submissions metadata
- SEC XBRL companyfacts
- latest 10-K filing text for a selected subset

### Cached Real-Data Snapshot Mode

The committed outputs include a cached public-data snapshot generated from live API calls on May 28, 2026. This is the dashboard default for recruiter review because it is reproducible and does not depend on API availability at viewing time.

### Sample Fallback Mode

Sample fallback data remains available so the repo still runs if Yahoo Finance or SEC endpoints are unavailable. Rows and chunks include source labels, so fallback usage is visible.

## Data Sources

- yfinance provides unofficial Yahoo Finance market data. It should be treated as research and educational data.
- SEC EDGAR provides official filing metadata and companyfacts data, but XBRL tags vary by issuer and require mapping.
- SEC filing text is retrieved from public filing documents where available.
- `.env.example` includes a configurable SEC user agent.

## Screening Methodology

The current pipeline calculates 27 structured metrics, including:

- revenue growth and 3-year revenue CAGR
- gross margin, EBIT margin proxy and net margin
- operating cash flow conversion and free cash flow proxy
- capex intensity and cash / revenue
- debt / EBIT proxy and net debt / EBIT proxy
- EV / sales and EV / EBIT proxy
- 52-week drawdown and realised volatility
- margin, revenue and leverage trend
- yfinance and SEC field completeness
- filing staleness and proxy-usage penalties
- market-cap / enterprise-value feasibility band

EBIT is labelled as an EBIT proxy. The project does not silently call EBIT EBITDA.

## Scorecards

The scorecards are deterministic and configurable:

- Public Quality Score
- Investment Screening Score
- Platform Candidate Score
- Public-to-Private Feasibility Score
- Value Creation Potential Score
- Credit Risk Score
- Red Flag Score
- Data Quality Score

Weights are stored in `config/scoring_config.yaml`.

## Company Categories

The taxonomy is designed for diligence triage, not transaction recommendations:

- Public Market Quality Compounder
- PE Platform Candidate
- Public-to-Private Candidate
- Value / Re-rating Candidate
- Operational Improvement Candidate
- Leveraged Credit Watchlist
- Distressed / Special Situations Watchlist
- Low Priority / Reject
- Insufficient Data

## Why MSFT/NOW/INTU May Rank Highly

Mega-cap companies can score well on public quality because they often have strong growth, margins, cash conversion, data coverage and filing quality. That does not make them realistic PE targets.

The dashboard separates:

- Public Quality Score: useful for benchmark quality comps.
- Platform Candidate Score: penalises enterprise values that are too large for most PE platform strategies.
- Public-to-Private Feasibility Score: rewards more realistic EV bands and penalises mega-cap scale.
- Market-cap band: labels mega-cap benchmark comps, large-cap strategic names, mid-cap public-to-private screens and small/mid-cap platform candidates.

## Screening Output Is Not an Investment Recommendation

The project uses language such as:

- benchmark quality comp
- high priority for further diligence
- public-to-private feasibility
- platform candidate
- watchlist

It does not use buy/sell/invest recommendation language.

## SEC Filing Intelligence / RAG Layer

The filing layer:

- downloads and parses real SEC filing text for a selected subset
- keeps fallback filing chunks for uncovered companies
- labels every chunk with `source_type`
- uses TF-IDF retrieval by default
- returns ticker, company, CIK, filing type, filing date, section label, chunk ID and evidence score
- flags missing evidence as `insufficient filing evidence found`

## RAG Evaluation

The gold set includes harder, less templated questions:

- multi-section liquidity and debt questions
- margin pressure questions that can appear in MD&A or Risk Factors
- competition / regulatory section-confuser questions
- no-answer questions designed to test refusal to overclaim

Current cached snapshot results:

- Questions evaluated: 96
- Answerable questions: 72
- Hit@3: 100.0%
- Hit@5: 100.0%
- Precision@3: 86.6%
- Precision@5: 82.5%
- MRR: 99.3%
- nDCG@5: 96.7%
- Section match rate: 95.8%
- Citation coverage: 100.0%
- Unsupported-claim rate: 0.0%
- No-answer accuracy: 100.0%

These results are strong, but not proof of legal or investment completeness. The evaluation checks retrieval behavior on a labelled public-data test set.

## Headline Results

Current cached public-data snapshot:

- Live companies attempted: 108
- Companies successfully screened: 107
- Companies with yfinance market data: 107
- Companies with SEC companyfacts rows: 106
- Companies with usable revenue fundamentals: 101
- Sectors/themes covered: 9
- Metrics calculated: 27
- Real SEC filing documents parsed: 20
- Filing chunks indexed: 11,360
- Real SEC filing chunks: 11,115
- Sample fallback chunks: 245
- Average Data Quality Score: 85.2
- Skipped ticker records: FYBR yfinance/companyfacts, TRI companyfacts/10-K text

Top private-markets-style Investment Screening Score names include HUBS, FSLR, DPZ, TEAM, CNC, ZS, LYV, IR, XYZ and AWK.

Top Public Quality Score names include CRM, MSFT, SBAC, FSLR, ADBE, NOW, COF, DDOG, AMT and GE.

Category mix:

- Value / Re-rating Candidate: 53
- Low Priority / Reject: 20
- Public-to-Private Candidate: 10
- Leveraged Credit Watchlist: 8
- Operational Improvement Candidate: 6
- PE Platform Candidate: 4
- Public Market Quality Compounder: 3
- Distressed / Special Situations Watchlist: 2
- Insufficient Data: 1

## Dashboard Preview

Run:

```bash
streamlit run dashboard/app.py
```

Screenshots are saved in `outputs/screenshots/`:

- `overview.png`
- `rankings.png`
- `target_deep_dive.png`
- `rag_evidence.png`
- `rag_eval.png`

## Example Outputs

- `outputs/excel/private_markets_screening_outputs.xlsx`
- `outputs/reports/investment_screening_memo.md`
- `outputs/reports/filing_evidence_report.md`
- `outputs/reports/rag_evaluation_report.md`
- `outputs/charts/top_screened_companies.png`
- `outputs/charts/valuation_vs_growth.png`
- `outputs/charts/peer_percentile_radar.png`
- `outputs/charts/rag_eval_summary.png`

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default online-refresh workflow with fallback:

```bash
python -m src.universe
python -m src.yfinance_client
python -m src.sec_client
python -m src.screening_pipeline
python -m src.filing_parser --mode online --limit 20
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
py -m src.filing_parser --mode online --limit 20
py -m src.filing_rag
py -m src.retrieval_eval
py -m src.reporting
py -m pytest
py -m streamlit run dashboard/app.py
```

Sample-only fallback:

```bash
python -m src.screening_pipeline --mode sample
python -m src.filing_parser --mode sample
python -m src.filing_rag
python -m src.retrieval_eval
python -m src.reporting
```

## Tests

The test suite covers:

- universe loading
- yfinance fallback behavior
- SEC CIK mapping
- SEC XBRL mapping
- feature engineering
- score bounds
- market-cap-band category rules
- mega-cap quality comp handling
- data quality penalties
- skipped ticker logging
- source-type coverage
- filing chunking
- retrieval metadata
- no-answer handling
- RAG evaluation
- memo, chart, Excel and report outputs
- dashboard module compilation

Latest local run: `18 passed`.

## Data Limitations

- yfinance is unofficial and may fail, change fields or return stale values.
- SEC XBRL concepts vary by company and some fields may be missing or mapped imperfectly.
- EBIT proxy is used when EBITDA is unavailable.
- Filing section parsing is imperfect because SEC filing formats vary.
- Retrieval evidence is not a legal, accounting or investment diligence substitute.
- Cached outputs are reproducible snapshots, not continuously refreshed monitoring.

## Future Enhancements

- Add more robust item-level SEC filing section extraction.
- Expand real filing parsing beyond 20 issuers.
- Add issuer-level XBRL mapping confidence diagnostics.
- Expand the gold set with human-labelled filing passages.
- Add historical refresh comparison and portfolio monitoring mode.
- Add role-specific workflows for co-investments, secondaries, infrastructure and private credit.

## CV Bullet

Detailed version:

**Private Markets Investment Screening & Filing Intelligence Dashboard | Python, pandas, SEC EDGAR, yfinance, Streamlit, RAG**

- Built a Python/Streamlit public-data investment screening and filing intelligence workflow to rank listed-company comparables across selectable sectors using market data, SEC companyfacts and filing text. Created explainable public-quality, investment-screening, platform-candidate, public-to-private-feasibility, value-creation, credit-risk, red-flag and data-quality scorecards, with peer benchmarking, target categorisation and source-backed diligence memo outputs.

Metrics version:

- Screened 107 public companies across 9 sectors, calculating 27 financial, valuation and risk metrics; parsed 20 real SEC filing documents into 11,115 real filing chunks; generated target categories, peer benchmarking and source-backed filing evidence, with retrieval evaluation achieving Hit@5 of 100.0%, Precision@5 of 82.5% and citation coverage of 100.0%.
