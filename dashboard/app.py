from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHART_DIR, INTERIM_DIR, PROCESSED_DIR, REPORT_DIR, SAMPLE_DIR
from src.filing_rag import FilingRetriever, load_filing_chunks
from src.rag_generator import RAGAnswerGenerator
from src.reporting import summary_metrics
from src.screening_pipeline import run_pipeline
from src.utils import format_pct, read_csv_if_exists


st.set_page_config(
    page_title="Private Markets Screening Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
  .block-container {padding-top: 1.35rem; padding-bottom: 2rem; max-width: 1420px;}
  h1 {font-size: 1.45rem !important; letter-spacing: 0 !important; margin-bottom: 0.15rem;}
  h2 {font-size: 1.05rem !important; margin-top: 1.1rem;}
  h3 {font-size: 0.95rem !important;}
  [data-testid="stSidebar"] {background: #f6f8fb; border-right: 1px solid #dde3ea;}
  [data-testid="stSidebar"] h2 {font-size: 0.95rem !important;}
  div[data-testid="stMetric"] {
      background: #ffffff; border: 1px solid #dbe3ec; border-radius: 8px;
      padding: 0.65rem 0.75rem; box-shadow: 0 1px 2px rgba(20, 31, 44, 0.04);
  }
  div[data-testid="stMetricLabel"] {font-size: 0.72rem;}
  div[data-testid="stMetricValue"] {font-size: 1.15rem;}
  .badge {
      display: inline-block; padding: 0.18rem 0.48rem; border-radius: 999px;
      border: 1px solid #c9d6e2; background: #eef4f8; color: #213547;
      font-size: 0.74rem; margin-right: 0.35rem; margin-bottom: 0.3rem;
  }
  .badge-live {background: #e8f5ed; border-color: #b8dec6; color: #1d5e36;}
  .badge-warn {background: #fff6df; border-color: #ead39a; color: #7a5614;}
  .evidence-card {
      border: 1px solid #d9e2ea; border-radius: 8px; padding: 0.75rem;
      background: #ffffff; margin-bottom: 0.65rem;
  }
  .muted {color: #66788a; font-size: 0.82rem;}
  .small-label {font-size: 0.75rem; color: #5a6b7b; text-transform: uppercase;}
  .stTabs [data-baseweb="tab-list"] {gap: 0.15rem;}
  .stTabs [data-baseweb="tab"] {height: 2.1rem; padding: 0 0.65rem; font-size: 0.82rem;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
    if scores.empty:
        run_pipeline(mode="sample")
        scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
    return {
        "scores": scores,
        "categories": read_csv_if_exists(PROCESSED_DIR / "company_categories.csv"),
        "benchmarks": read_csv_if_exists(PROCESSED_DIR / "peer_benchmarks.csv"),
        "evidence": read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv"),
        "questions": read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv"),
        "generated": read_csv_if_exists(PROCESSED_DIR / "generated_rag_answers.csv"),
        "eval": read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv"),
        "grounded_eval": read_csv_if_exists(PROCESSED_DIR / "grounded_generation_eval.csv"),
        "chunks": read_csv_if_exists(INTERIM_DIR / "filing_chunks.csv"),
        "skipped": read_csv_if_exists(PROCESSED_DIR / "skipped_tickers.csv"),
        "real_filings": read_csv_if_exists(PROCESSED_DIR / "real_filing_documents.csv"),
        "sample_universe": read_csv_if_exists(SAMPLE_DIR / "sample_universe.csv"),
    }


def metric_value(frame: pd.DataFrame, name: str) -> float:
    row = frame[frame["metric"] == name] if not frame.empty and "metric" in frame.columns else pd.DataFrame()
    return float(row["value"].iloc[0]) if not row.empty else 0.0


def compact_money(value: float | int | None) -> str:
    if pd.isna(value):
        return "n/a"
    value = float(value)
    return f"${value / 1_000_000_000:,.1f}bn"


def score_bar(label: str, value: float) -> None:
    st.caption(label)
    st.progress(max(0, min(100, int(round(float(value or 0))))))
    st.markdown(f"<span class='muted'>{float(value or 0):.1f} / 100</span>", unsafe_allow_html=True)


def evidence_cards(frame: pd.DataFrame, limit: int = 5) -> None:
    if frame.empty:
        st.warning("insufficient filing evidence found")
        return
    for _, row in frame.head(limit).iterrows():
        source_badge = "badge-live" if row.get("source_type") == "real_sec_filing" else "badge-warn"
        st.markdown(
            f"""
<div class="evidence-card">
  <div>
    <span class="badge {source_badge}">{row.get('source_type', 'unknown')}</span>
    <span class="badge">{row.get('filing_type', '')} | {row.get('filing_date', '')}</span>
    <span class="badge">{row.get('section_label', '')}</span>
    <span class="badge">{row.get('chunk_id', '')}</span>
  </div>
  <div class="muted">rank {int(row.get('rank', 0) or 0)} | keyword {float(row.get('keyword_score', 0) or 0):.3f}
  | section boost {float(row.get('section_boost', 0) or 0):.3f} | combined {float(row.get('combined_score', 0) or 0):.3f}</div>
  <p>{str(row.get('evidence_snippet', row.get('text', '')))[:900]}</p>
</div>
""",
            unsafe_allow_html=True,
        )
        if row.get("source_url"):
            st.link_button("Open source filing", str(row["source_url"]))


data = load_data()
scores = data["scores"]
categories = data["categories"]
benchmarks = data["benchmarks"]
eval_results = data["eval"]
grounded_eval = data["grounded_eval"]
chunks = data["chunks"]
skipped = data["skipped"]
real_filings = data["real_filings"]

summary = summary_metrics()
latest_refresh = scores["metrics_refresh_timestamp"].max() if "metrics_refresh_timestamp" in scores.columns and not scores.empty else "n/a"
real_chunk_count = int((chunks.get("source_type", pd.Series(dtype=str)) == "real_sec_filing").sum()) if not chunks.empty else 0
fallback_chunk_count = int((chunks.get("source_type", pd.Series(dtype=str)) == "sample_fallback").sum()) if not chunks.empty else 0
live_rows = int((scores.get("source_type", pd.Series(index=scores.index, dtype=str)) == "live_public_data").sum()) if not scores.empty else 0
data_badge = "Live/Cached Snapshot" if live_rows > len(scores) * 0.5 else "Sample Fallback"

st.markdown("### Private Markets Investment Screening & Filing Intelligence Dashboard")
st.markdown(
    "<span class='muted'>Public-data screening, peer benchmarking and source-backed filing intelligence for further diligence workflows. Not investment advice.</span>",
    unsafe_allow_html=True,
)
badge_class = "badge-live" if data_badge != "Sample Fallback" else "badge-warn"
st.markdown(
    f"""
<div style="margin-top: .55rem;">
  <span class="badge {badge_class}">{data_badge}</span>
  <span class="badge">Refresh: {latest_refresh}</span>
  <span class="badge">Coverage: {len(scores)} companies | {summary.get('sectors_covered', 0)} sectors</span>
  <span class="badge">Real SEC chunks: {real_chunk_count:,}</span>
</div>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Controls")
    data_mode = st.radio("Data mode", ["cached snapshot", "sample fallback", "live refresh"], index=0)
    sector_options = ["All"] + sorted(scores["sector_theme"].dropna().unique().tolist())
    sector = st.selectbox("Sector / theme", sector_options)
    band_options = ["All"] + sorted(scores["market_cap_band"].dropna().unique().tolist()) if "market_cap_band" in scores else ["All"]
    market_cap_band = st.selectbox("Market-cap band", band_options)
    min_market_cap = st.number_input("Min market cap ($bn)", min_value=0.0, value=0.0, step=1.0)
    max_market_cap = st.number_input("Max market cap ($bn)", min_value=0.0, value=5000.0, step=5.0)
    max_leverage = st.number_input("Max debt / EBIT proxy", min_value=0.0, value=8.0, step=0.25)
    min_growth = st.number_input("Min revenue growth", min_value=-1.0, max_value=1.0, value=-0.10, step=0.01)
    valuation_range = st.slider("EV / sales", 0.0, 30.0, (0.0, 15.0), step=0.5)
    archetypes = ["All"] + sorted(categories["primary_category"].dropna().unique().tolist()) if not categories.empty else ["All"]
    strategy = st.selectbox("Strategy archetype", archetypes)
    top_n = st.slider("Top N", 5, 40, 12)
    selected_company = st.selectbox("Selected company", scores.sort_values("investment_screening_score", ascending=False)["ticker"].tolist())
    question_options = data["questions"]["question"].drop_duplicates().tolist() if not data["questions"].empty else ["What are the company's main revenue drivers?"]
    selected_question = st.selectbox("Retrieval question", question_options)
    top_k = st.slider("Retrieval top-k", 3, 10, 5)
    retrieval_mode = st.selectbox("Retrieval mode", ["keyword", "hybrid", "semantic if available"], index=0)
    llm_mode = st.radio("LLM mode", ["off", "local Ollama"], index=0)
    model = st.selectbox("Ollama model", ["qwen3", "qwen2.5", "llama3.1", "llama3.2", "mistral"], index=0)
    temperature = st.slider("Temperature", 0.0, 0.8, 0.0, step=0.05)
    if st.button("Refresh sample snapshot"):
        run_pipeline(mode="sample")
        st.cache_data.clear()
        st.rerun()
    if st.button("Run live refresh"):
        run_pipeline(mode="online", runtime_mode="demo")
        st.cache_data.clear()
        st.rerun()

filtered = scores.copy()
if sector != "All":
    filtered = filtered[filtered["sector_theme"] == sector]
if market_cap_band != "All" and "market_cap_band" in filtered:
    filtered = filtered[filtered["market_cap_band"] == market_cap_band]
filtered = filtered[
    (filtered["market_cap"] >= min_market_cap * 1_000_000_000)
    & (filtered["market_cap"] <= max_market_cap * 1_000_000_000)
    & (filtered["debt_to_ebit_proxy"].isna() | (filtered["debt_to_ebit_proxy"] <= max_leverage))
    & (filtered["revenue_growth_yoy"] >= min_growth)
    & (filtered["ev_to_sales"].between(valuation_range[0], valuation_range[1]))
]
if strategy != "All" and not categories.empty:
    filtered = filtered[filtered["ticker"].isin(categories[categories["primary_category"] == strategy]["ticker"])]

overview, rankings, comps, target, filings, llm_tab, memo, eval_tab, methodology = st.tabs(
    ["Overview", "Rankings", "Comps", "Target", "Filings", "LLM RAG", "Memo", "Eval", "Methodology"]
)

with overview:
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Companies", summary.get("companies_screened", 0))
    kpi_cols[1].metric("Sectors", summary.get("sectors_covered", 0))
    kpi_cols[2].metric("SEC coverage", summary.get("companies_with_sec_fundamentals", 0))
    kpi_cols[3].metric("Real filings", summary.get("real_filing_documents_parsed", 0))
    kpi_cols[4].metric("Chunks", f"{summary.get('real_filing_chunks', 0):,}")
    kpi_cols = st.columns(5)
    kpi_cols[0].metric("Avg data quality", f"{summary.get('average_data_quality_score', 0):.1f}")
    kpi_cols[1].metric("Top screen", f"{scores['investment_screening_score'].max():.1f}")
    kpi_cols[2].metric("RAG Hit@5", format_pct(metric_value(eval_results, "hit_rate_at_5")))
    kpi_cols[3].metric("Precision@5", format_pct(metric_value(eval_results, "precision_at_5")))
    kpi_cols[4].metric("Unsupported", format_pct(metric_value(eval_results, "unsupported_claim_rate")))
    if fallback_chunk_count:
        st.warning(f"{fallback_chunk_count:,} fallback filing chunks are retained for offline reproducibility and labelled by source_type.")
    chart_cols = st.columns([1.15, 1])
    with chart_cols[0]:
        st.plotly_chart(px.histogram(scores, x="investment_screening_score", color="sector_theme", nbins=20, template="plotly_white"), use_container_width=True)
    with chart_cols[1]:
        if not categories.empty:
            counts = categories["primary_category"].value_counts().reset_index()
            st.plotly_chart(px.bar(counts, x="count", y="primary_category", orientation="h", template="plotly_white"), use_container_width=True)
    with st.expander("Skipped tickers and coverage notes", expanded=False):
        if skipped.empty:
            st.write("No skipped ticker records.")
        else:
            st.dataframe(skipped, use_container_width=True, hide_index=True)

with rankings:
    cols = [
        "ticker",
        "company_name",
        "sector_theme",
        "market_cap_band",
        "market_cap",
        "public_quality_score",
        "investment_screening_score",
        "platform_candidate_score",
        "public_to_private_feasibility_score",
        "value_creation_potential_score",
        "credit_risk_score",
        "red_flag_score",
        "data_quality_score",
    ]
    st.dataframe(filtered.sort_values("investment_screening_score", ascending=False).head(top_n)[cols], use_container_width=True, hide_index=True)
    st.plotly_chart(px.scatter(filtered, x="ev_to_sales", y="revenue_growth_yoy", size="market_cap", color="sector_theme", hover_name="ticker", template="plotly_white"), use_container_width=True)

with comps:
    st.plotly_chart(px.scatter(filtered, x="ev_to_sales", y="operating_margin", color="market_cap_band", hover_name="ticker", template="plotly_white"), use_container_width=True)
    selected_peer = benchmarks[benchmarks["ticker"] == selected_company]
    if not selected_peer.empty:
        radar_cols = [
            "revenue_growth_yoy_peer_percentile",
            "operating_margin_peer_percentile",
            "fcf_conversion_peer_percentile",
            "debt_to_ebit_proxy_peer_percentile",
            "ev_to_sales_peer_percentile",
            "realised_volatility_peer_percentile",
        ]
        radar = selected_peer[radar_cols].T.reset_index()
        radar.columns = ["metric", "percentile"]
        st.plotly_chart(px.line_polar(radar, r="percentile", theta="metric", line_close=True, range_r=[0, 100], template="plotly_white"), use_container_width=True)
        st.dataframe(selected_peer.T, use_container_width=True)

with target:
    company = scores[scores["ticker"] == selected_company].iloc[0]
    category = categories[categories["ticker"] == selected_company].iloc[0] if not categories.empty and selected_company in set(categories["ticker"]) else None
    st.markdown(f"## {company['company_name']} ({company['ticker']})")
    st.markdown(f"<span class='badge'>{company.get('sector_theme', '')}</span><span class='badge'>{company.get('market_cap_band', '')}</span><span class='badge'>{compact_money(company.get('market_cap'))} market cap</span>", unsafe_allow_html=True)
    score_cols = st.columns(4)
    with score_cols[0]:
        score_bar("Public Quality", company.get("public_quality_score", 0))
    with score_cols[1]:
        score_bar("Investment Screen", company.get("investment_screening_score", 0))
    with score_cols[2]:
        score_bar("Public-to-Private", company.get("public_to_private_feasibility_score", 0))
    with score_cols[3]:
        score_bar("Red Flag", company.get("red_flag_score", 0))
    if category is not None:
        st.info(category["why_this_category"])
    st.write(company.get("investment_screening_explanation", ""))
    metrics_cols = st.columns(5)
    metrics_cols[0].metric("Revenue growth", format_pct(company.get("revenue_growth_yoy")))
    metrics_cols[1].metric("EBIT margin proxy", format_pct(company.get("operating_margin")))
    metrics_cols[2].metric("FCF conversion", format_pct(company.get("fcf_conversion")))
    metrics_cols[3].metric("EV / sales", f"{company.get('ev_to_sales', 0):.1f}x")
    metrics_cols[4].metric("Drawdown", format_pct(company.get("drawdown_from_52w_high")))
    decomposition = pd.DataFrame(
        {
            "score": ["Public Quality", "Platform", "Public-to-Private", "Value Creation", "Credit Risk", "Red Flag", "Data Quality"],
            "value": [
                company.get("public_quality_score", 0),
                company.get("platform_candidate_score", 0),
                company.get("public_to_private_feasibility_score", 0),
                company.get("value_creation_potential_score", 0),
                company.get("credit_risk_score", 0),
                company.get("red_flag_score", 0),
                company.get("data_quality_score", 0),
            ],
        }
    )
    st.plotly_chart(px.bar(decomposition, x="score", y="value", range_y=[0, 100], template="plotly_white"), use_container_width=True)
    st.caption(company.get("data_quality_explanation", ""))

with filings:
    filing_chunks = load_filing_chunks()
    retriever = FilingRetriever(filing_chunks, mode=retrieval_mode)
    result = retriever.search(selected_question, ticker=selected_company, top_k=top_k)
    if not result.empty:
        result["rank"] = range(1, len(result) + 1)
    st.markdown("#### Retrieved Filing Evidence")
    st.caption("Section boost is transparent and query-intent based; it helps revenue questions favour Business, MD&A and Segment sections when relevant.")
    if not result.empty and not bool(result.iloc[0].get("section_match_flag", False)):
        st.warning("The top result is an adjacent section rather than the preferred section for this question type.")
    evidence_cards(result, limit=top_k)
    table_cols = [
        "rank",
        "query",
        "expected_question_type",
        "ticker",
        "company_name",
        "source_type",
        "filing_type",
        "filing_date",
        "section_label",
        "chunk_id",
        "keyword_score",
        "semantic_score",
        "section_boost",
        "combined_score",
        "section_match_flag",
        "evidence_strength",
        "source_url",
    ]
    if not result.empty:
        st.dataframe(result[[col for col in table_cols if col in result.columns]], use_container_width=True, hide_index=True)

with llm_tab:
    st.markdown("#### LLM RAG Summary")
    st.caption("LLM summaries are generated only from retrieved SEC filing evidence and are for workflow demonstration. They are not investment advice and should be reviewed against source filings.")
    filing_chunks = load_filing_chunks()
    retriever = FilingRetriever(filing_chunks, mode=retrieval_mode)
    result = retriever.search(selected_question, ticker=selected_company, top_k=top_k)
    if not result.empty:
        result["rank"] = range(1, len(result) + 1)
    evidence_cards(result, limit=min(top_k, 3))
    if st.button("Generate source-grounded answer"):
        company = scores[scores["ticker"] == selected_company].iloc[0]
        generator = RAGAnswerGenerator(llm_enabled=llm_mode == "local Ollama", model_name=model, temperature=temperature)
        answer = generator.generate_answer(selected_company, str(company["company_name"]), selected_question, result)
        st.markdown(f"**Generation mode:** {answer['generation_mode']}  ")
        st.markdown(f"**Model:** {answer['model_name'] or 'not used'}")
        if answer["llm_warning"]:
            st.warning(answer["llm_warning"])
        st.markdown(answer["answer"])
        st.json(
            {
                "cited_chunks": answer["cited_chunks"],
                "evidence_strength": answer["evidence_strength"],
                "citation_coverage": answer["citation_coverage"],
                "evidence_overlap_score": answer["evidence_overlap_score"],
                "banned_language_count": answer["banned_language_count"],
                "unsupported_claim_warning": answer["unsupported_claim_warning"],
                "no_answer_flag": answer["no_answer_flag"],
            }
        )
    generated = data["generated"]
    if not generated.empty:
        st.markdown("#### Precomputed Generated Answers")
        st.dataframe(generated[generated["ticker"] == selected_company].head(20), use_container_width=True, hide_index=True)

with memo:
    memo_path = REPORT_DIR / "investment_screening_memo.md"
    if memo_path.exists():
        st.markdown(memo_path.read_text(encoding="utf-8"))
    else:
        st.info("Run py -m src.reporting to generate the memo preview.")

with eval_tab:
    st.caption("Retrieval and generation are evaluated separately. Hit@5 should be read together with Precision@5 and section-match examples.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Retrieval")
        st.dataframe(eval_results, use_container_width=True, hide_index=True)
        if not eval_results.empty:
            st.plotly_chart(px.bar(eval_results, x="metric", y="value", color="metric_type", template="plotly_white"), use_container_width=True)
    with c2:
        st.markdown("#### Grounded Generation")
        st.dataframe(grounded_eval, use_container_width=True, hide_index=True)
        if not grounded_eval.empty:
            st.plotly_chart(px.bar(grounded_eval, x="metric", y="value", color="metric_type", template="plotly_white"), use_container_width=True)

with methodology:
    st.markdown(
        """
The structured scorecards are deterministic percentile-based screens using public market data,
SEC companyfacts and peer-relative metrics. Filing intelligence retrieves source passages and
metadata from real SEC filing chunks where available, with fallback rows clearly labelled.
Optional local language-model summaries use only retrieved evidence and are evaluated for
citations, unsupported wording and recommendation-style language.

yfinance is an unofficial open-source library using Yahoo Finance publicly available interfaces
and should be treated as research and educational data. SEC EDGAR APIs provide official filing
and fundamental data, but XBRL tags vary by company and require careful mapping. This dashboard
is for public-data screening workflow demonstration and is not investment advice.
"""
    )

if data_mode == "live refresh":
    st.sidebar.info("Live refresh can take several minutes and depends on Yahoo Finance and SEC availability. Cached outputs remain usable if refresh coverage is incomplete.")
