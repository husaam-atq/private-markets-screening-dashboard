from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import CHART_DIR, PROCESSED_DIR, REPORT_DIR, SAMPLE_DIR
from src.filing_rag import FilingRetriever, load_filing_chunks
from src.reporting import summary_metrics
from src.screening_pipeline import run_pipeline
from src.utils import format_pct, read_csv_if_exists


st.set_page_config(
    page_title="Private Markets Investment Screening & Filing Intelligence Dashboard",
    layout="wide",
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
        "screening": read_csv_if_exists(PROCESSED_DIR / "screening_universe.csv"),
        "evidence": read_csv_if_exists(PROCESSED_DIR / "retrieved_evidence.csv"),
        "questions": read_csv_if_exists(PROCESSED_DIR / "diligence_questions.csv"),
        "eval": read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv"),
        "target": read_csv_if_exists(PROCESSED_DIR / "target_deep_dive.csv"),
        "sample_universe": read_csv_if_exists(SAMPLE_DIR / "sample_universe.csv"),
    }


def metric_value(eval_frame: pd.DataFrame, name: str) -> float:
    row = eval_frame[eval_frame["metric"] == name]
    return float(row["value"].iloc[0]) if not row.empty else 0.0


data = load_data()
scores = data["scores"]
categories = data["categories"]
benchmarks = data["benchmarks"]
evidence = data["evidence"]
questions = data["questions"]
eval_results = data["eval"]

st.title("Private Markets Investment Screening & Filing Intelligence Dashboard")
st.caption(
    "This dashboard uses public market data, SEC filings and cached sample data for investment screening workflow demonstration. "
    "It does not provide investment advice, buy/sell recommendations or analysis of private confidential deal materials."
)

with st.sidebar:
    st.header("Controls")
    data_mode = st.radio("Data mode", ["cached sample", "online API"], index=0)
    sector_options = ["All"] + sorted(scores["sector_theme"].dropna().unique().tolist())
    sector = st.selectbox("Sector / theme", sector_options)
    min_market_cap = st.number_input("Minimum market cap ($bn)", min_value=0.0, value=0.0, step=1.0)
    max_leverage = st.number_input("Maximum debt / EBIT proxy", min_value=0.0, value=8.0, step=0.25)
    min_growth = st.number_input("Minimum revenue growth", min_value=-1.0, max_value=1.0, value=-0.10, step=0.01)
    valuation_range = st.slider("EV / sales range", 0.0, 25.0, (0.0, 15.0), step=0.5)
    archetypes = ["All"] + sorted(categories["primary_category"].dropna().unique().tolist()) if not categories.empty else ["All"]
    strategy = st.selectbox("Strategy archetype", archetypes)
    top_n = st.slider("Number of companies to display", 5, 30, 10)
    selected_company = st.selectbox(
        "Selected company",
        scores.sort_values("investment_screening_score", ascending=False)["ticker"].tolist(),
    )
    top_k = st.slider("Top-k retrieval", 3, 10, 5)
    retrieval_mode = st.selectbox("Retrieval mode", ["keyword", "hybrid", "semantic if available"], index=0)
    if st.button("Refresh cached sample"):
        run_pipeline(mode="sample")
        st.cache_data.clear()
        st.rerun()

filtered = scores.copy()
if sector != "All":
    filtered = filtered[filtered["sector_theme"] == sector]
filtered = filtered[
    (filtered["market_cap"] >= min_market_cap * 1_000_000_000)
    & (filtered["debt_to_ebit_proxy"] <= max_leverage)
    & (filtered["revenue_growth_yoy"] >= min_growth)
    & (filtered["ev_to_sales"].between(valuation_range[0], valuation_range[1]))
]
if strategy != "All" and not categories.empty:
    keep = categories[categories["primary_category"] == strategy]["ticker"]
    filtered = filtered[filtered["ticker"].isin(keep)]

summary = summary_metrics()
kpis = [
    ("Companies screened", summary["companies_screened"]),
    ("Sectors covered", summary["sectors_covered"]),
    ("Top screening score", f"{scores['investment_screening_score'].max():.1f}"),
    ("Avg data quality", f"{summary['average_data_quality_score']:.1f}"),
    ("RAG Hit@5", format_pct(metric_value(eval_results, "hit_rate_at_5"))),
    ("RAG Precision@5", format_pct(metric_value(eval_results, "precision_at_5"))),
    ("Citation coverage", format_pct(metric_value(eval_results, "citation_coverage"))),
    ("Unsupported claim rate", format_pct(metric_value(eval_results, "unsupported_claim_rate"))),
]

overview, sector_tab, rankings, peer_tab, deep_dive, rag_tab, memo_tab, eval_tab, methodology = st.tabs(
    [
        "Overview",
        "Sector Screening",
        "Company Rankings",
        "Peer Benchmarking",
        "Target Deep Dive",
        "Filing Intelligence / RAG Evidence",
        "Diligence Memo",
        "RAG Evaluation",
        "Methodology & Limitations",
    ]
)

with overview:
    cols = st.columns(4)
    for idx, (label, value) in enumerate(kpis[:4]):
        cols[idx % 4].metric(label, value)
    cols = st.columns(4)
    for idx, (label, value) in enumerate(kpis[4:]):
        cols[idx % 4].metric(label, value)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.histogram(scores, x="investment_screening_score", color="sector_theme", nbins=18), use_container_width=True)
    with c2:
        if not categories.empty:
            st.plotly_chart(px.bar(categories["primary_category"].value_counts().reset_index(), x="count", y="primary_category", orientation="h"), use_container_width=True)

with sector_tab:
    st.dataframe(
        filtered.sort_values("investment_screening_score", ascending=False).head(top_n)[
            [
                "ticker",
                "company_name",
                "sector_theme",
                "market_cap",
                "revenue_growth_yoy",
                "operating_margin",
                "ev_to_sales",
                "debt_to_ebit_proxy",
                "investment_screening_score",
                "red_flag_score",
            ]
        ],
        use_container_width=True,
    )
    st.plotly_chart(px.scatter(filtered, x="ev_to_sales", y="revenue_growth_yoy", size="market_cap", color="sector_theme", hover_name="ticker"), use_container_width=True)

with rankings:
    st.dataframe(
        scores.sort_values("investment_screening_score", ascending=False).head(top_n)[
            [
                "ticker",
                "company_name",
                "sector_theme",
                "investment_screening_score",
                "diligence_priority_score",
                "platform_candidate_score",
                "value_creation_potential_score",
                "credit_risk_score",
                "red_flag_score",
                "data_quality_score",
            ]
        ],
        use_container_width=True,
    )
    st.plotly_chart(px.scatter(scores, x="operating_margin", y="ev_to_sales", color="sector_theme", hover_name="ticker"), use_container_width=True)

with peer_tab:
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
        st.plotly_chart(px.line_polar(radar, r="percentile", theta="metric", line_close=True, range_r=[0, 100]), use_container_width=True)
        st.dataframe(selected_peer.T, use_container_width=True)

with deep_dive:
    company = scores[scores["ticker"] == selected_company].iloc[0]
    st.subheader(f"{company['company_name']} ({company['ticker']})")
    cols = st.columns(4)
    cols[0].metric("Investment score", f"{company['investment_screening_score']:.1f}")
    cols[1].metric("Platform score", f"{company['platform_candidate_score']:.1f}")
    cols[2].metric("Value creation", f"{company['value_creation_potential_score']:.1f}")
    cols[3].metric("Red flag", f"{company['red_flag_score']:.1f}")
    st.write(company["investment_screening_explanation"])
    chart_path = CHART_DIR / "red_flag_breakdown.png"
    if chart_path.exists():
        st.image(str(chart_path))

with rag_tab:
    chunks = load_filing_chunks()
    retriever = FilingRetriever(chunks, mode=retrieval_mode)
    default_question = "What are the company's main revenue drivers?"
    user_question = st.text_input("Diligence question", default_question)
    result = retriever.search(user_question, ticker=selected_company, top_k=top_k)
    if result.empty:
        st.warning("insufficient filing evidence found")
    else:
        display_result = result.copy()
        if "rank" not in display_result.columns:
            display_result["rank"] = range(1, len(display_result) + 1)
        st.dataframe(
            display_result[["rank", "section_label", "combined_score", "filing_type", "filing_date", "chunk_id", "text"]],
            use_container_width=True,
        )
    if not evidence.empty:
        st.subheader("Precomputed Evidence")
        st.dataframe(evidence[evidence["ticker"] == selected_company].head(25), use_container_width=True)

with memo_tab:
    memo_path = REPORT_DIR / "investment_screening_memo.md"
    if memo_path.exists():
        st.markdown(memo_path.read_text(encoding="utf-8"))
    else:
        st.info("Run python -m src.reporting to generate the memo preview.")

with eval_tab:
    st.dataframe(eval_results, use_container_width=True)
    if not eval_results.empty:
        st.plotly_chart(px.bar(eval_results, x="metric", y="value", color="metric_type"), use_container_width=True)

with methodology:
    st.markdown(
        """
The structured scorecards are deterministic percentile-based screens using public market data,
SEC-style fundamentals and peer-relative metrics. The filing evidence layer retrieves source
passages and metadata from cached SEC filing chunks. Missing evidence is handled as insufficient
filing evidence found.

yfinance is an unofficial open-source library using Yahoo Finance publicly available interfaces
and should be treated as research and educational data. SEC EDGAR APIs provide official filing
and fundamental data, but XBRL tags vary by company and require careful mapping. This dashboard is
for screening workflow demonstration and is not investment advice.
"""
    )

if data_mode == "online API":
    st.sidebar.info("Online mode is implemented in the CLI refresh commands. The dashboard defaults to cached sample outputs for reliability.")
