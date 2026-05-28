from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CHART_DIR, PROCESSED_DIR, ensure_project_dirs
from src.utils import read_csv_if_exists


plt.style.use("seaborn-v0_8-whitegrid")


def _save(fig: plt.Figure, name: str) -> None:
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHART_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sector_score_distribution(scores: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    scores.boxplot(column="investment_screening_score", by="sector_theme", ax=ax, rot=35)
    ax.set_title("Investment Screening Score by Sector Theme")
    ax.set_ylabel("Score")
    fig.suptitle("")
    _save(fig, "sector_score_distribution.png")


def top_screened_companies(scores: pd.DataFrame) -> None:
    top = scores.sort_values("investment_screening_score", ascending=True).tail(10)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["ticker"], top["investment_screening_score"], color="#2f6f8f")
    ax.set_xlabel("Investment Screening Score")
    ax.set_title("Top Screened Companies")
    _save(fig, "top_screened_companies.png")


def scatter_chart(scores: pd.DataFrame, x: str, y: str, name: str, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = scores["sector_theme"].astype("category")
    ax.scatter(scores[x], scores[y], c=categories.cat.codes, cmap="tab10", s=70, alpha=0.82)
    for _, row in scores.sort_values("investment_screening_score", ascending=False).head(5).iterrows():
        ax.annotate(row["ticker"], (row[x], row[y]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, name)


def peer_percentile_radar(scores: pd.DataFrame, benchmarks: pd.DataFrame) -> None:
    top_ticker = scores.sort_values("investment_screening_score", ascending=False).iloc[0]["ticker"]
    row = benchmarks[benchmarks["ticker"] == top_ticker].iloc[0]
    labels = ["Growth", "Margin", "FCF", "Leverage", "Valuation", "Volatility"]
    values = [
        row["revenue_growth_yoy_peer_percentile"],
        row["operating_margin_peer_percentile"],
        row["fcf_conversion_peer_percentile"],
        row["debt_to_ebit_proxy_peer_percentile"],
        row["ev_to_sales_peer_percentile"],
        row["realised_volatility_peer_percentile"],
    ]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    ax.plot(angles, values, color="#2f6f8f", linewidth=2)
    ax.fill(angles, values, color="#2f6f8f", alpha=0.18)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title(f"Peer Percentile Radar: {top_ticker}")
    _save(fig, "peer_percentile_radar.png")


def red_flag_breakdown(scores: pd.DataFrame) -> None:
    columns = [
        "credit_risk_score",
        "red_flag_score",
        "drawdown_risk_score",
        "volatility_risk_score",
        "data_gap_risk_score",
    ]
    means = scores[columns].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(means.index.str.replace("_", " ").str.title(), means.values, color="#9a4d41")
    ax.set_xlabel("Average Score")
    ax.set_title("Red Flag Component Breakdown")
    _save(fig, "red_flag_breakdown.png")


def category_mix(categories: pd.DataFrame) -> None:
    counts = categories["primary_category"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(counts.index, counts.values, color="#4f7f52")
    ax.set_xlabel("Companies")
    ax.set_title("Company Category Mix")
    _save(fig, "category_mix.png")


def rag_eval_summary(eval_results: pd.DataFrame) -> None:
    wanted = ["hit_rate_at_3", "hit_rate_at_5", "precision_at_3", "precision_at_5", "citation_coverage", "no_answer_accuracy"]
    subset = eval_results[eval_results["metric"].isin(wanted)].copy()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(subset["metric"].str.replace("_", "\n"), subset["value"].astype(float), color="#5a6f9a")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("RAG Evaluation Summary")
    _save(fig, "rag_eval_summary.png")


def generate_all_charts() -> None:
    ensure_project_dirs()
    scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
    categories = read_csv_if_exists(PROCESSED_DIR / "company_categories.csv")
    benchmarks = read_csv_if_exists(PROCESSED_DIR / "peer_benchmarks.csv")
    eval_results = read_csv_if_exists(PROCESSED_DIR / "rag_eval_results.csv")
    if scores.empty:
        return
    sector_score_distribution(scores)
    top_screened_companies(scores)
    scatter_chart(scores, "ev_to_sales", "revenue_growth_yoy", "valuation_vs_growth.png", "Valuation vs Revenue Growth", "EV / Sales", "Revenue Growth")
    scatter_chart(scores, "ev_to_sales", "operating_margin", "margin_vs_valuation.png", "Margin vs Valuation", "EV / Sales", "EBIT Margin Proxy")
    scatter_chart(scores, "debt_to_ebit_proxy", "fcf_conversion", "leverage_vs_cash_conversion.png", "Leverage vs Cash Conversion", "Debt / EBIT Proxy", "FCF Conversion Proxy")
    if not benchmarks.empty:
        peer_percentile_radar(scores, benchmarks)
    red_flag_breakdown(scores)
    if not categories.empty:
        category_mix(categories)
    if not eval_results.empty:
        rag_eval_summary(eval_results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate portfolio charts.")
    parser.parse_args()
    generate_all_charts()
    print("charts_generated=true")


if __name__ == "__main__":
    main()
