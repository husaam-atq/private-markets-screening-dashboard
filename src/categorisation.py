from __future__ import annotations

import argparse

import pandas as pd

from src.config import PROCESSED_DIR, ensure_project_dirs, load_config
from src.scoring import run_scoring
from src.utils import read_csv_if_exists, write_csv


CATEGORY_ORDER = [
    "Public Market Quality Compounder",
    "PE Platform Candidate",
    "Public-to-Private Candidate",
    "Value / Re-rating Candidate",
    "Operational Improvement Candidate",
    "Leveraged Credit Watchlist",
    "Distressed / Special Situations Watchlist",
    "Low Priority / Reject",
    "Insufficient Data",
]


def _tags(row: pd.Series) -> list[str]:
    tags: list[str] = []
    if row.get("public_quality_score", 0) >= 68:
        tags.append("benchmark quality comp")
    if row["investment_screening_score"] >= 70:
        tags.append("high structured screen")
    if row["platform_candidate_score"] >= 70 and row.get("platform_size_feasibility_score", 0) >= 50:
        tags.append("PE platform characteristics")
    if row.get("public_to_private_feasibility_score", 0) >= 65:
        tags.append("public-to-private feasibility")
    if row["value_creation_potential_score"] >= 68:
        tags.append("value creation angle")
    if row["credit_risk_score"] >= 64:
        tags.append("credit watch")
    if row["red_flag_score"] >= 65:
        tags.append("heightened diligence risk")
    if row["data_quality_score"] < 70:
        tags.append("data gaps")
    return tags or ["monitor"]


def categorize_company(row: pd.Series) -> tuple[str, list[str], str]:
    thresholds = load_config("scoring_config.yaml")["thresholds"]
    ev_band = str(row.get("market_cap_band", "Insufficient Data"))
    mega_or_strategic = ev_band in {
        "Mega-cap benchmark / public quality comp",
        "Large-cap strategic / unlikely PE target",
    }
    if row["data_quality_score"] < thresholds["insufficient_data_quality_score"]:
        category = "Insufficient Data"
    elif row["red_flag_score"] >= thresholds["distressed_min_red_flag_score"] and row["credit_risk_score"] >= thresholds["credit_watch_min_score"]:
        category = "Distressed / Special Situations Watchlist"
    elif row["investment_screening_score"] <= thresholds["low_priority_max_investment_score"] and row["red_flag_score"] >= 65:
        category = "Low Priority / Reject"
    elif row.get("public_quality_score", 0) >= 68 and (mega_or_strategic or row.get("public_to_private_feasibility_score", 0) < 45):
        category = "Public Market Quality Compounder"
    elif row.get("public_to_private_feasibility_score", 0) >= 62 and row["investment_screening_score"] >= 58 and row["red_flag_score"] < 65:
        category = "Public-to-Private Candidate"
    elif row["platform_candidate_score"] >= 60 and row.get("platform_size_feasibility_score", 0) >= 50 and row["red_flag_score"] < 62:
        category = "PE Platform Candidate"
    elif row["credit_risk_score"] >= thresholds["credit_watch_min_score"]:
        category = "Leveraged Credit Watchlist"
    elif row["value_creation_potential_score"] >= thresholds["value_creation_min_score"]:
        category = "Operational Improvement Candidate"
    elif row["investment_screening_score"] <= thresholds["low_priority_max_investment_score"] or row["red_flag_score"] > 70:
        category = "Low Priority / Reject"
    else:
        category = "Value / Re-rating Candidate"
    tags = _tags(row)
    reason = (
        f"{category}: public quality {row.get('public_quality_score', 0):.1f}, "
        f"investment score {row['investment_screening_score']:.1f}, platform score "
        f"{row['platform_candidate_score']:.1f}, public-to-private feasibility "
        f"{row.get('public_to_private_feasibility_score', 0):.1f}, value creation score "
        f"{row['value_creation_potential_score']:.1f}, credit risk {row['credit_risk_score']:.1f}, "
        f"red flag {row['red_flag_score']:.1f}, market-cap band {ev_band}. Tags: {', '.join(tags)}."
    )
    return category, tags, reason


def categorize_companies(scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in scores.iterrows():
        category, tags, reason = categorize_company(row)
        rows.append(
            {
                "ticker": row["ticker"],
                "company_name": row["company_name"],
                "sector_theme": row["sector_theme"],
                "market_cap_band": row.get("market_cap_band", "Insufficient Data"),
                "primary_category": category,
                "category_tags": "; ".join(tags),
                "why_this_category": reason,
                "public_quality_score": row.get("public_quality_score"),
                "investment_screening_score": row["investment_screening_score"],
                "diligence_priority_score": row["diligence_priority_score"],
                "platform_candidate_score": row["platform_candidate_score"],
                "public_to_private_feasibility_score": row.get("public_to_private_feasibility_score"),
                "value_creation_potential_score": row["value_creation_potential_score"],
                "credit_risk_score": row["credit_risk_score"],
                "red_flag_score": row["red_flag_score"],
                "data_quality_score": row["data_quality_score"],
            }
        )
    result = pd.DataFrame(rows)
    result["primary_category"] = pd.Categorical(result["primary_category"], categories=CATEGORY_ORDER, ordered=True)
    return result.sort_values(["primary_category", "investment_screening_score"], ascending=[True, False]).reset_index(drop=True)


def run_categorisation() -> pd.DataFrame:
    ensure_project_dirs()
    scores = read_csv_if_exists(PROCESSED_DIR / "investment_scores.csv")
    if scores.empty:
        scores = run_scoring()
    categories = categorize_companies(scores)
    categories["primary_category"] = categories["primary_category"].astype(str)
    write_csv(categories, PROCESSED_DIR / "company_categories.csv")
    return categories


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize companies into diligence archetypes.")
    parser.parse_args()
    frame = run_categorisation()
    print(frame["primary_category"].value_counts().to_string())


if __name__ == "__main__":
    main()
