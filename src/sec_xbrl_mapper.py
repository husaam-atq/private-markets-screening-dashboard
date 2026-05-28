from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


CONCEPT_TAGS = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss", "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities", "LiabilitiesCurrent"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebtAndFinanceLeaseObligationsNoncurrent"],
    "current_debt": ["ShortTermBorrowings", "ShortTermDebtCurrent", "LongTermDebtCurrent"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "shares_outstanding": ["CommonStocksIncludingAdditionalPaidInCapital", "EntityCommonStockSharesOutstanding"],
    "depreciation_amortization": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization"],
    "interest_expense": ["InterestExpenseNonOperating", "InterestExpense"],
}


@dataclass
class XbrlFact:
    metric: str
    value: float
    fiscal_year: int
    form: str
    filed: str
    frame: str | None
    concept: str


def _fact_units(companyfacts: dict[str, Any], concept: str) -> dict[str, list[dict[str, Any]]]:
    return companyfacts.get("facts", {}).get("us-gaap", {}).get(concept, {}).get("units", {})


def _candidate_unit(units: dict[str, list[dict[str, Any]]]) -> str | None:
    for preferred in ("USD", "shares", "USD/shares"):
        if preferred in units:
            return preferred
    return next(iter(units.keys()), None) if units else None


def _annual_facts(units: dict[str, list[dict[str, Any]]], unit: str) -> list[dict[str, Any]]:
    records = units.get(unit, [])
    annual = [
        item
        for item in records
        if item.get("form") in {"10-K", "20-F", "40-F"} and item.get("fy") and item.get("val") is not None
    ]
    return sorted(annual, key=lambda item: (int(item.get("fy", 0)), str(item.get("filed", ""))))


def latest_fact_for_tags(companyfacts: dict[str, Any], tags: list[str]) -> XbrlFact | None:
    best: XbrlFact | None = None
    for concept in tags:
        units = _fact_units(companyfacts, concept)
        unit = _candidate_unit(units)
        if not unit:
            continue
        facts = _annual_facts(units, unit)
        if not facts:
            continue
        item = facts[-1]
        candidate = XbrlFact(
            metric="",
            value=float(item["val"]),
            fiscal_year=int(item["fy"]),
            form=str(item.get("form", "")),
            filed=str(item.get("filed", "")),
            frame=item.get("frame"),
            concept=concept,
        )
        if best is None or (candidate.fiscal_year, candidate.filed) > (best.fiscal_year, best.filed):
            best = candidate
    return best


def map_companyfacts(companyfacts: dict[str, Any], ticker: str, company_name: str, cik: int | str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    facts_by_metric: dict[str, list[dict[str, Any]]] = {}
    for metric, tags in CONCEPT_TAGS.items():
        facts_by_metric[metric] = []
        for concept in tags:
            units = _fact_units(companyfacts, concept)
            unit = _candidate_unit(units)
            if not unit:
                continue
            for fact in _annual_facts(units, unit):
                facts_by_metric[metric].append(
                    {
                        "metric": metric,
                        "value": float(fact["val"]),
                        "fiscal_year": int(fact["fy"]),
                        "filing_type": fact.get("form"),
                        "filing_date": fact.get("filed"),
                        "concept": concept,
                    }
                )
    years = sorted({fact["fiscal_year"] for facts in facts_by_metric.values() for fact in facts})
    for year in years[-4:]:
        row: dict[str, Any] = {
            "ticker": ticker,
            "company_name": company_name,
            "cik": int(cik),
            "fiscal_year": year,
            "filing_type": "10-K",
            "filing_date": "",
            "data_source": "sec_companyfacts",
        }
        for metric, facts in facts_by_metric.items():
            matches = [fact for fact in facts if fact["fiscal_year"] == year]
            if matches:
                selected = sorted(matches, key=lambda fact: str(fact.get("filing_date", "")))[-1]
                row[metric] = selected["value"]
                if not row["filing_date"]:
                    row["filing_date"] = selected.get("filing_date", "")
            else:
                row[metric] = np.nan
        debt_parts = [row.get("long_term_debt"), row.get("current_debt")]
        row["total_debt"] = np.nan if all(pd.isna(value) for value in debt_parts) else np.nansum(debt_parts)
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_companyfacts_payload(payload: dict[str, Any], ticker: str, company_name: str, cik: int | str) -> pd.DataFrame:
    frame = map_companyfacts(payload, ticker, company_name, cik)
    if frame.empty:
        return frame
    numeric_cols = [col for col in frame.columns if col not in {"ticker", "company_name", "filing_type", "filing_date", "data_source"}]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame
