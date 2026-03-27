"""
Compute SaaS and general financial metrics from Yahoo Finance / EDGAR data.

Company types detected:
  SAAS      — software/cloud/subscription
  CYCLICAL  — energy, mining, semis, autos, steel, chemicals
  DEFENSE   — aerospace & defense
  FINANCIAL — banks, insurance, asset managers
  REIT      — real estate investment trusts
  BIOTECH   — pre-revenue or clinical-stage biotech
  GENERAL   — everything else
"""

import numpy as np
import pandas as pd
from typing import Optional


def _ttm_value(quarterly_df: pd.DataFrame, row_label_matches: list) -> Optional[float]:
    """Sum last 4 quarters for a given line item."""
    if quarterly_df is None or quarterly_df.empty:
        return None
    for row_label in quarterly_df.index:
        if any(m.lower() in str(row_label).lower() for m in row_label_matches):
            vals = quarterly_df.loc[row_label].dropna().sort_index(ascending=False).head(4)
            if len(vals) >= 2:
                return float(vals.sum())
    return None


def _compute_ttm_fcf(info: dict, financials: dict) -> Optional[float]:
    """
    Robustly compute TTM FCF.
    Priority: info["freeCashflow"] → quarterly OpCF-CapEx → annual OpCF-CapEx
    """
    # Yahoo pre-computes TTM FCF — most reliable
    fcf = info.get("freeCashflow")
    if fcf is not None:
        return fcf

    # Compute from quarterly: Operating CF - |CapEx|
    q_cf = financials.get("quarterly_cashflow")
    if q_cf is not None and not q_cf.empty:
        op_cf = _ttm_value(q_cf, ["operating cash flow", "cash from operations",
                                   "net cash from operating"])
        capex = _ttm_value(q_cf, ["capital expenditure", "capital expenditures",
                                   "purchase of property", "capex"])
        if op_cf is not None and capex is not None:
            return op_cf - abs(capex)
        # Try direct free cash flow row
        direct = _ttm_value(q_cf, ["free cash flow"])
        if direct is not None:
            return direct

    # Fall back to most recent annual
    cf = financials.get("cashflow")
    if cf is not None and not cf.empty:
        op_vals  = _get_annual_series(cf, ["operating cash flow", "cash from operations"], 1)
        cap_vals = _get_annual_series(cf, ["capital expenditure", "capital expenditures",
                                           "purchase of property"], 1)
        if op_vals and cap_vals:
            return op_vals[0] - abs(cap_vals[0])
        direct = _get_annual_series(cf, ["free cash flow"], 1)
        if direct:
            return direct[0]

    return None


# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------

def is_saas_company(info: dict) -> bool:
    industry = str(info.get("industry", "")).lower()
    sector   = str(info.get("sector",   "")).lower()
    biz      = str(info.get("longBusinessSummary", "")).lower()
    saas_kw  = ["software", "saas", "cloud", "application software",
                "software-infrastructure", "internet content"]
    for kw in saas_kw:
        if kw in industry or kw in sector:
            return True
    if "subscription" in biz and "software" in biz:
        return True
    return False


def classify_company(info: dict) -> str:
    """
    Return one of: SAAS, CYCLICAL, DEFENSE, FINANCIAL, REIT, BIOTECH, GENERAL
    Order matters — more specific checks first.
    """
    sector   = str(info.get("sector",   "")).lower()
    industry = str(info.get("industry", "")).lower()

    if sector == "financial services" or any(k in industry for k in [
        "bank", "insurance", "asset management", "capital markets", "mortgage", "credit"
    ]):
        return "FINANCIAL"

    if sector == "real estate" or "reit" in industry:
        return "REIT"

    if "aerospace & defense" in industry:
        return "DEFENSE"

    cyclical_kw = [
        "oil", "gas", "energy", "mining", "metal", "steel", "chemical",
        "semiconductor", "auto", "copper", "aluminum", "coal",
    ]
    if any(k in industry for k in cyclical_kw):
        return "CYCLICAL"

    if any(k in industry for k in ["biotechnology", "drug manufacturer"]):
        rev = info.get("totalRevenue") or 0
        fcf = info.get("freeCashflow") or 0
        if rev < 500_000_000 or fcf < 0:
            return "BIOTECH"

    if is_saas_company(info):
        return "SAAS"

    return "GENERAL"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _get_annual_series(df: pd.DataFrame, label_matches: list, n: int = 7) -> list:
    """Return up to n annual values (most-recent first) for a row label."""
    if df is None or df.empty:
        return []
    for row_label in df.index:
        if any(m in str(row_label).lower() for m in label_matches):
            return [float(v) for v in df.loc[row_label].dropna().sort_index(ascending=False).values[:n]]
    return []


# ---------------------------------------------------------------------------
# Cycle-normalized FCF margin (for CYCLICAL)
# ---------------------------------------------------------------------------

def compute_normalized_fcf_margin(
    income: pd.DataFrame,
    cashflow: pd.DataFrame,
    years: int = 7,
) -> tuple:
    """Return (normalized_fcf_margin, years_used). Uses up to `years` annual periods."""
    fcf_vals = _get_annual_series(cashflow, ["free cash flow"], years) if cashflow is not None else []
    rev_vals = _get_annual_series(income,   ["total revenue", "revenue"], years) if income is not None else []
    margins  = [f / r for f, r in zip(fcf_vals, rev_vals) if r > 0]
    if not margins:
        return 0.08, 0
    return float(np.mean(margins)), len(margins)


# ---------------------------------------------------------------------------
# General metrics (all company types)
# ---------------------------------------------------------------------------

def compute_general_metrics(info: dict, financials: dict) -> dict:
    results  = {}
    income   = financials.get("income_stmt")

    market_cap = info.get("marketCap")
    ev         = info.get("enterpriseValue")
    rev_ttm    = info.get("totalRevenue")

    results["Market Cap"]        = market_cap
    results["Enterprise Value"]  = ev
    results["TTM Revenue"]       = rev_ttm

    # Revenue CAGR 3yr — from annual statements (needs 4 data points for 3yr CAGR)
    rev_cagr = None
    rev_series = _get_annual_series(income, ["total revenue", "revenue"], 4)
    if len(rev_series) >= 4 and rev_series[3] > 0:
        rev_cagr = ((rev_series[0] / rev_series[3]) ** (1 / 3) - 1) * 100
    elif len(rev_series) >= 2 and rev_series[-1] > 0:
        rev_cagr = (rev_series[0] / rev_series[-1] - 1) * 100
    results["Revenue CAGR 3yr %"] = rev_cagr

    ebitda = info.get("ebitda")
    results["EBITDA Margin %"] = (ebitda / rev_ttm * 100) if (ebitda and rev_ttm) else None

    gross_margin = info.get("grossMargins")
    results["Gross Margin %"]  = (gross_margin * 100) if gross_margin else None

    net_margin = info.get("profitMargins")
    results["Net Margin %"]    = (net_margin * 100) if net_margin else None

    # FCF: use info (Yahoo TTM) first, then compute from statements
    fcf = _compute_ttm_fcf(info, financials)
    results["TTM FCF"]     = fcf
    results["FCF Yield %"] = ((fcf / market_cap) * 100) if (fcf and market_cap) else None

    results["EV/Revenue"]      = (ev / rev_ttm) if (ev and rev_ttm) else None
    results["EV/EBITDA"]       = info.get("enterpriseToEbitda")
    results["P/E (trailing)"]  = info.get("trailingPE")
    results["P/FCF"]           = (market_cap / fcf) if (fcf and fcf > 0 and market_cap) else None

    total_debt = info.get("totalDebt", 0) or 0
    cash       = info.get("totalCash",  0) or 0
    results["Total Debt"]         = total_debt
    results["Cash & Equivalents"] = cash
    results["Net Debt"]           = total_debt - cash

    return results


# ---------------------------------------------------------------------------
# SaaS metrics
# ---------------------------------------------------------------------------

def compute_saas_metrics(info: dict, financials: dict) -> dict:
    results  = {}
    q_income = financials.get("quarterly_income")
    income   = financials.get("income_stmt")

    # --- TTM Revenue: Yahoo pre-computes this, most reliable ---
    rev_ttm = info.get("totalRevenue")
    if rev_ttm is None:
        rev_ttm = _ttm_value(q_income, ["total revenue", "revenue"])
    results["TTM Revenue"] = rev_ttm

    # --- ARR: most recent single quarter * 4 ---
    # We need one quarter's value, NOT a TTM sum.
    arr = None
    if q_income is not None and not q_income.empty:
        for row_label in q_income.index:
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                vals = q_income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 1:
                    arr = float(vals.iloc[0]) * 4
                break
    results["ARR (approx)"] = arr

    # --- Revenue Growth YoY ---
    # info["revenueGrowth"] is Yahoo's pre-computed YoY % (decimal) — use it first.
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        rev_growth = rev_growth * 100
    else:
        # Fallback: two most recent full fiscal years from annual stmt
        rev_series = _get_annual_series(income, ["total revenue", "revenue"], 2)
        if len(rev_series) >= 2 and rev_series[1] != 0:
            rev_growth = ((rev_series[0] / rev_series[1]) - 1) * 100
    results["Revenue Growth YoY %"] = rev_growth

    # --- TTM FCF ---
    ttm_fcf = _compute_ttm_fcf(info, financials)
    results["TTM FCF"] = ttm_fcf

    # --- FCF Margin (use same base: TTM FCF / TTM Revenue) ---
    fcf_margin = ((ttm_fcf / rev_ttm) * 100) if (ttm_fcf is not None and rev_ttm) else None
    results["FCF Margin %"] = fcf_margin

    # --- Rule of 40 ---
    results["Rule of 40"] = (rev_growth + fcf_margin) if (rev_growth is not None and fcf_margin is not None) else None
    results["NRR"]         = "N/A (requires subscription cohort data)"

    # --- Gross Margin: info has this pre-computed ---
    gm = info.get("grossMargins")
    results["Gross Margin %"] = (gm * 100) if gm else None

    return results


# ---------------------------------------------------------------------------
# Financial-sector metrics (banks, insurance, asset managers)
# ---------------------------------------------------------------------------

def compute_financial_metrics(info: dict, financials: dict) -> dict:
    results = {}
    income  = financials.get("income_stmt")

    results["P/B Ratio"]        = info.get("priceToBook")
    roe = info.get("returnOnEquity")
    results["ROE %"]            = (roe * 100) if roe else None
    div = info.get("dividendYield")
    results["Dividend Yield %"] = (div * 100) if div else None
    results["P/E (trailing)"]   = info.get("trailingPE")
    results["P/E (forward)"]    = info.get("forwardPE")

    # Net Interest Margin proxy
    nim = None
    nim_series = _get_annual_series(income, ["net interest income"], 1) if income is not None else []
    if nim_series:
        total_assets = info.get("totalAssets")
        if total_assets and total_assets > 0:
            nim = (nim_series[0] / total_assets) * 100
    results["Net Interest Margin % (approx)"] = nim
    results["Tier 1 Capital Ratio"]           = "N/A — check latest 10-K"

    return results


# ---------------------------------------------------------------------------
# REIT metrics
# ---------------------------------------------------------------------------

def compute_reit_metrics(info: dict, financials: dict) -> dict:
    results = {}
    income  = financials.get("income_stmt")

    div = info.get("dividendYield")
    results["Dividend Yield %"] = (div * 100) if div else None
    results["P/E (trailing)"]   = info.get("trailingPE")

    # FFO proxy = Net Income + D&A
    ni_series = _get_annual_series(income, ["net income"], 1)  if income is not None else []
    da_series = _get_annual_series(income, ["depreciation"],  1) if income is not None else []

    ffo = None
    if ni_series and da_series:
        ffo = ni_series[0] + abs(da_series[0])
    results["FFO proxy (NI + D&A)"] = ffo

    market_cap = info.get("marketCap")
    results["P/FFO (proxy)"] = (market_cap / ffo) if (ffo and ffo > 0 and market_cap) else None
    results["AFFO Note"]     = "AFFO requires maintenance capex split — not in public feeds"

    return results


# ---------------------------------------------------------------------------
# Biotech / pre-revenue metrics
# ---------------------------------------------------------------------------

def compute_biotech_metrics(info: dict, financials: dict) -> dict:
    results    = {}
    q_cashflow = financials.get("quarterly_cashflow")

    cash = info.get("totalCash") or 0
    results["Cash & Equivalents"] = cash

    quarterly_burn = None
    if q_cashflow is not None and not q_cashflow.empty:
        for row_label in q_cashflow.index:
            if "operating" in str(row_label).lower():
                vals = q_cashflow.loc[row_label].dropna().sort_index(ascending=False).head(4)
                outflows = [abs(float(v)) for v in vals if float(v) < 0]
                if outflows:
                    quarterly_burn = float(np.mean(outflows))
                break

    results["Avg Quarterly Cash Burn"] = quarterly_burn
    results["Cash Runway (quarters)"]  = (cash / quarterly_burn) if (quarterly_burn and quarterly_burn > 0 and cash > 0) else None
    results["DCF Warning"] = (
        "DCF not meaningful — negative/near-zero FCF. "
        "Valuation depends on pipeline probability weighting."
    )
    return results


# ---------------------------------------------------------------------------
# Defense / A&D flags
# ---------------------------------------------------------------------------

def compute_defense_flags(info: dict, financials: dict) -> dict:
    return {
        "Backlog Note": (
            "Revenue is backlog-driven (government contracts). "
            "DCF uses conservative terminal growth capped at 2.0%."
        ),
        "_terminal_growth_override": 0.020,
    }


# ---------------------------------------------------------------------------
# DCF input extraction
# ---------------------------------------------------------------------------

def extract_dcf_inputs_from_data(
    ticker: str,
    info: dict,
    financials: dict,
    risk_free_rate: float,
    company_type: str = "GENERAL",
) -> dict:
    """
    Extract DCF building blocks from fetched data.
    Keys prefixed with '_' are metadata for display — pop them before DCFInputs().
    """
    income   = financials.get("income_stmt")
    cashflow = financials.get("cashflow")

    rev_ttm = info.get("totalRevenue")

    # Historical annual revenue growth rates (up to 5, oldest first)
    rev_series = _get_annual_series(income, ["total revenue", "revenue"], 6)
    revenue_growth_rates = []
    for i in range(min(5, len(rev_series) - 1)):
        g = (rev_series[i] / rev_series[i + 1]) - 1
        revenue_growth_rates.append(g)
    revenue_growth_rates = list(reversed(revenue_growth_rates)) or [0.07]

    # FCF margin: cycle-normalized for CYCLICAL, else 3yr average
    fcf_margin_stable = 0.10
    metadata = {}

    if company_type == "CYCLICAL":
        fcf_margin_stable, yrs = compute_normalized_fcf_margin(income, cashflow, years=7)
        metadata["_margin_note"] = (
            f"FCF margin cycle-normalized over {yrs} yr(s)" if yrs > 0
            else "FCF margin: insufficient history, using 8% default"
        )
    else:
        fcf_vals = _get_annual_series(cashflow, ["free cash flow"], 3) if cashflow is not None else []
        rev_vals = _get_annual_series(income,   ["total revenue", "revenue"], 3)
        margins  = [f / r for f, r in zip(fcf_vals, rev_vals) if r > 0]
        if margins:
            fcf_margin_stable = float(np.mean(margins))

    # TTM FCF — use robust helper
    ttm_fcf = _compute_ttm_fcf(info, financials)
    if ttm_fcf is None:
        ttm_fcf = (rev_ttm or 0) * fcf_margin_stable

    # WACC
    beta = max(0.3, min(info.get("beta", 1.0) or 1.0, 3.0))
    market_cap = info.get("marketCap", 0) or 0
    total_debt = info.get("totalDebt",  0) or 0
    total_cash = info.get("totalCash",  0) or 0

    interest_vals = _get_annual_series(income, ["interest expense"], 1) if income is not None else []
    interest_expense = abs(interest_vals[0]) if interest_vals else None
    cost_of_debt = (interest_expense / total_debt) if (interest_expense and total_debt > 0) else 0.05

    from dcf import build_wacc, TAX_RATE_DEFAULT, EQUITY_RISK_PREMIUM
    wacc = build_wacc(
        beta=beta,
        risk_free_rate=risk_free_rate,
        equity_value=market_cap,
        debt=total_debt,
        cost_of_debt=cost_of_debt,
        tax_rate=TAX_RATE_DEFAULT,
        equity_risk_premium=EQUITY_RISK_PREMIUM,
    )

    terminal_growth_rate = 0.025
    if company_type == "DEFENSE":
        terminal_growth_rate = 0.020
        metadata["_margin_note"] = "Terminal growth capped at 2.0% (defense/govt contract)"

    shares  = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0
    net_debt = total_debt - total_cash

    result = {
        "ticker": ticker,
        "current_fcf": ttm_fcf,
        "revenue_growth_rates": revenue_growth_rates,
        "fcf_margin_stable": max(fcf_margin_stable, 0.01),
        "revenue_ttm": rev_ttm or 0,
        "wacc": wacc,
        "terminal_growth_rate": terminal_growth_rate,
        "projection_years": 10,
        "shares_outstanding": shares,
        "net_debt": net_debt,
    }
    result.update(metadata)
    return result
