"""
Compute SaaS and general financial metrics from Yahoo Finance / EDGAR data.

SaaS metrics:
  ARR  — Annual Recurring Revenue (approximated from quarterly revenue run-rate)
  NRR  — Net Revenue Retention (requires subscription revenue detail; approximated)
  Rule of 40 — Revenue Growth YoY% + FCF Margin%

General metrics:
  Revenue CAGR (3yr)
  EBITDA Margin
  FCF Yield (FCF / Market Cap)
  EV/Revenue, EV/EBITDA, P/FCF
  Gross Margin
  Net Income Margin
"""

import numpy as np
import pandas as pd
from typing import Optional


def _safe(series, key, default=None):
    try:
        val = series.loc[key]
        if pd.isna(val):
            return default
        return float(val)
    except (KeyError, TypeError):
        return default


def _most_recent_col(df: pd.DataFrame, label_matches: list) -> Optional[float]:
    """Search for a row label containing any of label_matches (case-insensitive)."""
    if df is None or df.empty:
        return None
    for col in df.columns:
        for row_label in df.index:
            if any(m.lower() in str(row_label).lower() for m in label_matches):
                val = df.loc[row_label, col]
                if pd.notna(val):
                    return float(val)
    return None


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


def compute_saas_metrics(info: dict, financials: dict) -> dict:
    """
    Compute SaaS-specific metrics.
    ARR is approximated as most recent quarter revenue * 4.
    Rule of 40 = YoY revenue growth % + FCF margin %.
    NRR is approximated if subscription/service revenue segments available,
    otherwise flagged as unavailable.
    """
    results = {}

    # --- ARR (most recent quarter revenue annualized) ---
    q_income = financials.get("quarterly_income")
    q_cashflow = financials.get("quarterly_cashflow")
    income = financials.get("income_stmt")
    cashflow = financials.get("cashflow")

    arr = None
    latest_q_rev = None
    if q_income is not None and not q_income.empty:
        for row_label in q_income.index:
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                vals = q_income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 1:
                    latest_q_rev = float(vals.iloc[0])
                    arr = latest_q_rev * 4
                    break

    results["ARR (approx)"] = arr

    # --- TTM Revenue ---
    ttm_rev = None
    if q_income is not None and not q_income.empty:
        ttm_rev = _ttm_value(q_income, ["total revenue", "revenue"])
    if ttm_rev is None and income is not None and not income.empty:
        for row_label in income.index:
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                vals = income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 1:
                    ttm_rev = float(vals.iloc[0])
                    break

    results["TTM Revenue"] = ttm_rev

    # --- Revenue Growth YoY ---
    rev_growth = None
    if income is not None and not income.empty:
        for row_label in income.index:
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                vals = income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 2:
                    rev_growth = (float(vals.iloc[0]) / float(vals.iloc[1]) - 1) * 100
                break

    results["Revenue Growth YoY %"] = rev_growth

    # --- TTM FCF ---
    ttm_fcf = None
    if q_cashflow is not None and not q_cashflow.empty:
        ttm_fcf = _ttm_value(q_cashflow, ["free cash flow", "capital expenditure"])
    if ttm_fcf is None:
        # Try from info
        ttm_fcf = info.get("freeCashflow")

    results["TTM FCF"] = ttm_fcf

    # --- FCF Margin ---
    fcf_margin = None
    if ttm_fcf is not None and ttm_rev and ttm_rev > 0:
        fcf_margin = (ttm_fcf / ttm_rev) * 100
    results["FCF Margin %"] = fcf_margin

    # --- Rule of 40 ---
    rule_of_40 = None
    if rev_growth is not None and fcf_margin is not None:
        rule_of_40 = rev_growth + fcf_margin
    results["Rule of 40"] = rule_of_40

    # --- Gross Margin ---
    gross_margin = None
    if income is not None and not income.empty:
        gross_profit = None
        revenue = None
        for row_label in income.index:
            if "gross profit" in str(row_label).lower():
                vals = income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 1:
                    gross_profit = float(vals.iloc[0])
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                vals = income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 1:
                    revenue = float(vals.iloc[0])
        if gross_profit is not None and revenue and revenue > 0:
            gross_margin = (gross_profit / revenue) * 100
    results["Gross Margin %"] = gross_margin

    # --- NRR: not reliably available from public filings without segment detail ---
    results["NRR"] = "N/A (requires subscription cohort data)"

    return results


def compute_general_metrics(info: dict, financials: dict) -> dict:
    """Metrics applicable to all companies."""
    results = {}
    income = financials.get("income_stmt")
    cashflow = financials.get("cashflow")
    balance = financials.get("balance_sheet")
    q_cashflow = financials.get("quarterly_cashflow")

    market_cap = info.get("marketCap")
    results["Market Cap"] = market_cap

    # EV
    ev = info.get("enterpriseValue")
    results["Enterprise Value"] = ev

    # Revenue (TTM from info or annual)
    rev_ttm = info.get("totalRevenue")
    results["TTM Revenue"] = rev_ttm

    # Revenue growth 3yr CAGR
    rev_cagr = None
    if income is not None and not income.empty:
        for row_label in income.index:
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                vals = income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 4:
                    rev_cagr = ((float(vals.iloc[0]) / float(vals.iloc[3])) ** (1 / 3) - 1) * 100
                elif len(vals) >= 2:
                    rev_cagr = (float(vals.iloc[0]) / float(vals.iloc[-1]) - 1) * 100
                break
    results["Revenue CAGR 3yr %"] = rev_cagr

    # EBITDA Margin
    ebitda = info.get("ebitda")
    ebitda_margin = None
    if ebitda and rev_ttm and rev_ttm > 0:
        ebitda_margin = (ebitda / rev_ttm) * 100
    results["EBITDA Margin %"] = ebitda_margin

    # FCF
    fcf = info.get("freeCashflow")
    if fcf is None and q_cashflow is not None:
        fcf = _ttm_value(q_cashflow, ["free cash flow"])
    results["TTM FCF"] = fcf

    fcf_yield = None
    if fcf and market_cap and market_cap > 0:
        fcf_yield = (fcf / market_cap) * 100
    results["FCF Yield %"] = fcf_yield

    # Gross Margin
    gross_margin = info.get("grossMargins")
    if gross_margin:
        gross_margin = gross_margin * 100
    results["Gross Margin %"] = gross_margin

    # Net Margin
    net_margin = info.get("profitMargins")
    if net_margin:
        net_margin = net_margin * 100
    results["Net Margin %"] = net_margin

    # Multiples
    ev_rev = None
    if ev and rev_ttm and rev_ttm > 0:
        ev_rev = ev / rev_ttm
    results["EV/Revenue"] = ev_rev

    ev_ebitda = info.get("enterpriseToEbitda")
    results["EV/EBITDA"] = ev_ebitda

    pe = info.get("trailingPE")
    results["P/E (trailing)"] = pe

    p_fcf = None
    if fcf and market_cap and market_cap > 0 and fcf > 0:
        p_fcf = market_cap / fcf
    results["P/FCF"] = p_fcf

    # Debt info
    total_debt = info.get("totalDebt", 0) or 0
    cash = info.get("totalCash", 0) or 0
    results["Total Debt"] = total_debt
    results["Cash & Equivalents"] = cash
    results["Net Debt"] = total_debt - cash

    return results


def is_saas_company(info: dict) -> bool:
    """Heuristic: check industry/sector tags for SaaS indicators."""
    industry = str(info.get("industry", "")).lower()
    sector = str(info.get("sector", "")).lower()
    long_biz = str(info.get("longBusinessSummary", "")).lower()

    saas_keywords = [
        "software", "saas", "cloud", "subscription", "platform", "application software",
        "software-infrastructure", "internet content"
    ]
    for kw in saas_keywords:
        if kw in industry or kw in sector:
            return True
    # Heuristic from description
    if "subscription" in long_biz and "software" in long_biz:
        return True
    return False


def extract_dcf_inputs_from_data(ticker: str, info: dict, financials: dict, risk_free_rate: float) -> dict:
    """
    Extract the key DCF building blocks from fetched data.
    Returns a dict with fields needed to populate DCFInputs.
    """
    income = financials.get("income_stmt")
    cashflow = financials.get("cashflow")
    q_cashflow = financials.get("quarterly_cashflow")

    # TTM Revenue
    rev_ttm = info.get("totalRevenue")

    # Historical revenue growth rates (annual, up to 5 years)
    revenue_growth_rates = []
    if income is not None and not income.empty:
        for row_label in income.index:
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                vals = income.loc[row_label].dropna().sort_index(ascending=False)
                for i in range(min(5, len(vals) - 1)):
                    g = (float(vals.iloc[i]) / float(vals.iloc[i + 1])) - 1
                    revenue_growth_rates.append(g)
                break

    if not revenue_growth_rates:
        revenue_growth_rates = [0.07]  # fallback 7% growth

    # Reverse so oldest first, then extend for projection
    revenue_growth_rates = list(reversed(revenue_growth_rates))

    # Estimate stable FCF margin from last 3 years average
    fcf_margin_stable = 0.10
    if cashflow is not None and not cashflow.empty:
        fcf_vals = []
        rev_vals = []
        for row_label in cashflow.index:
            if "free cash flow" in str(row_label).lower():
                fcf_vals = [float(v) for v in cashflow.loc[row_label].dropna().values[:3]]
        for row_label in (income.index if income is not None else []):
            if any(m in str(row_label).lower() for m in ["total revenue", "revenue"]):
                rev_vals = [float(v) for v in income.loc[row_label].dropna().values[:3]]
        if fcf_vals and rev_vals:
            margins = [f / r for f, r in zip(fcf_vals, rev_vals) if r > 0]
            if margins:
                fcf_margin_stable = np.mean(margins)

    # TTM FCF
    ttm_fcf = info.get("freeCashflow")
    if ttm_fcf is None and q_cashflow is not None:
        ttm_fcf = _ttm_value(q_cashflow, ["free cash flow"])
    if ttm_fcf is None:
        ttm_fcf = (rev_ttm or 0) * fcf_margin_stable

    # WACC components
    beta = info.get("beta", 1.0) or 1.0
    beta = max(0.3, min(beta, 3.0))  # clamp extreme betas

    market_cap = info.get("marketCap", 0) or 0
    total_debt = info.get("totalDebt", 0) or 0
    total_cash = info.get("totalCash", 0) or 0

    # Approximate cost of debt from interest expense
    interest_expense = None
    if income is not None and not income.empty:
        for row_label in income.index:
            if "interest expense" in str(row_label).lower():
                vals = income.loc[row_label].dropna().sort_index(ascending=False)
                if len(vals) >= 1:
                    interest_expense = abs(float(vals.iloc[0]))
                break
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

    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0
    net_debt = total_debt - total_cash

    return {
        "ticker": ticker,
        "current_fcf": ttm_fcf,
        "revenue_growth_rates": revenue_growth_rates,
        "fcf_margin_stable": max(fcf_margin_stable, 0.01),
        "revenue_ttm": rev_ttm or 0,
        "wacc": wacc,
        "terminal_growth_rate": 0.025,
        "projection_years": 10,
        "shares_outstanding": shares,
        "net_debt": net_debt,
    }
