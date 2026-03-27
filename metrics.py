"""
Financial metrics computed from the structured financials dict (edgar.py schema).

All compute_* functions accept:
  info        — Yahoo Finance info dict (price, market cap, beta, sector, etc.)
  financials  — structured dict from edgar.get_edgar_financials() or yf_dfs_to_structured()
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers retained for edgar.yf_dfs_to_structured fallback
# ---------------------------------------------------------------------------

def _get_annual_series(df, label_matches: list, n: int = 7) -> list:
    """Extract up to n annual values from a yfinance DataFrame (most-recent first)."""
    if df is None or df.empty:
        return []
    for row_label in df.index:
        if any(m.lower() in str(row_label).lower() for m in label_matches):
            return [float(v) for v in df.loc[row_label].dropna().sort_index(ascending=False).values[:n]]
    return []


def _ttm_value(q_df, label_matches: list) -> Optional[float]:
    """Sum last 4 quarters from a yfinance quarterly DataFrame."""
    if q_df is None or q_df.empty:
        return None
    for row_label in q_df.index:
        if any(m.lower() in str(row_label).lower() for m in label_matches):
            vals = q_df.loc[row_label].dropna().sort_index(ascending=False).head(4)
            if len(vals) >= 2:
                return float(vals.sum())
    return None


def _compute_ttm_fcf(info: dict, financials: dict) -> Optional[float]:
    """Retained for edgar.yf_dfs_to_structured; real path uses financials['fcf_ttm']."""
    return (
        financials.get("fcf_ttm")
        or info.get("freeCashflow")
    )


# ---------------------------------------------------------------------------
# Structured dict accessors
# ---------------------------------------------------------------------------

def _annual(fin: dict, key: str, n: int = 7) -> list:
    return (fin.get(key) or [])[:n]


def _ttm(fin: dict, key: str) -> Optional[float]:
    return fin.get(key)


def _safe_pct(numerator, denominator) -> Optional[float]:
    if numerator is not None and denominator and denominator != 0:
        return (numerator / denominator) * 100
    return None


def _cagr(series: list, years: int) -> Optional[float]:
    """CAGR from a most-recent-first list over `years` years."""
    if len(series) > years and series[years] and series[years] > 0:
        return ((series[0] / series[years]) ** (1 / years) - 1) * 100
    if len(series) >= 2 and series[-1] and series[-1] > 0:
        n = len(series) - 1
        return ((series[0] / series[-1]) ** (1 / n) - 1) * 100
    return None


# ---------------------------------------------------------------------------
# Company classification
# ---------------------------------------------------------------------------

def is_saas_company(info: dict) -> bool:
    industry = str(info.get("industry", "")).lower()
    sector   = str(info.get("sector",   "")).lower()
    biz      = str(info.get("longBusinessSummary", "")).lower()
    kws = ["software", "saas", "cloud", "application software",
           "software-infrastructure", "internet content"]
    for kw in kws:
        if kw in industry or kw in sector:
            return True
    return "subscription" in biz and "software" in biz


def classify_company(info: dict) -> str:
    """Return one of: SAAS CYCLICAL DEFENSE FINANCIAL REIT BIOTECH GENERAL"""
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

    if any(k in industry for k in [
        "oil", "gas", "energy", "mining", "metal", "steel", "chemical",
        "semiconductor", "auto", "copper", "aluminum", "coal",
    ]):
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
# General metrics — all company types
# ---------------------------------------------------------------------------

def compute_general_metrics(info: dict, fin: dict) -> dict:
    market_cap = info.get("marketCap")
    ev         = info.get("enterpriseValue")

    rev_a  = _annual(fin, "revenue_annual")
    fcf_a  = _annual(fin, "fcf_annual")
    gp_a   = _annual(fin, "gross_profit_annual")
    ni_a   = _annual(fin, "net_income_annual")
    ebi_a  = _annual(fin, "ebitda_annual")

    rev_ttm = _ttm(fin, "revenue_ttm") or (rev_a[0] if rev_a else None)
    fcf_ttm = _ttm(fin, "fcf_ttm")     or (fcf_a[0] if fcf_a else None)
    gp_ttm  = _ttm(fin, "gross_profit_ttm") or (gp_a[0] if gp_a else None)
    ni_ttm  = _ttm(fin, "net_income_ttm")   or (ni_a[0] if ni_a else None)

    ebitda_ttm = ebi_a[0] if ebi_a else None

    gross_margin  = _safe_pct(gp_ttm,    rev_ttm)
    ebitda_margin = _safe_pct(ebitda_ttm, rev_ttm)
    net_margin    = _safe_pct(ni_ttm,    rev_ttm)
    fcf_yield     = _safe_pct(fcf_ttm,   market_cap)

    # Use info gross/net margins as override if EDGAR didn't provide gross profit
    if gross_margin is None:
        gm = info.get("grossMargins")
        gross_margin = gm * 100 if gm else None
    if net_margin is None:
        nm = info.get("profitMargins")
        net_margin = nm * 100 if nm else None

    total_debt = fin.get("total_debt") or info.get("totalDebt") or 0
    cash       = fin.get("cash")       or info.get("totalCash")  or 0
    net_debt   = fin.get("net_debt",   total_debt - cash)

    return {
        "Market Cap":          market_cap,
        "Enterprise Value":    ev,
        "TTM Revenue":         rev_ttm,
        "Revenue CAGR 3yr %":  _cagr(rev_a, 3),
        "Gross Margin %":      gross_margin,
        "EBITDA Margin %":     ebitda_margin,
        "Net Margin %":        net_margin,
        "TTM FCF":             fcf_ttm,
        "FCF Yield %":         fcf_yield,
        "EV/Revenue":          (ev / rev_ttm) if (ev and rev_ttm) else None,
        "EV/EBITDA":           info.get("enterpriseToEbitda"),
        "P/E (trailing)":      info.get("trailingPE"),
        "P/FCF":               (market_cap / fcf_ttm) if (fcf_ttm and fcf_ttm > 0 and market_cap) else None,
        "Total Debt":          total_debt,
        "Cash & Equivalents":  cash,
        "Net Debt":            net_debt,
        "Source":              fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# SaaS metrics
# ---------------------------------------------------------------------------

def compute_saas_metrics(info: dict, fin: dict) -> dict:
    rev_a  = _annual(fin, "revenue_annual")
    rev_ttm = _ttm(fin, "revenue_ttm") or (rev_a[0] if rev_a else None) or info.get("totalRevenue")
    fcf_ttm = _ttm(fin, "fcf_ttm") or info.get("freeCashflow")

    # ARR: most recent quarter annualized
    # Best proxy: TTM revenue / 4 * 4 = TTM (but TTM != ARR for growing cos)
    # Better: most recent single quarter * 4
    # We get individual Q from the quarterly series in edgar, but structured dict
    # doesn't expose individual quarters. Use info["totalRevenue"] / 4 as proxy
    # since Yahoo uses TTM, then note it's approximate.
    arr = None
    if rev_ttm:
        arr = rev_ttm  # TTM ≈ ARR for subscription businesses; true ARR = last Q * 4
        # If we have at least 2 annual points, recent Q ≈ last_annual/4 * (1 + growth/4)
        rev_growth_decimal = info.get("revenueGrowth")
        if rev_growth_decimal is not None and rev_a:
            # Annualize most recent quarter: rev_ttm is already TTM
            # A better ARR = TTM * (1 + quarterly_growth)
            # quarterly_growth ≈ (1 + annual_growth)^0.25 - 1
            qg = (1 + rev_growth_decimal) ** 0.25 - 1
            arr = rev_ttm * (1 + qg)

    # Revenue growth: Yahoo pre-computed YoY % is most reliable
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        rev_growth = rev_growth * 100
    elif len(rev_a) >= 2 and rev_a[1]:
        rev_growth = (rev_a[0] / rev_a[1] - 1) * 100

    fcf_margin = _safe_pct(fcf_ttm, rev_ttm)
    rule_of_40 = (rev_growth + fcf_margin) if (rev_growth is not None and fcf_margin is not None) else None

    gm = info.get("grossMargins")
    gross_margin = (gm * 100) if gm else _safe_pct(
        _ttm(fin, "gross_profit_ttm") or (_annual(fin, "gross_profit_annual") or [None])[0],
        rev_ttm
    )

    return {
        "TTM Revenue":          rev_ttm,
        "ARR (approx)":         arr,
        "Revenue Growth YoY %": rev_growth,
        "TTM FCF":              fcf_ttm,
        "FCF Margin %":         fcf_margin,
        "Rule of 40":           rule_of_40,
        "Gross Margin %":       gross_margin,
        "NRR":                  "N/A (requires subscription cohort data)",
    }


# ---------------------------------------------------------------------------
# Financial sector (banks, insurance)
# ---------------------------------------------------------------------------

def compute_financial_metrics(info: dict, fin: dict) -> dict:
    roe = info.get("returnOnEquity")
    div = info.get("dividendYield")

    # NIM: net interest income from EDGAR if available
    # Not a standard XBRL concept we track, so leave as note
    return {
        "P/B Ratio":                      info.get("priceToBook"),
        "ROE %":                          (roe * 100) if roe else None,
        "Dividend Yield %":               (div * 100) if div else None,
        "P/E (trailing)":                 info.get("trailingPE"),
        "P/E (forward)":                  info.get("forwardPE"),
        "Net Interest Margin % (approx)": None,
        "Tier 1 Capital Ratio":           "N/A — check latest 10-K",
        "Data Source":                    fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# REIT metrics
# ---------------------------------------------------------------------------

def compute_reit_metrics(info: dict, fin: dict) -> dict:
    ni_a = _annual(fin, "net_income_annual")
    da_a = _annual(fin, "da_annual")

    ffo = None
    if ni_a and da_a:
        ffo = ni_a[0] + abs(da_a[0])

    market_cap = info.get("marketCap")
    div = info.get("dividendYield")

    return {
        "Dividend Yield %":      (div * 100) if div else None,
        "P/E (trailing)":        info.get("trailingPE"),
        "FFO proxy (NI + D&A)":  ffo,
        "P/FFO (proxy)":         (market_cap / ffo) if (ffo and ffo > 0 and market_cap) else None,
        "AFFO Note":             "AFFO requires maintenance capex split — not in public feeds",
        "Data Source":           fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# Biotech / pre-revenue
# ---------------------------------------------------------------------------

def compute_biotech_metrics(info: dict, fin: dict) -> dict:
    cash     = fin.get("cash") or info.get("totalCash") or 0
    op_cf_a  = _annual(fin, "op_cf_annual")
    # Quarterly burn: approximate from TTM op_cf / 4
    op_cf_ttm = _ttm(fin, "op_cf_ttm")
    quarterly_burn = None
    if op_cf_ttm is not None and op_cf_ttm < 0:
        quarterly_burn = abs(op_cf_ttm) / 4

    runway = (cash / quarterly_burn) if (quarterly_burn and quarterly_burn > 0 and cash > 0) else None

    return {
        "Cash & Equivalents":       cash,
        "Avg Quarterly Cash Burn":  quarterly_burn,
        "Cash Runway (quarters)":   runway,
        "DCF Warning":              "DCF not meaningful — negative/near-zero FCF. Valuation depends on pipeline probability weighting.",
        "Data Source":              fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# Defense flags
# ---------------------------------------------------------------------------

def compute_defense_flags(info: dict, fin: dict) -> dict:
    return {
        "Backlog Note":               "Revenue is backlog-driven (govt contracts). Terminal growth capped at 2.0%.",
        "_terminal_growth_override":  0.020,
    }


# ---------------------------------------------------------------------------
# Cyclical: cycle-normalized FCF margin
# ---------------------------------------------------------------------------

def compute_normalized_fcf_margin(fin: dict, years: int = 7) -> tuple:
    """Return (normalized_fcf_margin, years_used) using EDGAR annual series."""
    fcf_a = _annual(fin, "fcf_annual", years)
    rev_a = _annual(fin, "revenue_annual", years)
    margins = [f / r for f, r in zip(fcf_a, rev_a) if r > 0]
    if not margins:
        return 0.08, 0
    return float(np.mean(margins)), len(margins)


# ---------------------------------------------------------------------------
# DCF input extraction
# ---------------------------------------------------------------------------

def extract_dcf_inputs_from_data(
    ticker: str,
    info: dict,
    fin: dict,
    risk_free_rate: float,
    company_type: str = "GENERAL",
) -> dict:
    """
    Build DCFInputs kwargs from info + structured financials dict.
    Keys prefixed '_' are display metadata — pop before DCFInputs(**).
    """
    rev_a = _annual(fin, "revenue_annual")
    fcf_a = _annual(fin, "fcf_annual")

    rev_ttm = _ttm(fin, "revenue_ttm") or (rev_a[0] if rev_a else 0) or info.get("totalRevenue") or 0
    fcf_ttm = _ttm(fin, "fcf_ttm") or (fcf_a[0] if fcf_a else None) or info.get("freeCashflow")
    if fcf_ttm is None:
        fcf_ttm = rev_ttm * 0.10

    # Historical revenue growth rates (oldest first, up to 5 years)
    growth_rates = []
    for i in range(min(5, len(rev_a) - 1)):
        if rev_a[i + 1] and rev_a[i + 1] > 0:
            growth_rates.append(rev_a[i] / rev_a[i + 1] - 1)
    growth_rates = list(reversed(growth_rates)) or [0.07]

    # Stable FCF margin
    metadata = {}
    if company_type == "CYCLICAL":
        fcf_margin_stable, yrs = compute_normalized_fcf_margin(fin, years=7)
        metadata["_margin_note"] = (
            f"FCF margin cycle-normalized over {yrs} yr(s) from EDGAR"
            if yrs > 0 else "Insufficient history — using 8% default"
        )
    else:
        margins = [f / r for f, r in zip(fcf_a, rev_a) if r and r > 0]
        fcf_margin_stable = float(np.mean(margins[:3])) if margins else 0.10

    # WACC
    beta       = max(0.3, min(info.get("beta", 1.0) or 1.0, 3.0))
    market_cap = info.get("marketCap", 0) or 0
    total_debt = fin.get("total_debt") or info.get("totalDebt") or 0
    int_a      = _annual(fin, "interest_expense_annual", 1)
    cost_of_debt = (int_a[0] / total_debt) if (int_a and total_debt > 0) else 0.05

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

    cash    = fin.get("cash") or info.get("totalCash") or 0
    net_debt = total_debt - cash

    shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0

    result = {
        "ticker":               ticker,
        "current_fcf":          fcf_ttm,
        "revenue_growth_rates": growth_rates,
        "fcf_margin_stable":    max(fcf_margin_stable, 0.01),
        "revenue_ttm":          rev_ttm,
        "wacc":                 wacc,
        "terminal_growth_rate": terminal_growth_rate,
        "projection_years":     10,
        "shares_outstanding":   shares,
        "net_debt":             net_debt,
    }
    result.update(metadata)
    return result
