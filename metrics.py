"""
Financial metrics from the structured financials dict (edgar.py schema).

All compute_* functions accept:
  info  — Yahoo Finance info dict (price, market cap, beta, sector, etc.)
  fin   — structured dict from edgar.get_edgar_financials() or yf_dfs_to_structured()
"""

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers retained for edgar.yf_dfs_to_structured fallback
# ---------------------------------------------------------------------------

def _get_annual_series(df, label_matches: list, n: int = 7) -> list:
    if df is None or df.empty:
        return []
    for row_label in df.index:
        if any(m.lower() in str(row_label).lower() for m in label_matches):
            return [float(v) for v in df.loc[row_label].dropna()
                    .sort_index(ascending=False).values[:n]]
    return []


def _ttm_value(q_df, label_matches: list) -> Optional[float]:
    if q_df is None or q_df.empty:
        return None
    for row_label in q_df.index:
        if any(m.lower() in str(row_label).lower() for m in label_matches):
            vals = q_df.loc[row_label].dropna().sort_index(ascending=False).head(4)
            if len(vals) >= 2:
                return float(vals.sum())
    return None


# ---------------------------------------------------------------------------
# Structured dict accessors
# ---------------------------------------------------------------------------

def _a(fin: dict, key: str, n: int = 7) -> list:
    return (fin.get(key) or [])[:n]


def _v(fin: dict, key: str) -> Optional[float]:
    v = fin.get(key)
    return float(v) if v is not None else None


def _pct(num, denom) -> Optional[float]:
    return (num / denom * 100) if (num is not None and denom and denom != 0) else None


def _cagr(series: list, years: int) -> Optional[float]:
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
    for kw in ["software", "saas", "cloud", "application software",
               "software-infrastructure", "internet content"]:
        if kw in industry or kw in sector:
            return True
    return "subscription" in biz and "software" in biz


def classify_company(info: dict) -> str:
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
# General metrics
# ---------------------------------------------------------------------------

def compute_general_metrics(info: dict, fin: dict) -> dict:
    market_cap = info.get("marketCap")
    ev         = info.get("enterpriseValue")

    rev_a   = _a(fin, "revenue_annual")
    fcf_a   = _a(fin, "fcf_annual")
    oe_a    = _a(fin, "owner_earnings_annual")
    gp_a    = _a(fin, "gross_profit_annual")
    ni_a    = _a(fin, "net_income_annual")
    ebi_a   = _a(fin, "ebitda_annual")
    sbc_a   = _a(fin, "sbc_annual")
    wc_a    = _a(fin, "wc_changes_annual")

    rev_ttm = _v(fin, "revenue_ttm")  or (rev_a[0] if rev_a else None)
    fcf_ttm = _v(fin, "fcf_ttm")      or (fcf_a[0] if fcf_a else None)
    oe_ttm  = _v(fin, "owner_earnings_ttm") or (oe_a[0] if oe_a else None)
    gp_ttm  = _v(fin, "gross_profit_ttm")   or (gp_a[0] if gp_a else None)
    ni_ttm  = _v(fin, "net_income_ttm")     or (ni_a[0] if ni_a else None)
    sbc_ttm = _v(fin, "sbc_ttm")            or (sbc_a[0] if sbc_a else None)

    ebitda_ttm = ebi_a[0] if ebi_a else None

    # Info-level overrides for margins (Yahoo pre-computed ratios)
    gm  = info.get("grossMargins")
    nm  = info.get("profitMargins")
    gross_margin  = _pct(gp_ttm, rev_ttm)  or (gm  * 100 if gm  else None)
    net_margin    = _pct(ni_ttm, rev_ttm)  or (nm  * 100 if nm  else None)
    ebitda_margin = _pct(ebitda_ttm, rev_ttm)

    # Avg working capital change as % of revenue (flag if FCF is WC-distorted)
    avg_wc_pct = None
    if wc_a and rev_a:
        pairs  = [(w, r) for w, r in zip(wc_a, rev_a) if r > 0]
        if pairs:
            avg_wc_pct = np.mean([w / r * 100 for w, r in pairs])

    total_debt = fin.get("total_debt") or info.get("totalDebt") or 0
    cash       = fin.get("cash")       or info.get("totalCash")  or 0
    net_debt   = fin.get("net_debt",   total_debt - cash)

    return {
        "Market Cap":            market_cap,
        "Enterprise Value":      ev,
        "TTM Revenue":           rev_ttm,
        "Revenue CAGR 3yr %":    _cagr(rev_a, 3),
        "Gross Margin %":        gross_margin,
        "EBITDA Margin %":       ebitda_margin,
        "Net Margin %":          net_margin,
        "TTM FCF":               fcf_ttm,
        "TTM Owner Earnings":    oe_ttm,
        "SBC (TTM)":             sbc_ttm,
        "SBC % Revenue":         _pct(sbc_ttm, rev_ttm),
        "FCF Yield %":           _pct(fcf_ttm, market_cap),
        "Owner Earnings Yield %":_pct(oe_ttm,  market_cap),
        "Avg WC Change % Rev":   avg_wc_pct,
        "EV/Revenue":            (ev / rev_ttm) if (ev and rev_ttm) else None,
        "EV/EBITDA":             info.get("enterpriseToEbitda"),
        "P/E (trailing)":        info.get("trailingPE"),
        "P/FCF":                 (market_cap / fcf_ttm) if (fcf_ttm and fcf_ttm > 0 and market_cap) else None,
        "P/Owner Earnings":      (market_cap / oe_ttm)  if (oe_ttm  and oe_ttm  > 0 and market_cap) else None,
        "Total Debt":            total_debt,
        "Cash & Equivalents":    cash,
        "Net Debt":              net_debt,
        "Source":                fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# SaaS metrics
# ---------------------------------------------------------------------------

def compute_saas_metrics(info: dict, fin: dict) -> dict:
    rev_a  = _a(fin, "revenue_annual")
    rev_ttm = _v(fin, "revenue_ttm") or (rev_a[0] if rev_a else None) or info.get("totalRevenue")
    fcf_ttm = _v(fin, "fcf_ttm")     or info.get("freeCashflow")
    oe_ttm  = _v(fin, "owner_earnings_ttm")
    sbc_ttm = _v(fin, "sbc_ttm")

    # ARR: latest single quarter * 4 (best proxy from EDGAR)
    latest_q = _v(fin, "revenue_latest_quarter")
    arr = (latest_q * 4) if latest_q else (rev_ttm * ((1 + info.get("revenueGrowth", 0)) ** 0.25) if rev_ttm else None)

    # Revenue growth: Yahoo pre-computed YoY % most reliable
    rev_growth = info.get("revenueGrowth")
    if rev_growth is not None:
        rev_growth = rev_growth * 100
    elif len(rev_a) >= 2 and rev_a[1]:
        rev_growth = (rev_a[0] / rev_a[1] - 1) * 100

    fcf_margin = _pct(fcf_ttm, rev_ttm)
    oe_margin  = _pct(oe_ttm,  rev_ttm)

    rule_of_40    = (rev_growth + fcf_margin) if (rev_growth is not None and fcf_margin is not None) else None
    rule_of_40_oe = (rev_growth + oe_margin)  if (rev_growth is not None and oe_margin  is not None) else None

    gm = info.get("grossMargins")
    gross_margin = (gm * 100) if gm else _pct(
        _v(fin, "gross_profit_ttm") or (_a(fin, "gross_profit_annual") or [None])[0],
        rev_ttm
    )

    # RPO: Remaining Performance Obligations (contracted future revenue)
    rpo   = _v(fin, "rpo_latest")
    defer = _v(fin, "deferred_rev_latest")

    return {
        "TTM Revenue":            rev_ttm,
        "ARR (approx)":           arr,
        "Revenue Growth YoY %":   rev_growth,
        "TTM FCF":                fcf_ttm,
        "TTM Owner Earnings":     oe_ttm,
        "SBC (TTM)":              sbc_ttm,
        "SBC % Revenue":          _pct(sbc_ttm, rev_ttm),
        "FCF Margin %":           fcf_margin,
        "Owner Earnings Margin %":oe_margin,
        "Rule of 40 (FCF)":       rule_of_40,
        "Rule of 40 (OE)":        rule_of_40_oe,
        "Gross Margin %":         gross_margin,
        "RPO (contracted rev)":   rpo,
        "Deferred Revenue":       defer,
        "NRR":                    "N/A (requires subscription cohort data)",
    }


# ---------------------------------------------------------------------------
# Financial sector
# ---------------------------------------------------------------------------

def compute_financial_metrics(info: dict, fin: dict) -> dict:
    roe = info.get("returnOnEquity")
    div = info.get("dividendYield")
    return {
        "P/B Ratio":                      info.get("priceToBook"),
        "ROE %":                          (roe * 100) if roe else None,
        "Dividend Yield %":               (div * 100) if div else None,
        "P/E (trailing)":                 info.get("trailingPE"),
        "P/E (forward)":                  info.get("forwardPE"),
        "Net Interest Margin % (approx)": None,
        "Tier 1 Capital Ratio":           "N/A — check latest 10-K",
        "Source":                         fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# REIT metrics
# ---------------------------------------------------------------------------

def compute_reit_metrics(info: dict, fin: dict) -> dict:
    ni_a = _a(fin, "net_income_annual", 1)
    da_a = _a(fin, "da_annual", 1)
    ffo  = (ni_a[0] + abs(da_a[0])) if (ni_a and da_a) else None
    mc   = info.get("marketCap")
    div  = info.get("dividendYield")
    return {
        "Dividend Yield %":     (div * 100) if div else None,
        "P/E (trailing)":       info.get("trailingPE"),
        "FFO proxy (NI+D&A)":   ffo,
        "P/FFO (proxy)":        (mc / ffo) if (ffo and ffo > 0 and mc) else None,
        "AFFO Note":            "AFFO requires maintenance capex split — not in public feeds",
        "Source":               fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# Biotech / pre-revenue
# ---------------------------------------------------------------------------

def compute_biotech_metrics(info: dict, fin: dict) -> dict:
    cash    = fin.get("cash") or info.get("totalCash") or 0
    op_ttm  = _v(fin, "op_cf_ttm")
    qburn   = (abs(op_ttm) / 4) if (op_ttm is not None and op_ttm < 0) else None
    runway  = (cash / qburn) if (qburn and qburn > 0 and cash > 0) else None
    return {
        "Cash & Equivalents":      cash,
        "Avg Quarterly Cash Burn": qburn,
        "Cash Runway (quarters)":  runway,
        "DCF Warning":             "DCF not meaningful — negative/near-zero FCF.",
        "Source":                  fin.get("source", "unknown"),
    }


# ---------------------------------------------------------------------------
# Defense flags
# ---------------------------------------------------------------------------

def compute_defense_flags(info: dict, fin: dict) -> dict:
    return {
        "Backlog Note":              "Revenue is backlog-driven. Terminal growth capped at 2.0%.",
        "_terminal_growth_override": 0.020,
    }


# ---------------------------------------------------------------------------
# Cyclical: cycle-normalized FCF margin
# ---------------------------------------------------------------------------

def compute_normalized_fcf_margin(fin: dict, years: int = 7) -> tuple:
    fcf_a = _a(fin, "fcf_annual", years)
    rev_a = _a(fin, "revenue_annual", years)
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
    Build DCFInputs kwargs. Keys prefixed '_' are display metadata — pop before DCFInputs(**).
    """
    rev_a = _a(fin, "revenue_annual")
    fcf_a = _a(fin, "fcf_annual")
    sbc_a = _a(fin, "sbc_annual")
    ebi_a = _a(fin, "ebitda_annual")
    rev_a_full = _a(fin, "revenue_annual", 7)

    rev_ttm = _v(fin, "revenue_ttm") or (rev_a[0] if rev_a else 0) or info.get("totalRevenue") or 0
    fcf_ttm = _v(fin, "fcf_ttm")     or (fcf_a[0] if fcf_a else None) or info.get("freeCashflow")
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

    # SBC as % of revenue (for owner-earnings DCF)
    sbc_pct = 0.0
    if sbc_a and rev_a_full:
        sbc_pcts = [s / r for s, r in zip(sbc_a, rev_a_full) if r > 0]
        if sbc_pcts:
            sbc_pct = float(np.mean(sbc_pcts[:3]))

    # Stable EBITDA margin (for exit multiple TV)
    ebitda_margin_stable = 0.0
    if ebi_a and rev_a:
        em_list = [e / r for e, r in zip(ebi_a, rev_a) if r > 0]
        ebitda_margin_stable = float(np.mean(em_list[:3])) if em_list else 0.0
    if ebitda_margin_stable <= 0 and info.get("ebitda") and rev_ttm:
        ebitda_margin_stable = info["ebitda"] / rev_ttm

    # Exit multiple from sector defaults
    from dcf import EXIT_MULTIPLES
    exit_multiple = EXIT_MULTIPLES.get(company_type, 12.0)

    # WACC
    beta       = max(0.3, min(info.get("beta", 1.0) or 1.0, 3.0))
    market_cap = info.get("marketCap", 0) or 0
    total_debt = fin.get("total_debt") or info.get("totalDebt") or 0
    int_a      = _a(fin, "interest_expense_annual", 1)
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

    cash     = fin.get("cash") or info.get("totalCash") or 0
    net_debt = total_debt - cash
    shares   = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding") or 0

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
        "sbc_pct_revenue":      sbc_pct,
        "ebitda_margin_stable": max(ebitda_margin_stable, 0.0),
        "exit_ebitda_multiple": exit_multiple,
    }
    result.update(metadata)
    return result
