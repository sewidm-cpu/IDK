"""
SEC EDGAR XBRL data fetching with disk caching.

Pulls financial statement data directly from SEC filings via the
EDGAR XBRL API (free, no key required).

Key design:
  - Annual series from 10-K / 20-F
  - Individual quarter values derived from ~90-day period filter
  - TTM = sum of 4 most recent individual quarters
  - Balance sheet = most recent point-in-time entry
  - Concept fallback chains handle tag differences across companies
  - Disk cache (24hr TTL) avoids re-fetching large JSON blobs
"""

import json
import requests
from datetime import datetime
from pathlib import Path
from time import time
from typing import Optional

HEADERS      = {"User-Agent": "dcf-tool research@example.com"}
BASE         = "https://data.sec.gov"
TICKER_JSON  = "https://www.sec.gov/files/company_tickers.json"
CACHE_DIR    = Path(__file__).parent / ".edgar_cache"
CACHE_TTL    = 86_400          # 24 hours in seconds
ANNUAL_FORMS = {"10-K", "20-F"}
QUART_FORMS  = {"10-Q", "6-K"}

# ---------------------------------------------------------------------------
# Concept fallback chains
# ---------------------------------------------------------------------------
CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsForCapitalImprovements",
        "PaymentsToAcquireProductiveAssets",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "ShareBasedCompensationExpense",
        "AllocatedShareBasedCompensationExpense",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "gross_profit":     ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "da": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
        "Depreciation",
        "DepreciationAmortizationAndAccretionNet",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestAndDebtExpense",
        "InterestExpenseDebt",
    ],
    "wc_change": [
        "IncreaseDecreaseInOperatingCapital",
        "IncreaseDecreaseInOperatingLiabilities",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashCashEquivalentsAndShortTermInvestments",
    ],
    "lt_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermNotesPayable",
    ],
    "st_debt": [
        "DebtCurrent",
        "ShortTermBorrowings",
        "LongTermDebtCurrent",
    ],
    "rpo": [
        "RevenueRemainingPerformanceObligation",
    ],
    "deferred_rev": [
        "DeferredRevenueCurrent",
        "ContractWithCustomerLiabilityCurrent",
    ],
    "rd":  ["ResearchAndDevelopmentExpense"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
}


# ---------------------------------------------------------------------------
# Disk cache
# ---------------------------------------------------------------------------

def _cache_path(cik: str) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    return CACHE_DIR / f"{cik}.json"


def _load_cache(cik: str) -> Optional[dict]:
    p = _cache_path(cik)
    if not p.exists():
        return None
    if time() - p.stat().st_mtime > CACHE_TTL:
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_cache(cik: str, data: dict) -> None:
    try:
        _cache_path(cik).write_text(json.dumps(data))
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CIK lookup & facts fetch
# ---------------------------------------------------------------------------

def get_cik(ticker: str) -> Optional[str]:
    r = requests.get(TICKER_JSON, headers=HEADERS, timeout=10)
    r.raise_for_status()
    t = ticker.upper()
    for entry in r.json().values():
        if entry.get("ticker", "").upper() == t:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_facts(cik: str) -> dict:
    cached = _load_cache(cik)
    if cached is not None:
        return cached
    url = f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    data = r.json()
    _save_cache(cik, data)
    return data


# ---------------------------------------------------------------------------
# Entry extraction helpers
# ---------------------------------------------------------------------------

def _entries(facts: dict, concept: str) -> list:
    try:
        return facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return []


def _annual_series(entries: list, n: int = 7) -> list:
    """Up to n annual values, most-recent-first."""
    annual = [
        e for e in entries
        if e.get("form") in ANNUAL_FORMS and e.get("fp") == "FY"
        and "end" in e and "filed" in e
    ]
    by_end: dict[str, dict] = {}
    for e in annual:
        end = e["end"]
        if end not in by_end or e["filed"] > by_end[end]["filed"]:
            by_end[end] = e
    ordered = sorted(by_end.values(), key=lambda x: x["end"], reverse=True)
    return [float(e["val"]) for e in ordered[:n]]


def _quarter_entries(entries: list, n: int = 8) -> list:
    """Up to n individual-quarter entries (75–105 day period), most-recent-first."""
    quarters = []
    for e in entries:
        if e.get("form") not in QUART_FORMS:
            continue
        if "start" not in e or "end" not in e or "filed" not in e:
            continue
        try:
            days = (
                datetime.strptime(e["end"],   "%Y-%m-%d") -
                datetime.strptime(e["start"], "%Y-%m-%d")
            ).days
        except ValueError:
            continue
        if 75 <= days <= 105:
            quarters.append(e)
    by_end: dict[str, dict] = {}
    for e in quarters:
        end = e["end"]
        if end not in by_end or e["filed"] > by_end[end]["filed"]:
            by_end[end] = e
    return sorted(by_end.values(), key=lambda x: x["end"], reverse=True)[:n]


def _quarter_vals(entries: list, n: int = 8) -> list:
    return [float(e["val"]) for e in _quarter_entries(entries, n)]


def _ttm(entries: list) -> Optional[float]:
    qs = _quarter_vals(entries, 4)
    return sum(qs) if len(qs) >= 4 else None


def _latest_instant(entries: list) -> Optional[float]:
    valid = [
        e for e in entries
        if e.get("form") in ANNUAL_FORMS | QUART_FORMS and "end" in e
    ]
    if not valid:
        return None
    return float(max(valid, key=lambda x: x["end"])["val"])


# ---------------------------------------------------------------------------
# Concept resolver
# ---------------------------------------------------------------------------

def _resolve(facts: dict, key: str) -> list:
    """Return entries from the first concept tag in the fallback chain that has data."""
    for concept in CONCEPTS.get(key, []):
        e = _entries(facts, concept)
        if e:
            return e
    return []


def resolve_annual(facts: dict, key: str, n: int = 7) -> list:
    return _annual_series(_resolve(facts, key), n)


def resolve_ttm(facts: dict, key: str) -> Optional[float]:
    return _ttm(_resolve(facts, key))


def resolve_instant(facts: dict, key: str) -> Optional[float]:
    return _latest_instant(_resolve(facts, key))


def resolve_latest_quarter(facts: dict, key: str) -> Optional[float]:
    qs = _quarter_entries(_resolve(facts, key), 1)
    return float(qs[0]["val"]) if qs else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_edgar_financials(ticker: str) -> Optional[dict]:
    """
    Fetch and structure all financial data from EDGAR.
    Returns None if ticker not found or EDGAR request fails.
    """
    try:
        cik = get_cik(ticker)
        if not cik:
            return None
        facts = get_facts(cik)
    except Exception:
        return None

    # Annual series (most recent first)
    rev_a  = resolve_annual(facts, "revenue",          7)
    op_a   = resolve_annual(facts, "operating_cf",     7)
    cap_a  = [abs(x) for x in resolve_annual(facts, "capex", 7)]
    sbc_a  = resolve_annual(facts, "sbc",              7)
    ni_a   = resolve_annual(facts, "net_income",       7)
    gp_a   = resolve_annual(facts, "gross_profit",     7)
    oi_a   = resolve_annual(facts, "operating_income", 7)
    da_a   = resolve_annual(facts, "da",               7)
    int_a  = [abs(x) for x in resolve_annual(facts, "interest_expense", 7)]
    wc_a   = resolve_annual(facts, "wc_change",        7)
    rd_a   = resolve_annual(facts, "rd",               7)
    sga_a  = resolve_annual(facts, "sga",              7)

    # Derived annual series
    fcf_a    = [o - c          for o, c    in zip(op_a, cap_a)]
    ebitda_a = [o + d          for o, d    in zip(oi_a, da_a)]
    # Owner earnings = FCF - SBC  (SBC is real dilution cost, not cash)
    oe_a     = [f - s          for f, s    in zip(fcf_a, sbc_a)]

    # TTM values
    rev_ttm    = resolve_ttm(facts, "revenue")
    op_ttm     = resolve_ttm(facts, "operating_cf")
    cap_ttm_r  = resolve_ttm(facts, "capex")
    cap_ttm    = abs(cap_ttm_r) if cap_ttm_r is not None else None
    sbc_ttm    = resolve_ttm(facts, "sbc")
    gp_ttm     = resolve_ttm(facts, "gross_profit")
    ni_ttm     = resolve_ttm(facts, "net_income")

    fcf_ttm = None
    if op_ttm is not None and cap_ttm is not None:
        fcf_ttm = op_ttm - cap_ttm
    elif op_ttm is not None and cap_a and rev_a:
        avg_cap_pct = sum(c / r for c, r in zip(cap_a, rev_a) if r > 0) / max(len(cap_a), 1)
        fcf_ttm = op_ttm - (rev_ttm or (rev_a[0] if rev_a else 0)) * avg_cap_pct

    oe_ttm = None
    if fcf_ttm is not None and sbc_ttm is not None:
        oe_ttm = fcf_ttm - sbc_ttm

    # Most recent single quarter (for ARR)
    rev_latest_q = resolve_latest_quarter(facts, "revenue")

    # Balance sheet
    cash    = resolve_instant(facts, "cash")     or 0.0
    lt_debt = resolve_instant(facts, "lt_debt")  or 0.0
    st_debt = resolve_instant(facts, "st_debt")  or 0.0
    total_debt = lt_debt + st_debt

    # RPO (SaaS forward revenue commitment)
    rpo_latest  = resolve_instant(facts, "rpo")
    defer_rev   = resolve_instant(facts, "deferred_rev")

    return {
        "source":  "edgar",
        # Annual
        "revenue_annual":           rev_a,
        "op_cf_annual":             op_a,
        "capex_annual":             cap_a,
        "fcf_annual":               fcf_a,
        "sbc_annual":               sbc_a,
        "owner_earnings_annual":    oe_a,
        "net_income_annual":        ni_a,
        "gross_profit_annual":      gp_a,
        "operating_income_annual":  oi_a,
        "da_annual":                da_a,
        "ebitda_annual":            ebitda_a,
        "interest_expense_annual":  int_a,
        "wc_changes_annual":        wc_a,
        "rd_annual":                rd_a,
        "sga_annual":               sga_a,
        # TTM
        "revenue_ttm":              rev_ttm,
        "op_cf_ttm":                op_ttm,
        "capex_ttm":                cap_ttm,
        "fcf_ttm":                  fcf_ttm,
        "sbc_ttm":                  sbc_ttm,
        "owner_earnings_ttm":       oe_ttm,
        "gross_profit_ttm":         gp_ttm,
        "net_income_ttm":           ni_ttm,
        # Quarterly
        "revenue_latest_quarter":   rev_latest_q,
        # Balance sheet
        "cash":                     cash,
        "lt_debt":                  lt_debt,
        "st_debt":                  st_debt,
        "total_debt":               total_debt,
        "net_debt":                 total_debt - cash,
        # SaaS
        "rpo_latest":               rpo_latest,
        "deferred_rev_latest":      defer_rev,
    }


def yf_dfs_to_structured(financials_yf: dict) -> dict:
    """
    Convert Yahoo Finance DataFrames to the same structured dict format.
    Used as fallback when EDGAR is unavailable.
    """
    from metrics import _get_annual_series, _ttm_value

    income   = financials_yf.get("income_stmt")
    cashflow = financials_yf.get("cashflow")
    q_income = financials_yf.get("quarterly_income")
    q_cf     = financials_yf.get("quarterly_cashflow")
    balance  = financials_yf.get("balance_sheet")

    def _a(df, labels, n=7):
        return _get_annual_series(df, labels, n) if df is not None else []

    def _t(df, labels):
        return _ttm_value(df, labels) if df is not None else None

    rev_a  = _a(income,   ["total revenue", "revenue"])
    op_a   = _a(cashflow, ["operating cash flow"])
    cap_a  = [abs(x) for x in _a(cashflow, ["capital expenditure"])]
    sbc_a  = _a(cashflow, ["stock based compensation", "share based"])
    ni_a   = _a(income,   ["net income"])
    gp_a   = _a(income,   ["gross profit"])
    oi_a   = _a(income,   ["operating income", "ebit"])
    da_a   = _a(cashflow, ["depreciation"])
    int_a  = [abs(x) for x in _a(income, ["interest expense"])]
    wc_a   = _a(cashflow, ["change in working capital", "changes in operating"])
    rd_a   = _a(income,   ["research and development"])
    sga_a  = _a(income,   ["selling general", "general and admin"])

    fcf_a    = [o - c for o, c in zip(op_a, cap_a)]
    ebitda_a = [o + d for o, d in zip(oi_a, da_a)]
    oe_a     = [f - s for f, s in zip(fcf_a, sbc_a)]

    rev_ttm   = _t(q_income, ["total revenue", "revenue"])
    op_ttm    = _t(q_cf, ["operating cash flow"])
    cap_ttm_r = _t(q_cf, ["capital expenditure"])
    cap_ttm   = abs(cap_ttm_r) if cap_ttm_r is not None else None
    sbc_ttm   = _t(q_cf, ["stock based compensation", "share based"])
    gp_ttm    = _t(q_income, ["gross profit"])
    ni_ttm    = _t(q_income, ["net income"])

    fcf_ttm = (op_ttm - cap_ttm) if (op_ttm is not None and cap_ttm is not None) else None
    oe_ttm  = (fcf_ttm - sbc_ttm) if (fcf_ttm is not None and sbc_ttm is not None) else None

    # Balance sheet
    def _bs(df, labels):
        if df is None or df.empty:
            return 0.0
        for row in df.index:
            if any(l in str(row).lower() for l in labels):
                vals = df.loc[row].dropna()
                return float(vals.iloc[0]) if len(vals) else 0.0
        return 0.0

    cash       = _bs(balance, ["cash and cash equiv", "cash and short"])
    lt_debt    = _bs(balance, ["long term debt", "long-term debt"])
    st_debt    = _bs(balance, ["short term debt", "current portion", "debt current"])
    total_debt = lt_debt + st_debt

    return {
        "source":  "yahoo",
        "revenue_annual":           rev_a,
        "op_cf_annual":             op_a,
        "capex_annual":             cap_a,
        "fcf_annual":               fcf_a,
        "sbc_annual":               sbc_a,
        "owner_earnings_annual":    oe_a,
        "net_income_annual":        ni_a,
        "gross_profit_annual":      gp_a,
        "operating_income_annual":  oi_a,
        "da_annual":                da_a,
        "ebitda_annual":            ebitda_a,
        "interest_expense_annual":  int_a,
        "wc_changes_annual":        wc_a,
        "rd_annual":                rd_a,
        "sga_annual":               sga_a,
        "revenue_ttm":              rev_ttm,
        "op_cf_ttm":                op_ttm,
        "capex_ttm":                cap_ttm,
        "fcf_ttm":                  fcf_ttm,
        "sbc_ttm":                  sbc_ttm,
        "owner_earnings_ttm":       oe_ttm,
        "gross_profit_ttm":         gp_ttm,
        "net_income_ttm":           ni_ttm,
        "revenue_latest_quarter":   None,
        "cash":                     cash,
        "lt_debt":                  lt_debt,
        "st_debt":                  st_debt,
        "total_debt":               total_debt,
        "net_debt":                 total_debt - cash,
        "rpo_latest":               None,
        "deferred_rev_latest":      None,
    }
