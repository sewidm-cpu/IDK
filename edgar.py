"""
SEC EDGAR XBRL data fetching.

Pulls financial statement data directly from SEC filings via the
EDGAR XBRL API (free, no key required).

Key design:
  - Annual series from 10-K (or 20-F for foreign issuers)
  - Individual quarter values derived from period length (~90 days)
  - TTM = sum of 4 most recent individual quarters
  - Balance sheet = most recent point-in-time entry
  - Concept fallback chains resolve tag differences across companies
"""

import requests
from datetime import datetime
from typing import Optional

HEADERS  = {"User-Agent": "dcf-tool research@example.com"}
BASE     = "https://data.sec.gov"
TICKER_JSON = "https://www.sec.gov/files/company_tickers.json"

# ---------------------------------------------------------------------------
# Concept fallback chains
# Each key maps to a list of US-GAAP XBRL tags tried in order.
# ---------------------------------------------------------------------------
CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenueFromContractsWithCustomers",
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
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
    ],
    "gross_profit": [
        "GrossProfit",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
    ],
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
        "NotesPayableCurrent",
    ],
    "rd": [
        "ResearchAndDevelopmentExpense",
        "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
    "sga": [
        "SellingGeneralAndAdministrativeExpense",
        "GeneralAndAdministrativeExpense",
    ],
}

# Annual form types (US domestic + foreign private issuers)
ANNUAL_FORMS  = {"10-K", "20-F"}
QUARTER_FORMS = {"10-Q", "6-K"}


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
    url = f"{BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=HEADERS, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Entry extraction helpers
# ---------------------------------------------------------------------------

def _entries(facts: dict, concept: str) -> list:
    """Raw USD entries for a single US-GAAP concept."""
    try:
        return facts["facts"]["us-gaap"][concept]["units"]["USD"]
    except KeyError:
        return []


def _annual_series(entries: list, n: int = 7) -> list:
    """
    Up to n annual values, most-recent-first.
    Filters for 10-K/20-F FY filings; deduplicates by fiscal-year-end date
    (keeps the most recently *filed* amendment if restated).
    """
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


def _quarter_series(entries: list, n: int = 8) -> list:
    """
    Up to n individual-quarter values, most-recent-first.
    Selects entries where the reporting period is 75–105 days
    (i.e. standalone quarter, not YTD).
    """
    quarters = []
    for e in entries:
        if e.get("form") not in QUARTER_FORMS:
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
    ordered = sorted(by_end.values(), key=lambda x: x["end"], reverse=True)
    return [float(e["val"]) for e in ordered[:n]]


def _latest_instant(entries: list) -> Optional[float]:
    """
    Most recent point-in-time value (balance sheet items).
    Entries that have no 'start' key or where start == end are instants.
    """
    instants = [
        e for e in entries
        if e.get("form") in ANNUAL_FORMS | QUARTER_FORMS and "end" in e
        and ("start" not in e or e["start"] == e["end"])
    ]
    if not instants:
        # Fall back to any recent filing entry
        instants = [e for e in entries if e.get("form") in ANNUAL_FORMS | QUARTER_FORMS and "end" in e]
    if not instants:
        return None
    return float(max(instants, key=lambda x: x["end"])["val"])


# ---------------------------------------------------------------------------
# Concept resolver — applies fallback chain
# ---------------------------------------------------------------------------

def resolve_annual(facts: dict, key: str, n: int = 7) -> list:
    for concept in CONCEPTS.get(key, []):
        vals = _annual_series(_entries(facts, concept), n)
        if vals:
            return vals
    return []


def resolve_quarters(facts: dict, key: str, n: int = 8) -> list:
    for concept in CONCEPTS.get(key, []):
        vals = _quarter_series(_entries(facts, concept), n)
        if vals:
            return vals
    return []


def resolve_instant(facts: dict, key: str) -> Optional[float]:
    for concept in CONCEPTS.get(key, []):
        val = _latest_instant(_entries(facts, concept))
        if val is not None:
            return val
    return None


def compute_ttm(facts: dict, key: str) -> Optional[float]:
    """
    TTM = sum of 4 most recent individual quarters.
    Falls back to most recent annual if fewer than 4 quarters available.
    """
    qs = resolve_quarters(facts, key, n=4)
    if len(qs) >= 4:
        return sum(qs)
    annual = resolve_annual(facts, key, n=1)
    return annual[0] if annual else None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_edgar_financials(ticker: str) -> Optional[dict]:
    """
    Fetch and structure all financial data for a ticker from EDGAR.
    Returns None if ticker not found or EDGAR request fails.
    """
    try:
        cik = get_cik(ticker)
        if not cik:
            return None
        facts = get_facts(cik)
    except Exception:
        return None

    # --- Annual series (most recent first) ---
    rev_a  = resolve_annual(facts, "revenue",          7)
    op_a   = resolve_annual(facts, "operating_cf",     7)
    cap_a  = [abs(x) for x in resolve_annual(facts, "capex", 7)]
    ni_a   = resolve_annual(facts, "net_income",       7)
    gp_a   = resolve_annual(facts, "gross_profit",     7)
    oi_a   = resolve_annual(facts, "operating_income", 7)
    da_a   = resolve_annual(facts, "da",               7)
    int_a  = [abs(x) for x in resolve_annual(facts, "interest_expense", 7)]
    rd_a   = resolve_annual(facts, "rd",               7)
    sga_a  = resolve_annual(facts, "sga",              7)

    # FCF annual = operating CF − CapEx
    fcf_a = [o - c for o, c in zip(op_a, cap_a)]

    # EBITDA annual = operating income + D&A
    ebitda_a = [o + d for o, d in zip(oi_a, da_a)]

    # --- TTM values ---
    rev_ttm   = compute_ttm(facts, "revenue")
    op_ttm    = compute_ttm(facts, "operating_cf")
    cap_ttm   = compute_ttm(facts, "capex")
    if cap_ttm is not None:
        cap_ttm = abs(cap_ttm)

    fcf_ttm = None
    if op_ttm is not None and cap_ttm is not None:
        fcf_ttm = op_ttm - cap_ttm
    elif op_ttm is not None and cap_a and rev_a:
        # Approximate CapEx from avg historical ratio
        avg_cap_pct = sum(c / r for c, r in zip(cap_a, rev_a) if r > 0) / max(len(cap_a), 1)
        fcf_ttm = op_ttm - (rev_ttm or rev_a[0]) * avg_cap_pct

    gp_ttm = compute_ttm(facts, "gross_profit")
    ni_ttm = compute_ttm(facts, "net_income")

    # --- Balance sheet (most recent point-in-time) ---
    cash    = resolve_instant(facts, "cash")  or 0.0
    lt_debt = resolve_instant(facts, "lt_debt") or 0.0
    st_debt = resolve_instant(facts, "st_debt") or 0.0
    total_debt = lt_debt + st_debt

    return {
        "source":  "edgar",
        # Annual series
        "revenue_annual":           rev_a,
        "op_cf_annual":             op_a,
        "capex_annual":             cap_a,
        "fcf_annual":               fcf_a,
        "net_income_annual":        ni_a,
        "gross_profit_annual":      gp_a,
        "operating_income_annual":  oi_a,
        "da_annual":                da_a,
        "ebitda_annual":            ebitda_a,
        "interest_expense_annual":  int_a,
        "rd_annual":                rd_a,
        "sga_annual":               sga_a,
        # TTM
        "revenue_ttm":              rev_ttm,
        "op_cf_ttm":                op_ttm,
        "capex_ttm":                cap_ttm,
        "fcf_ttm":                  fcf_ttm,
        "gross_profit_ttm":         gp_ttm,
        "net_income_ttm":           ni_ttm,
        # Balance sheet
        "cash":                     cash,
        "lt_debt":                  lt_debt,
        "st_debt":                  st_debt,
        "total_debt":               total_debt,
        "net_debt":                 total_debt - cash,
    }


def yf_dfs_to_structured(financials_yf: dict) -> dict:
    """
    Convert Yahoo Finance DataFrames to the same structured dict format
    as get_edgar_financials(), so metrics.py only ever sees one format.
    Used as fallback when EDGAR is unavailable.
    """
    from metrics import _get_annual_series, _ttm_value, _compute_ttm_fcf

    income   = financials_yf.get("income_stmt")
    cashflow = financials_yf.get("cashflow")
    q_income = financials_yf.get("quarterly_income")
    q_cf     = financials_yf.get("quarterly_cashflow")
    balance  = financials_yf.get("balance_sheet")

    def _a(df, labels, n=7):
        return _get_annual_series(df, labels, n) if df is not None else []

    def _ttm(df, labels):
        return _ttm_value(df, labels) if df is not None else None

    rev_a  = _a(income,   ["total revenue", "revenue"])
    op_a   = _a(cashflow, ["operating cash flow"])
    cap_a  = [abs(x) for x in _a(cashflow, ["capital expenditure"])]
    ni_a   = _a(income,   ["net income"])
    gp_a   = _a(income,   ["gross profit"])
    oi_a   = _a(income,   ["operating income", "ebit"])
    da_a   = _a(cashflow, ["depreciation"])
    int_a  = [abs(x) for x in _a(income, ["interest expense"])]
    rd_a   = _a(income,   ["research and development"])
    sga_a  = _a(income,   ["selling general", "general and admin"])

    fcf_a    = [o - c for o, c in zip(op_a, cap_a)]
    ebitda_a = [o + d for o, d in zip(oi_a, da_a)]

    rev_ttm = _ttm(q_income, ["total revenue", "revenue"])
    op_ttm  = _ttm(q_cf, ["operating cash flow"])
    cap_ttm_raw = _ttm(q_cf, ["capital expenditure"])
    cap_ttm = abs(cap_ttm_raw) if cap_ttm_raw is not None else None

    fcf_ttm = None
    if op_ttm is not None and cap_ttm is not None:
        fcf_ttm = op_ttm - cap_ttm

    gp_ttm = _ttm(q_income, ["gross profit"])
    ni_ttm = _ttm(q_income, ["net income"])

    # Balance sheet
    def _bs(df, labels):
        if df is None or df.empty:
            return 0.0
        for row in df.index:
            if any(l in str(row).lower() for l in labels):
                vals = df.loc[row].dropna()
                return float(vals.iloc[0]) if len(vals) else 0.0
        return 0.0

    cash       = _bs(balance, ["cash and cash equivalents", "cash and short"])
    lt_debt    = _bs(balance, ["long term debt", "long-term debt"])
    st_debt    = _bs(balance, ["short term debt", "current portion", "debt current"])
    total_debt = lt_debt + st_debt

    return {
        "source":  "yahoo",
        "revenue_annual":           rev_a,
        "op_cf_annual":             op_a,
        "capex_annual":             cap_a,
        "fcf_annual":               fcf_a,
        "net_income_annual":        ni_a,
        "gross_profit_annual":      gp_a,
        "operating_income_annual":  oi_a,
        "da_annual":                da_a,
        "ebitda_annual":            ebitda_a,
        "interest_expense_annual":  int_a,
        "rd_annual":                rd_a,
        "sga_annual":               sga_a,
        "revenue_ttm":              rev_ttm,
        "op_cf_ttm":                op_ttm,
        "capex_ttm":                cap_ttm,
        "fcf_ttm":                  fcf_ttm,
        "gross_profit_ttm":         gp_ttm,
        "net_income_ttm":           ni_ttm,
        "cash":                     cash,
        "lt_debt":                  lt_debt,
        "st_debt":                  st_debt,
        "total_debt":               total_debt,
        "net_debt":                 total_debt - cash,
    }
