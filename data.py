"""
Data fetching from Yahoo Finance and SEC EDGAR.
"""
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional


EDGAR_HEADERS = {"User-Agent": "dcf-tool research@example.com"}
EDGAR_BASE = "https://data.sec.gov"


def get_ticker_info(ticker: str) -> dict:
    t = yf.Ticker(ticker)
    info = t.info
    return info


def get_financials(ticker: str) -> dict:
    """Pull income statement, balance sheet, cash flow from Yahoo Finance."""
    t = yf.Ticker(ticker)
    return {
        "income_stmt": t.income_stmt,
        "balance_sheet": t.balance_sheet,
        "cashflow": t.cashflow,
        "quarterly_income": t.quarterly_income_stmt,
        "quarterly_cashflow": t.quarterly_cashflow,
    }


def _get_edgar_cik(ticker: str) -> Optional[str]:
    url = "https://efts.sec.gov/LATEST/search-index?q=%22" + ticker + "%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
    # Use company tickers JSON
    r = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=EDGAR_HEADERS,
        timeout=10,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_edgar_facts(ticker: str) -> Optional[dict]:
    """Fetch company facts from SEC EDGAR (US-GAAP tags)."""
    cik = _get_edgar_cik(ticker)
    if not cik:
        return None
    url = f"{EDGAR_BASE}/api/xbrl/companyfacts/CIK{cik}.json"
    r = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
    if r.status_code != 200:
        return None
    return r.json()


def extract_annual_series(facts: dict, concept: str, unit: str = "USD") -> pd.Series:
    """Extract annual 10-K values for a US-GAAP concept from EDGAR facts."""
    try:
        entries = facts["facts"]["us-gaap"][concept]["units"][unit]
    except (KeyError, TypeError):
        return pd.Series(dtype=float)

    rows = [e for e in entries if e.get("form") == "10-K" and e.get("fp") == "FY"]
    if not rows:
        return pd.Series(dtype=float)

    df = pd.DataFrame(rows)[["end", "val"]].drop_duplicates("end")
    df["end"] = pd.to_datetime(df["end"])
    df = df.sort_values("end").set_index("end")["val"]
    return df


def get_risk_free_rate() -> float:
    """Approximate 10-year US Treasury yield from Yahoo Finance."""
    try:
        t = yf.Ticker("^TNX")
        rate = t.info.get("regularMarketPrice", 4.2)
        return rate / 100
    except Exception:
        return 0.042  # fallback ~4.2%
