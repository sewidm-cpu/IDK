"""
Data fetching — Yahoo Finance for price/market data, SEC EDGAR for financials.
"""
import yfinance as yf
from typing import Optional

from edgar import get_edgar_financials, yf_dfs_to_structured


def get_ticker_info(ticker: str) -> dict:
    """Yahoo Finance: price, market cap, beta, shares, sector/industry."""
    return yf.Ticker(ticker).info


def get_financials(ticker: str) -> dict:
    """
    Returns structured financial data dict (see edgar.py for schema).
    Tries EDGAR XBRL first; falls back to Yahoo Finance DataFrames.
    """
    edgar = get_edgar_financials(ticker)
    if edgar:
        return edgar

    # EDGAR failed (foreign issuer, new company, network error) — use Yahoo
    t = yf.Ticker(ticker)
    yf_dfs = {
        "income_stmt":        t.income_stmt,
        "balance_sheet":      t.balance_sheet,
        "cashflow":           t.cashflow,
        "quarterly_income":   t.quarterly_income_stmt,
        "quarterly_cashflow": t.quarterly_cashflow,
    }
    return yf_dfs_to_structured(yf_dfs)


def get_risk_free_rate() -> float:
    """10-year US Treasury yield from Yahoo Finance (^TNX)."""
    try:
        rate = yf.Ticker("^TNX").info.get("regularMarketPrice", 4.2)
        return rate / 100
    except Exception:
        return 0.042
