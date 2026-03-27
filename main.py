#!/usr/bin/env python3
"""
DCF Valuation Tool
==================
Usage:
    python main.py AAPL MSFT CRM SNOW LMT XOM JPM PLD

Pulls data from Yahoo Finance + SEC EDGAR, classifies the company type,
and runs an appropriate analysis:
  SAAS      — Rule of 40, ARR, FCF margin + DCF
  CYCLICAL  — cycle-normalized FCF margin + DCF with wide sensitivity
  DEFENSE   — conservative terminal growth (2%) + DCF
  FINANCIAL — P/B, ROE, NIM, dividend yield  (DCF skipped)
  REIT      — FFO proxy, P/FFO, dividend yield (DCF skipped)
  BIOTECH   — burn rate, cash runway + DCF with warning
  GENERAL   — standard metrics + DCF
"""

import sys
import time
import math
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH = True
except ImportError:
    RICH = False

from data import get_ticker_info, get_financials, get_risk_free_rate
from dcf import DCFInputs, run_dcf, sensitivity_table
from metrics import (
    classify_company,
    compute_general_metrics,
    compute_saas_metrics,
    compute_financial_metrics,
    compute_reit_metrics,
    compute_biotech_metrics,
    compute_defense_flags,
    extract_dcf_inputs_from_data,
)


console = Console() if RICH else None

# Sensitivity grid ranges per company type
# wacc_delta: offsets from computed WACC; tgr: absolute terminal growth rates
SENSITIVITY_RANGES = {
    "CYCLICAL": {
        "wacc_delta": [-0.04, -0.02, 0.0, 0.02, 0.04],
        "tgr":        [0.005, 0.01, 0.02, 0.025, 0.03],
    },
    "DEFENSE": {
        "wacc_delta": [-0.02, -0.01, 0.0, 0.01, 0.02],
        "tgr":        [0.005, 0.01, 0.015, 0.02, 0.025],
    },
    "SAAS": {
        "wacc_delta": [-0.02, -0.01, 0.0, 0.01, 0.02],
        "tgr":        [0.02, 0.025, 0.03, 0.035, 0.04],
    },
    "BIOTECH": {
        "wacc_delta": [-0.03, -0.015, 0.0, 0.015, 0.03],
        "tgr":        [0.01, 0.02, 0.025, 0.03, 0.035],
    },
}
DEFAULT_RANGES = {
    "wacc_delta": [-0.02, -0.01, 0.0, 0.01, 0.02],
    "tgr":        [0.01, 0.02, 0.025, 0.03, 0.04],
}

DCF_SKIP_TYPES = {"FINANCIAL", "REIT"}

DCF_SKIP_MSG = {
    "FINANCIAL": (
        "DCF skipped — FCF-based DCF is not appropriate for financial companies. "
        "Use P/B, ROE, and dividend yield metrics above."
    ),
    "REIT": (
        "DCF skipped — REITs are valued on FFO/AFFO and dividend yield, not levered FCF."
    ),
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_dollars(v, precision: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    v = float(v)
    if abs(v) >= 1e12:  return f"${v/1e12:.{precision}f}T"
    if abs(v) >= 1e9:   return f"${v/1e9:.{precision}f}B"
    if abs(v) >= 1e6:   return f"${v/1e6:.{precision}f}M"
    return f"${v:,.{precision}f}"


def fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{float(v):.1f}%"


def fmt_mult(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{float(v):.1f}x"


def fmt_any(v) -> str:
    if v is None:           return "N/A"
    if isinstance(v, str):  return v
    if isinstance(v, float) and math.isnan(v): return "N/A"
    if isinstance(v, (int, float)):
        f = float(v)
        if abs(f) >= 1e6:   return fmt_dollars(f)
        if abs(f) < 1000:   return f"{f:.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# Print helpers (rich / plain fallback)
# ---------------------------------------------------------------------------

def _make_table(title: str):
    if RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY, show_header=True)
        t.add_column("Metric", style="bold")
        t.add_column("Value",  justify="right")
        return t
    return None


def _row(t, label: str, value: str):
    if RICH:
        t.add_row(label, value)
    else:
        print(f"  {label:<36} {value}")


def _print_table(t, header: str = ""):
    if RICH:
        console.print(t)
    else:
        if header:
            print(f"\n--- {header} ---")


def _warn(msg: str):
    if RICH:
        console.print(f"[bold yellow]  ⚠  {msg}[/bold yellow]")
    else:
        print(f"  [!] {msg}")


def _info_line(msg: str):
    if RICH:
        console.print(f"[dim]  {msg}[/dim]")
    else:
        print(f"  {msg}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, risk_free_rate: float):
    ticker = ticker.upper()

    if RICH:
        console.print(f"\n[bold cyan]{'='*62}[/bold cyan]")
        console.print(f"[bold white]  {ticker}[/bold white]")
        console.print(f"[bold cyan]{'='*62}[/bold cyan]")
    else:
        print(f"\n{'='*62}\n  {ticker}\n{'='*62}")

    # --- Fetch ---
    try:
        info = get_ticker_info(ticker)
        if not info or info.get("regularMarketPrice") is None:
            _warn(f"Could not fetch data for {ticker}. Check ticker symbol.")
            return
    except Exception as e:
        _warn(f"Error fetching {ticker}: {e}")
        return

    try:
        financials = get_financials(ticker)
    except Exception as e:
        _warn(f"Error fetching financials for {ticker}: {e}")
        return

    company_name  = info.get("longName") or info.get("shortName") or ticker
    sector        = info.get("sector",   "Unknown")
    industry      = info.get("industry", "Unknown")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    company_type  = classify_company(info)

    type_colors = {
        "SAAS": "green", "CYCLICAL": "yellow", "DEFENSE": "blue",
        "FINANCIAL": "magenta", "REIT": "cyan", "BIOTECH": "red", "GENERAL": "white",
    }
    color = type_colors.get(company_type, "white")

    if RICH:
        console.print(f"[bold]{company_name}[/bold]  |  {sector} / {industry}")
        console.print(
            f"Price: [yellow]{fmt_dollars(current_price)}[/yellow]  |  "
            f"Type: [{color}]{company_type}[/{color}]"
        )
    else:
        print(f"{company_name}  |  {sector} / {industry}")
        print(f"Price: {fmt_dollars(current_price)}  |  Type: {company_type}")

    # --- General metrics (all types) ---
    general = compute_general_metrics(info, financials)
    t = _make_table("Key Metrics")
    if not RICH:
        print("\n--- Key Metrics ---")

    _row(t, "Market Cap",         fmt_dollars(general["Market Cap"]))
    _row(t, "Enterprise Value",   fmt_dollars(general["Enterprise Value"]))
    _row(t, "TTM Revenue",        fmt_dollars(general["TTM Revenue"]))
    _row(t, "Revenue CAGR 3yr",   fmt_pct(general["Revenue CAGR 3yr %"]))
    _row(t, "Gross Margin",       fmt_pct(general["Gross Margin %"]))
    _row(t, "EBITDA Margin",      fmt_pct(general["EBITDA Margin %"]))
    _row(t, "Net Margin",         fmt_pct(general["Net Margin %"]))
    _row(t, "TTM FCF",            fmt_dollars(general["TTM FCF"]))
    _row(t, "FCF Yield",          fmt_pct(general["FCF Yield %"]))
    _row(t, "EV/Revenue",         fmt_mult(general["EV/Revenue"]))
    _row(t, "EV/EBITDA",          fmt_mult(general["EV/EBITDA"]))
    _row(t, "P/E (trailing)",     fmt_mult(general["P/E (trailing)"]))
    _row(t, "P/FCF",              fmt_mult(general["P/FCF"]))
    _row(t, "Net Debt",           fmt_dollars(general["Net Debt"]))

    # --- Type-specific metrics ---
    if company_type == "SAAS":
        m = compute_saas_metrics(info, financials)
        _row(t, "── SaaS ──",            "")
        _row(t, "ARR (approx.)",          fmt_dollars(m["ARR (approx)"]))
        _row(t, "Revenue Growth YoY",     fmt_pct(m["Revenue Growth YoY %"]))
        _row(t, "FCF Margin",             fmt_pct(m["FCF Margin %"]))
        r40 = m["Rule of 40"]
        r40_str = fmt_pct(r40) if r40 is not None else "N/A"
        if RICH and r40 is not None:
            r40_str = f"[{'green' if r40 >= 40 else 'yellow' if r40 >= 20 else 'red'}]{r40_str}[/{'green' if r40 >= 40 else 'yellow' if r40 >= 20 else 'red'}]"
        _row(t, "Rule of 40",             r40_str)
        _row(t, "NRR",                    str(m["NRR"]))

    elif company_type == "FINANCIAL":
        m = compute_financial_metrics(info, financials)
        _row(t, "── Banking/Financial ──", "")
        _row(t, "P/B Ratio",              fmt_mult(m["P/B Ratio"]))
        _row(t, "ROE",                    fmt_pct(m["ROE %"]))
        _row(t, "Net Interest Margin",    fmt_pct(m["Net Interest Margin % (approx)"]))
        _row(t, "P/E (forward)",          fmt_mult(m["P/E (forward)"]))
        _row(t, "Dividend Yield",         fmt_pct(m["Dividend Yield %"]))
        _row(t, "Tier 1 Capital",         m["Tier 1 Capital Ratio"])

    elif company_type == "REIT":
        m = compute_reit_metrics(info, financials)
        _row(t, "── REIT ──",             "")
        _row(t, "FFO (NI + D&A proxy)",   fmt_dollars(m["FFO proxy (NI + D&A)"]))
        _row(t, "P/FFO (proxy)",          fmt_mult(m["P/FFO (proxy)"]))
        _row(t, "Dividend Yield",         fmt_pct(m["Dividend Yield %"]))
        _row(t, "AFFO Note",              m["AFFO Note"])

    elif company_type == "BIOTECH":
        m = compute_biotech_metrics(info, financials)
        _row(t, "── Biotech ──",          "")
        _row(t, "Cash & Equivalents",     fmt_dollars(m["Cash & Equivalents"]))
        _row(t, "Quarterly Burn (avg)",   fmt_dollars(m["Avg Quarterly Cash Burn"]))
        runway = m["Cash Runway (quarters)"]
        runway_str = f"{runway:.1f} quarters" if runway is not None else "N/A"
        if RICH and runway is not None:
            runway_str = f"[{'green' if runway > 8 else 'yellow' if runway > 4 else 'red'}]{runway_str}[/{'green' if runway > 8 else 'yellow' if runway > 4 else 'red'}]"
        _row(t, "Cash Runway",            runway_str)

    elif company_type == "DEFENSE":
        flags = compute_defense_flags(info, financials)
        _row(t, "── Defense ──",          "")
        _row(t, "Backlog Note",           flags["Backlog Note"])

    elif company_type == "CYCLICAL":
        _row(t, "── Cyclical ──",         "")
        _row(t, "Note",                   "DCF uses cycle-normalized FCF margin (7yr avg)")

    _print_table(t, "Key Metrics")

    # --- DCF ---
    if company_type in DCF_SKIP_TYPES:
        if RICH:
            console.print(f"\n[yellow]  ⚠  {DCF_SKIP_MSG[company_type]}[/yellow]")
        else:
            print(f"\n  [!] {DCF_SKIP_MSG[company_type]}")
        _footer()
        return

    if RICH:
        console.print("\n[bold]Running DCF to Perpetuity...[/bold]")
    else:
        print("\n--- DCF to Perpetuity ---")

    if company_type == "BIOTECH":
        _warn("DCF reliability is LOW for pre-revenue/negative-FCF companies.")

    try:
        raw = extract_dcf_inputs_from_data(ticker, info, financials, risk_free_rate, company_type)

        # Pop display-only metadata before constructing DCFInputs
        meta = {k: raw.pop(k) for k in list(raw) if k.startswith("_")}
        margin_note = meta.get("_margin_note")
        if margin_note:
            _info_line(margin_note)

        dcf_inputs = DCFInputs(**raw)
        result     = run_dcf(dcf_inputs)

        upside = ((result.intrinsic_value_per_share / current_price) - 1) * 100 if current_price else None
        upside_str   = f"{upside:+.1f}%" if upside is not None else "N/A"
        upside_color = "green" if (upside or 0) > 0 else "red"

        if RICH:
            d = Table(title="DCF Results", box=box.SIMPLE_HEAVY)
            d.add_column("Item",  style="bold")
            d.add_column("Value", justify="right")
            d.add_row("WACC",                  fmt_pct(result.wacc * 100))
            d.add_row("Terminal Growth Rate",  fmt_pct(result.terminal_growth_rate * 100))
            d.add_row("PV of FCFs",            fmt_dollars(result.pv_fcfs))
            d.add_row("Terminal Value",        fmt_dollars(result.terminal_value))
            d.add_row("PV of Terminal Value",  fmt_dollars(result.pv_terminal))
            d.add_row("Equity Value",          fmt_dollars(result.intrinsic_value_equity))
            d.add_row("Intrinsic Value/Share", f"[bold]{fmt_dollars(result.intrinsic_value_per_share)}[/bold]")
            d.add_row("Current Price",         fmt_dollars(current_price))
            d.add_row("Upside / (Downside)",   f"[{upside_color}]{upside_str}[/{upside_color}]")
            d.add_row("MoS 20%",               fmt_dollars(result.margin_of_safety_20))
            d.add_row("MoS 30%",               fmt_dollars(result.margin_of_safety_30))
            console.print(d)
        else:
            print(f"  WACC:                    {fmt_pct(result.wacc * 100)}")
            print(f"  Terminal Growth Rate:    {fmt_pct(result.terminal_growth_rate * 100)}")
            print(f"  PV of FCFs:              {fmt_dollars(result.pv_fcfs)}")
            print(f"  Terminal Value:          {fmt_dollars(result.terminal_value)}")
            print(f"  PV of Terminal Value:    {fmt_dollars(result.pv_terminal)}")
            print(f"  Equity Value:            {fmt_dollars(result.intrinsic_value_equity)}")
            print(f"  Intrinsic Value/Share:   {fmt_dollars(result.intrinsic_value_per_share)}")
            print(f"  Current Price:           {fmt_dollars(current_price)}")
            print(f"  Upside / (Downside):     {upside_str}")
            print(f"  MoS 20%:                 {fmt_dollars(result.margin_of_safety_20)}")
            print(f"  MoS 30%:                 {fmt_dollars(result.margin_of_safety_30)}")

        # Sensitivity table
        ranges    = SENSITIVITY_RANGES.get(company_type, DEFAULT_RANGES)
        wacc_range = [dcf_inputs.wacc + d for d in ranges["wacc_delta"]]
        tgr_range  = ranges["tgr"]
        sens = sensitivity_table(dcf_inputs, wacc_range=wacc_range, tgr_range=tgr_range)

        if RICH:
            s = Table(title="Sensitivity: IV/Share  (rows=TGR, cols=WACC)", box=box.SIMPLE)
            s.add_column("TGR \\ WACC", style="bold")
            for col in sens.columns:
                s.add_column(col, justify="right")
            for idx, row in sens.iterrows():
                colored = []
                for v_raw in row.values:
                    try:
                        v = float(v_raw)
                        v_str = str(round(v, 2))
                        if current_price and abs(v - current_price) / current_price < 0.10:
                            colored.append(f"[yellow]{v_str}[/yellow]")
                        elif current_price and v > current_price * 1.1:
                            colored.append(f"[green]{v_str}[/green]")
                        else:
                            colored.append(f"[red]{v_str}[/red]")
                    except Exception:
                        colored.append("N/A")
                s.add_row(str(idx), *colored)
            console.print(s)
        else:
            print("\n  Sensitivity Table (Intrinsic Value per Share):")
            print(f"  TGR \\ WACC  |  " + "  |  ".join(sens.columns))
            print("  " + "-" * (len(sens.columns) * 12 + 14))
            for idx, row in sens.iterrows():
                vals = "  |  ".join(str(round(v, 2)) if str(v) != "nan" else "N/A" for v in row.values)
                print(f"  {str(idx):<11} |  {vals}")

    except Exception as e:
        _warn(f"DCF failed: {e}")

    _footer()


def _footer():
    if RICH:
        console.print("\n[dim]Data: Yahoo Finance. Always verify against SEC filings.[/dim]")
    else:
        print("\n  Data: Yahoo Finance. Always verify against SEC filings.\n")


def main():
    tickers = [t.upper() for t in sys.argv[1:] if not t.startswith("-")]

    if not tickers:
        print("Usage: python main.py TICKER1 TICKER2 ...")
        print("  e.g. python main.py AAPL MSFT CRM LMT XOM JPM PLD MRNA")
        sys.exit(1)

    if RICH:
        console.print(Panel.fit(
            "[bold cyan]DCF Valuation Tool[/bold cyan]\n"
            "Perpetuity DCF  |  Sector-Aware  |  SaaS / Cyclical / Defense / Financial / REIT / Biotech\n"
            "Data: Yahoo Finance + SEC EDGAR",
            border_style="cyan"
        ))

    risk_free_rate = get_risk_free_rate()
    if RICH:
        console.print(f"Risk-free rate (10yr UST): [yellow]{risk_free_rate:.2%}[/yellow]")
    else:
        print(f"Risk-free rate (10yr UST): {risk_free_rate:.2%}")

    for i, ticker in enumerate(tickers):
        analyze_ticker(ticker, risk_free_rate)
        if i < len(tickers) - 1:
            time.sleep(1)


if __name__ == "__main__":
    main()
