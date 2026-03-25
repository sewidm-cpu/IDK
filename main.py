#!/usr/bin/env python3
"""
DCF Valuation Tool
==================
Usage:
    python main.py AAPL MSFT CRM SNOW

Pulls data from Yahoo Finance + SEC EDGAR, runs a DCF to perpetuity,
and prints SaaS (Rule of 40, ARR, NRR) or general metrics depending
on company type.
"""

import sys
import time
import math
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    RICH = True
except ImportError:
    RICH = False

from data import get_ticker_info, get_financials, get_risk_free_rate
from dcf import DCFInputs, run_dcf, sensitivity_table
from metrics import (
    compute_saas_metrics,
    compute_general_metrics,
    is_saas_company,
    extract_dcf_inputs_from_data,
)


console = Console() if RICH else None


def fmt_dollars(v: Optional[float], precision: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    if abs(v) >= 1e12:
        return f"${v/1e12:.{precision}f}T"
    if abs(v) >= 1e9:
        return f"${v/1e9:.{precision}f}B"
    if abs(v) >= 1e6:
        return f"${v/1e6:.{precision}f}M"
    return f"${v:,.{precision}f}"


def fmt_pct(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.1f}%"


def fmt_mult(v: Optional[float]) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "N/A"
    return f"{v:.1f}x"


def fmt_val(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, str):
        return v
    if isinstance(v, float):
        if math.isnan(v):
            return "N/A"
        if abs(v) >= 1e9:
            return fmt_dollars(v)
        if abs(v) < 100:
            return f"{v:.2f}"
    return str(v)


def print_separator(char="-", width=70):
    print(char * width)


def analyze_ticker(ticker: str, risk_free_rate: float):
    ticker = ticker.upper()

    if RICH:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold white]  Analyzing: {ticker}[/bold white]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    else:
        print(f"\n{'='*60}")
        print(f"  Analyzing: {ticker}")
        print(f"{'='*60}")

    # --- Fetch data ---
    try:
        info = get_ticker_info(ticker)
        if not info or info.get("regularMarketPrice") is None:
            print(f"  [!] Could not fetch data for {ticker}. Check ticker symbol.")
            return
    except Exception as e:
        print(f"  [!] Error fetching {ticker}: {e}")
        return

    try:
        financials = get_financials(ticker)
    except Exception as e:
        print(f"  [!] Error fetching financials for {ticker}: {e}")
        return

    company_name = info.get("longName") or info.get("shortName") or ticker
    sector = info.get("sector", "Unknown")
    industry = info.get("industry", "Unknown")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")

    saas = is_saas_company(info)

    if RICH:
        console.print(f"[bold]{company_name}[/bold]  |  {sector} / {industry}")
        console.print(f"Current Price: [yellow]{fmt_dollars(current_price, 2)}[/yellow]  |  "
                      f"Type: [green]{'SaaS/Software' if saas else 'General'}[/green]")
    else:
        print(f"{company_name}  |  {sector} / {industry}")
        print(f"Current Price: {fmt_dollars(current_price, 2)}  |  Type: {'SaaS/Software' if saas else 'General'}")

    # --- Metrics ---
    general = compute_general_metrics(info, financials)

    if RICH:
        t = Table(title="Key Metrics", box=box.SIMPLE_HEAVY, show_header=True)
        t.add_column("Metric", style="bold")
        t.add_column("Value", justify="right")
    else:
        print("\n--- Key Metrics ---")

    def add_row(label, value):
        if RICH:
            t.add_row(label, value)
        else:
            print(f"  {label:<28} {value}")

    add_row("Market Cap",          fmt_dollars(general["Market Cap"]))
    add_row("Enterprise Value",    fmt_dollars(general["Enterprise Value"]))
    add_row("TTM Revenue",         fmt_dollars(general["TTM Revenue"]))
    add_row("Revenue CAGR 3yr",    fmt_pct(general["Revenue CAGR 3yr %"]))
    add_row("Gross Margin",        fmt_pct(general["Gross Margin %"]))
    add_row("EBITDA Margin",       fmt_pct(general["EBITDA Margin %"]))
    add_row("Net Margin",          fmt_pct(general["Net Margin %"]))
    add_row("TTM FCF",             fmt_dollars(general["TTM FCF"]))
    add_row("FCF Yield",           fmt_pct(general["FCF Yield %"]))
    add_row("EV/Revenue",          fmt_mult(general["EV/Revenue"]))
    add_row("EV/EBITDA",           fmt_mult(general["EV/EBITDA"]))
    add_row("P/E (trailing)",      fmt_mult(general["P/E (trailing)"]))
    add_row("P/FCF",               fmt_mult(general["P/FCF"]))
    add_row("Net Debt",            fmt_dollars(general["Net Debt"]))

    if saas:
        saas_m = compute_saas_metrics(info, financials)
        add_row("─── SaaS Metrics ───", "")
        add_row("ARR (approx.)",       fmt_dollars(saas_m["ARR (approx.)"]))
        add_row("Revenue Growth YoY",  fmt_pct(saas_m["Revenue Growth YoY %"]))
        add_row("FCF Margin",          fmt_pct(saas_m["FCF Margin %"]))
        add_row("Rule of 40",          fmt_pct(saas_m["Rule of 40"]) if saas_m["Rule of 40"] is not None else "N/A")
        add_row("NRR",                 str(saas_m["NRR"]))
    else:
        # General company extras already covered above
        pass

    if RICH:
        console.print(t)
    else:
        print()

    # --- DCF ---
    if RICH:
        console.print("\n[bold]Running DCF to Perpetuity...[/bold]")
    else:
        print("\n--- DCF to Perpetuity ---")

    try:
        dcf_in_dict = extract_dcf_inputs_from_data(ticker, info, financials, risk_free_rate)
        dcf_inputs = DCFInputs(**dcf_in_dict)
        result = run_dcf(dcf_inputs)

        upside = ((result.intrinsic_value_per_share / current_price) - 1) * 100 if current_price else None
        upside_str = f"{upside:+.1f}%" if upside is not None else "N/A"
        upside_color = "green" if (upside or 0) > 0 else "red"

        if RICH:
            d = Table(title="DCF Results", box=box.SIMPLE_HEAVY)
            d.add_column("Item", style="bold")
            d.add_column("Value", justify="right")
            d.add_row("WACC",                   fmt_pct(result.wacc * 100))
            d.add_row("Terminal Growth Rate",    fmt_pct(result.terminal_growth_rate * 100))
            d.add_row("PV of FCFs",             fmt_dollars(result.pv_fcfs))
            d.add_row("Terminal Value",          fmt_dollars(result.terminal_value))
            d.add_row("PV of Terminal Value",    fmt_dollars(result.pv_terminal))
            d.add_row("Equity Value",            fmt_dollars(result.intrinsic_value_equity))
            d.add_row("Intrinsic Value/Share",   f"[bold]{fmt_dollars(result.intrinsic_value_per_share)}[/bold]")
            d.add_row("Current Price",           fmt_dollars(current_price))
            d.add_row("Upside / (Downside)",     f"[{upside_color}]{upside_str}[/{upside_color}]")
            d.add_row("MoS 20% Price",           fmt_dollars(result.margin_of_safety_20))
            d.add_row("MoS 30% Price",           fmt_dollars(result.margin_of_safety_30))
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
            print(f"  MoS 20% Price:           {fmt_dollars(result.margin_of_safety_20)}")
            print(f"  MoS 30% Price:           {fmt_dollars(result.margin_of_safety_30)}")
            print()

        # Sensitivity table
        sens = sensitivity_table(dcf_inputs)
        if RICH:
            s = Table(title="Sensitivity: IV/Share  (rows=TGR, cols=WACC)", box=box.SIMPLE)
            s.add_column("TGR \\ WACC", style="bold")
            for col in sens.columns:
                s.add_column(col, justify="right")
            for idx, row in sens.iterrows():
                vals = [str(v) if str(v) != "nan" else "N/A" for v in row.values]
                # Highlight closest to current price
                colored = []
                for v_str, v_raw in zip(vals, row.values):
                    try:
                        v = float(v_raw)
                        if current_price and abs(v - current_price) / current_price < 0.10:
                            colored.append(f"[yellow]{v_str}[/yellow]")
                        elif current_price and v > current_price * 1.1:
                            colored.append(f"[green]{v_str}[/green]")
                        else:
                            colored.append(f"[red]{v_str}[/red]")
                    except Exception:
                        colored.append(v_str)
                s.add_row(str(idx), *colored)
            console.print(s)
        else:
            print("\n  Sensitivity Table (Intrinsic Value per Share):")
            print(f"  TGR \\ WACC  |  " + "  |  ".join(sens.columns))
            print("  " + "-" * (len(sens.columns) * 12 + 14))
            for idx, row in sens.iterrows():
                vals = "  |  ".join(
                    str(v) if str(v) != "nan" else "N/A" for v in row.values
                )
                print(f"  {idx:<11} |  {vals}")

    except Exception as e:
        print(f"  [!] DCF failed: {e}")

    if RICH:
        console.print("\n[dim]Note: DCF uses Yahoo Finance data. ARR/NRR approximations may "
                      "differ from company-reported figures. Always verify inputs.[/dim]")
    else:
        print("\n  Note: DCF uses Yahoo Finance data. ARR approximated from quarterly run-rate.")
        print("        Always verify DCF inputs against actual filings.\n")


def main():
    tickers = [t.upper() for t in sys.argv[1:] if not t.startswith("-")]

    if not tickers:
        print("Usage: python main.py TICKER1 TICKER2 ...")
        print("  e.g. python main.py AAPL MSFT CRM SNOW NVDA")
        sys.exit(1)

    if RICH:
        console.print(Panel.fit(
            "[bold cyan]DCF Valuation Tool[/bold cyan]\n"
            "Perpetuity-based DCF  |  SaaS & General Metrics\n"
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
            time.sleep(1)  # be kind to Yahoo Finance rate limits


if __name__ == "__main__":
    main()
