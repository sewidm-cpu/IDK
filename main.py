#!/usr/bin/env python3
"""
DCF Valuation Tool
==================
Usage:
    python main.py AAPL MSFT CRM SNOW LMT XOM JPM PLD
    python main.py CRM SNOW --export csv
    python main.py CRM SNOW --export excel

Features:
  - SEC EDGAR XBRL as primary data source (disk-cached 24hr)
  - Yahoo Finance for price, market cap, beta, shares
  - Sector-aware: SaaS / Cyclical / Defense / Financial / REIT / Biotech
  - DCF: perpetuity + exit multiple (side by side)
  - Owner earnings (FCF minus SBC)
  - Bull / base / bear scenarios
  - Reverse DCF (implied growth rate at current price)
  - WACC × TGR sensitivity table
  - Side-by-side comparison when multiple tickers
  - CSV / Excel export
"""

import sys
import csv
import time
import math
from datetime import datetime
from pathlib import Path
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
from dcf import (DCFInputs, DCFResult, run_dcf, run_scenarios,
                 reverse_dcf, sensitivity_table, EXIT_MULTIPLES)
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

SENSITIVITY_RANGES = {
    "CYCLICAL": {"wacc_delta": [-0.04,-0.02,0,.02,.04], "tgr": [.005,.01,.02,.025,.03]},
    "DEFENSE":  {"wacc_delta": [-0.02,-0.01,0,.01,.02], "tgr": [.005,.01,.015,.02,.025]},
    "SAAS":     {"wacc_delta": [-0.02,-0.01,0,.01,.02], "tgr": [.02,.025,.03,.035,.04]},
    "BIOTECH":  {"wacc_delta": [-0.03,-0.015,0,.015,.03],"tgr":[.01,.02,.025,.03,.035]},
}
DEFAULT_RANGES = {"wacc_delta": [-0.02,-0.01,0,.01,.02], "tgr": [.01,.02,.025,.03,.04]}
DCF_SKIP_TYPES = {"FINANCIAL", "REIT"}
TYPE_COLORS    = {
    "SAAS":"green","CYCLICAL":"yellow","DEFENSE":"blue",
    "FINANCIAL":"magenta","REIT":"cyan","BIOTECH":"red","GENERAL":"white",
}


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_d(v, p=2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
    v = float(v)
    if abs(v) >= 1e12: return f"${v/1e12:.{p}f}T"
    if abs(v) >= 1e9:  return f"${v/1e9:.{p}f}B"
    if abs(v) >= 1e6:  return f"${v/1e6:.{p}f}M"
    return f"${v:,.{p}f}"

def _fmt_p(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
    return f"{float(v):.1f}%"

def _fmt_m(v) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)): return "N/A"
    return f"{float(v):.1f}x"

def _fmt_any(v) -> str:
    if v is None: return "N/A"
    if isinstance(v, str): return v
    if isinstance(v, float) and math.isnan(v): return "N/A"
    if isinstance(v, (int, float)):
        f = float(v)
        if abs(f) >= 1e6: return _fmt_d(f)
        return f"{f:.2f}"
    return str(v)


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _table(title: str):
    if RICH:
        t = Table(title=title, box=box.SIMPLE_HEAVY, show_header=True)
        t.add_column("", style="bold")
        t.add_column("", justify="right")
        return t
    return None

def _row(t, label: str, value: str):
    if RICH: t.add_row(label, value)
    else: print(f"  {label:<38} {value}")

def _print_table(t, plain_header=""):
    if RICH: console.print(t)
    elif plain_header: print(f"\n--- {plain_header} ---")

def _warn(msg):
    if RICH: console.print(f"[bold yellow]  ⚠  {msg}[/bold yellow]")
    else: print(f"  [!] {msg}")

def _note(msg):
    if RICH: console.print(f"[dim]  {msg}[/dim]")
    else: print(f"  {msg}")

def _head(msg, color="cyan"):
    if RICH: console.print(f"[bold {color}]{msg}[/bold {color}]")
    else: print(msg)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_ticker(ticker: str, risk_free_rate: float) -> Optional[dict]:
    """Analyze one ticker. Returns a summary dict for comparison table / export."""
    ticker = ticker.upper()

    if RICH:
        console.print(f"\n[bold cyan]{'='*64}[/bold cyan]")
        console.print(f"[bold white]  {ticker}[/bold white]")
        console.print(f"[bold cyan]{'='*64}[/bold cyan]")
    else:
        print(f"\n{'='*64}\n  {ticker}\n{'='*64}")

    try:
        info = get_ticker_info(ticker)
        if not info or info.get("regularMarketPrice") is None:
            _warn(f"No data for {ticker}"); return None
    except Exception as e:
        _warn(f"Error fetching {ticker}: {e}"); return None

    try:
        fin = get_financials(ticker)
    except Exception as e:
        _warn(f"Error fetching financials for {ticker}: {e}"); return None

    name    = info.get("longName") or info.get("shortName") or ticker
    sector  = info.get("sector",   "Unknown")
    industry= info.get("industry", "Unknown")
    price   = info.get("currentPrice") or info.get("regularMarketPrice")
    ctype   = classify_company(info)
    color   = TYPE_COLORS.get(ctype, "white")
    source  = fin.get("source", "yahoo")

    if RICH:
        console.print(f"[bold]{name}[/bold]  |  {sector} / {industry}")
        console.print(
            f"Price: [yellow]{_fmt_d(price)}[/yellow]  |  "
            f"Type: [{color}]{ctype}[/{color}]  |  "
            f"Data: [dim]{source.upper()}[/dim]"
        )
    else:
        print(f"{name}  |  {sector} / {industry}")
        print(f"Price: {_fmt_d(price)}  |  Type: {ctype}  |  Data: {source.upper()}")

    # --- General metrics ---
    general = compute_general_metrics(info, fin)
    t = _table("Key Metrics")
    if not RICH: print("\n--- Key Metrics ---")

    _row(t, "Market Cap",             _fmt_d(general["Market Cap"]))
    _row(t, "Enterprise Value",       _fmt_d(general["Enterprise Value"]))
    _row(t, "TTM Revenue",            _fmt_d(general["TTM Revenue"]))
    _row(t, "Revenue CAGR 3yr",       _fmt_p(general["Revenue CAGR 3yr %"]))
    _row(t, "Gross Margin",           _fmt_p(general["Gross Margin %"]))
    _row(t, "EBITDA Margin",          _fmt_p(general["EBITDA Margin %"]))
    _row(t, "Net Margin",             _fmt_p(general["Net Margin %"]))
    _row(t, "TTM FCF",                _fmt_d(general["TTM FCF"]))
    _row(t, "TTM Owner Earnings",     _fmt_d(general["TTM Owner Earnings"]))
    _row(t, "SBC (TTM)",              _fmt_d(general["SBC (TTM)"]))
    _row(t, "SBC % Revenue",          _fmt_p(general["SBC % Revenue"]))
    _row(t, "FCF Yield",              _fmt_p(general["FCF Yield %"]))
    _row(t, "Owner Earnings Yield",   _fmt_p(general["Owner Earnings Yield %"]))

    wc = general["Avg WC Change % Rev"]
    if wc is not None:
        wc_str = _fmt_p(wc)
        if RICH and abs(wc) > 3:
            wc_str = f"[yellow]{wc_str} ⚠[/yellow]"
        _row(t, "Avg WC Change % Rev", wc_str)

    _row(t, "EV/Revenue",             _fmt_m(general["EV/Revenue"]))
    _row(t, "EV/EBITDA",              _fmt_m(general["EV/EBITDA"]))
    _row(t, "P/E (trailing)",         _fmt_m(general["P/E (trailing)"]))
    _row(t, "P/FCF",                  _fmt_m(general["P/FCF"]))
    _row(t, "P/Owner Earnings",       _fmt_m(general["P/Owner Earnings"]))
    _row(t, "Net Debt",               _fmt_d(general["Net Debt"]))

    # --- Type-specific metrics ---
    if ctype == "SAAS":
        m = compute_saas_metrics(info, fin)
        _row(t, "── SaaS ──",             "")
        _row(t, "ARR (approx.)",           _fmt_d(m["ARR (approx.)"]) if "ARR (approx.)" in m else _fmt_d(m.get("ARR (approx)")))
        _row(t, "Revenue Growth YoY",      _fmt_p(m["Revenue Growth YoY %"]))
        _row(t, "FCF Margin",              _fmt_p(m["FCF Margin %"]))
        _row(t, "Owner Earnings Margin",   _fmt_p(m["Owner Earnings Margin %"]))
        r40 = m["Rule of 40 (FCF)"]
        r40_oe = m["Rule of 40 (OE)"]
        def _r40_str(v):
            if v is None: return "N/A"
            s = _fmt_p(v)
            if RICH: return f"[{'green' if v>=40 else 'yellow' if v>=20 else 'red'}]{s}[/{'green' if v>=40 else 'yellow' if v>=20 else 'red'}]"
            return s
        _row(t, "Rule of 40 (FCF)",        _r40_str(r40))
        _row(t, "Rule of 40 (OE)",         _r40_str(r40_oe))
        _row(t, "RPO (contracted rev)",    _fmt_d(m["RPO (contracted rev)"]))
        _row(t, "Deferred Revenue",        _fmt_d(m["Deferred Revenue"]))
        _row(t, "NRR",                     str(m["NRR"]))

    elif ctype == "FINANCIAL":
        m = compute_financial_metrics(info, fin)
        _row(t, "── Financial ──",         "")
        _row(t, "P/B Ratio",               _fmt_m(m["P/B Ratio"]))
        _row(t, "ROE",                     _fmt_p(m["ROE %"]))
        _row(t, "P/E (forward)",           _fmt_m(m["P/E (forward)"]))
        _row(t, "Dividend Yield",          _fmt_p(m["Dividend Yield %"]))
        _row(t, "Tier 1 Capital",          m["Tier 1 Capital Ratio"])

    elif ctype == "REIT":
        m = compute_reit_metrics(info, fin)
        _row(t, "── REIT ──",              "")
        _row(t, "FFO (NI+D&A proxy)",      _fmt_d(m["FFO proxy (NI+D&A)"]))
        _row(t, "P/FFO (proxy)",           _fmt_m(m["P/FFO (proxy)"]))
        _row(t, "Dividend Yield",          _fmt_p(m["Dividend Yield %"]))

    elif ctype == "BIOTECH":
        m = compute_biotech_metrics(info, fin)
        _row(t, "── Biotech ──",           "")
        _row(t, "Cash & Equivalents",      _fmt_d(m["Cash & Equivalents"]))
        _row(t, "Quarterly Burn (avg)",    _fmt_d(m["Avg Quarterly Cash Burn"]))
        runway = m["Cash Runway (quarters)"]
        rs = f"{runway:.1f}q" if runway else "N/A"
        if RICH and runway:
            rs = f"[{'green' if runway>8 else 'yellow' if runway>4 else 'red'}]{rs}[/{'green' if runway>8 else 'yellow' if runway>4 else 'red'}]"
        _row(t, "Cash Runway",             rs)

    elif ctype == "DEFENSE":
        flags = compute_defense_flags(info, fin)
        _row(t, "── Defense ──",           "")
        _row(t, "Backlog Note",            flags["Backlog Note"])

    elif ctype == "CYCLICAL":
        _row(t, "── Cyclical ──",          "")
        _row(t, "Note",                    "DCF uses 7yr cycle-normalized FCF margin")

    _print_table(t, "Key Metrics")

    # --- DCF skipped for FINANCIAL / REIT ---
    if ctype in DCF_SKIP_TYPES:
        msgs = {
            "FINANCIAL": "DCF skipped — use P/B, ROE, and dividend yield above.",
            "REIT":      "DCF skipped — use FFO/AFFO and dividend yield above.",
        }
        _warn(msgs[ctype])
        _footer()
        return _summary(ticker, ctype, price, general, None, None, None, None, source)

    if ctype == "BIOTECH":
        _warn("DCF reliability LOW — pre-revenue / negative FCF.")

    # --- Build DCF inputs ---
    try:
        raw = extract_dcf_inputs_from_data(ticker, info, fin, risk_free_rate, ctype)
        meta = {k: raw.pop(k) for k in list(raw) if k.startswith("_")}
        if meta.get("_margin_note"):
            _note(meta["_margin_note"])

        di = DCFInputs(**raw)

        # --- Scenarios ---
        scenarios = run_scenarios(di)
        base_result = scenarios["base"]

        # --- Reverse DCF ---
        implied_g = reverse_dcf(di, price) if price else None

        # --- Print DCF results ---
        _head("\nDCF Results", "bold")
        d = _table("DCF")
        if not RICH: print("\n--- DCF to Perpetuity ---")

        _row(d, "WACC",                   _fmt_p(base_result.wacc * 100))
        _row(d, "Terminal Growth Rate",   _fmt_p(base_result.terminal_growth_rate * 100))
        _row(d, "PV of FCFs",             _fmt_d(base_result.pv_fcfs))
        _row(d, "Terminal Value (perp.)", _fmt_d(base_result.terminal_value))
        _row(d, "PV of Terminal Value",   _fmt_d(base_result.pv_terminal))
        _row(d, "── Perpetuity Method ──","")
        _row(d, "IV/Share",               _fmt_d(base_result.intrinsic_value_per_share))

        if base_result.iv_per_share_exit is not None:
            mult = di.exit_ebitda_multiple
            _row(d, f"── Exit {mult:.0f}x EBITDA ──", "")
            _row(d, "IV/Share (exit mult)", _fmt_d(base_result.iv_per_share_exit))
            avg_iv = (base_result.intrinsic_value_per_share + base_result.iv_per_share_exit) / 2
            _row(d, "IV/Share (avg both)",  _fmt_d(avg_iv))

        if base_result.iv_per_share_owner_earnings is not None:
            _row(d, "── Owner Earnings ──", "")
            _row(d, "IV/Share (OE-based)", _fmt_d(base_result.iv_per_share_owner_earnings))

        _row(d, "── vs Market ──",        "")
        _row(d, "Current Price",          _fmt_d(price))

        def _upside(iv):
            if iv and price:
                u = (iv / price - 1) * 100
                s = f"{u:+.1f}%"
                return f"[{'green' if u>0 else 'red'}]{s}[/{'green' if u>0 else 'red'}]" if RICH else s
            return "N/A"

        _row(d, "Upside (perp.)",         _upside(base_result.intrinsic_value_per_share))
        if base_result.iv_per_share_exit:
            _row(d, "Upside (exit mult)", _upside(base_result.iv_per_share_exit))
        _row(d, "MoS 20% Entry",          _fmt_d(base_result.margin_of_safety_20))
        _row(d, "MoS 30% Entry",          _fmt_d(base_result.margin_of_safety_30))

        if implied_g is not None:
            ig_str = f"{implied_g:.1%}"
            if RICH:
                color_ig = "green" if implied_g < 0.10 else "yellow" if implied_g < 0.20 else "red"
                ig_str = f"[{color_ig}]{ig_str}[/{color_ig}]"
            _row(d, "Implied Growth Rate", ig_str)

        _print_table(d, "DCF Results")

        # --- Bull / Base / Bear ---
        _head("\nScenario Analysis", "bold")
        s = _table("Scenarios") if RICH else None
        if RICH:
            s = Table(title="Bull / Base / Bear", box=box.SIMPLE_HEAVY)
            s.add_column("Scenario", style="bold")
            s.add_column("IV/Share",   justify="right")
            s.add_column("Upside",     justify="right")
            s.add_column("WACC",       justify="right")
            s.add_column("TGR",        justify="right")
            s.add_column("FCF Margin", justify="right")
        else:
            print(f"\n  {'Scenario':<10} {'IV/Share':>10} {'Upside':>9} {'WACC':>7} {'TGR':>6} {'FCF Marg':>9}")
            print("  " + "-" * 55)

        bull_in, _, bear_in = (
            DCFInputs(**{**di.__dict__,
                "revenue_growth_rates": [min(r*m, 0.80) for r in di.revenue_growth_rates],
                "wacc": max(di.wacc+wd, 0.04),
                "terminal_growth_rate": di.terminal_growth_rate+td,
                "fcf_margin_stable": max(di.fcf_margin_stable+md, 0.01),
            })
            for m, wd, td, md in [(1.30,-0.01,+0.005,+0.05),(1.0,0,0,0),(0.70,+0.01,-0.005,-0.05)]
        )

        for label, inp, col in [
            ("Bull", bull_in,  "green"),
            ("Base", di,       "white"),
            ("Bear", bear_in,  "red"),
        ]:
            try:
                r = run_dcf(inp)
                iv = r.intrinsic_value_per_share
                upside = ((iv / price) - 1) * 100 if price else None
                us_str = f"{upside:+.1f}%" if upside is not None else "N/A"
                if RICH:
                    s.add_row(
                        f"[{col}]{label}[/{col}]",
                        f"[{col}]{_fmt_d(iv)}[/{col}]",
                        f"[{col}]{us_str}[/{col}]",
                        _fmt_p(inp.wacc*100),
                        _fmt_p(inp.terminal_growth_rate*100),
                        _fmt_p(inp.fcf_margin_stable*100),
                    )
                else:
                    print(f"  {label:<10} {_fmt_d(iv):>10} {us_str:>9} "
                          f"{inp.wacc:.1%}  {inp.terminal_growth_rate:.1%}  {inp.fcf_margin_stable:.1%}")
            except Exception:
                if RICH: s.add_row(label, "N/A","N/A","","","")
                else: print(f"  {label:<10} N/A")

        if RICH: console.print(s)

        # --- Sensitivity table ---
        ranges    = SENSITIVITY_RANGES.get(ctype, DEFAULT_RANGES)
        wacc_rng  = [di.wacc + d for d in ranges["wacc_delta"]]
        tgr_rng   = ranges["tgr"]
        sens      = sensitivity_table(di, wacc_range=wacc_rng, tgr_range=tgr_rng)

        if RICH:
            st = Table(title="Sensitivity: IV/Share (rows=TGR, cols=WACC)", box=box.SIMPLE)
            st.add_column("TGR \\ WACC", style="bold")
            for col in sens.columns:
                st.add_column(col, justify="right")
            for idx, row in sens.iterrows():
                colored = []
                for v_raw in row.values:
                    try:
                        v = float(v_raw)
                        vs = str(round(v, 2))
                        if price and abs(v - price) / price < 0.10:
                            colored.append(f"[yellow]{vs}[/yellow]")
                        elif price and v > price * 1.1:
                            colored.append(f"[green]{vs}[/green]")
                        else:
                            colored.append(f"[red]{vs}[/red]")
                    except Exception:
                        colored.append("N/A")
                st.add_row(str(idx), *colored)
            console.print(st)
        else:
            print("\n  Sensitivity (IV/Share):")
            hdr_label = "TGR\\WACC"
            print(f"  {hdr_label:<11} | " + " | ".join(sens.columns))
            print("  " + "-" * (len(sens.columns) * 10 + 14))
            for idx, row in sens.iterrows():
                vals = " | ".join(f"{round(v,2):>8}" if str(v)!="nan" else "     N/A" for v in row.values)
                print(f"  {str(idx):<11} | {vals}")

    except Exception as e:
        _warn(f"DCF failed: {e}")
        base_result, implied_g = None, None

    _footer(source)
    return _summary(ticker, ctype, price, general,
                    base_result if "base_result" in dir() else None,
                    implied_g   if "implied_g"   in dir() else None,
                    fin, info, source)


def _footer(source="unknown"):
    msg = f"Data: {'SEC EDGAR + Yahoo Finance' if source=='edgar' else 'Yahoo Finance (EDGAR unavailable)'}. Verify against filings."
    if RICH: console.print(f"\n[dim]  {msg}[/dim]")
    else: print(f"\n  {msg}\n")


def _summary(ticker, ctype, price, general, dcf_result, implied_g, fin, info, source) -> dict:
    """Collect key values for comparison table and export."""
    iv      = dcf_result.intrinsic_value_per_share if dcf_result else None
    iv_exit = dcf_result.iv_per_share_exit         if dcf_result else None
    iv_oe   = dcf_result.iv_per_share_owner_earnings if dcf_result else None
    upside  = ((iv / price - 1) * 100) if (iv and price) else None
    return {
        "Ticker":              ticker,
        "Type":                ctype,
        "Source":              source,
        "Price":               price,
        "IV/Share (perp)":     iv,
        "IV/Share (exit)":     iv_exit,
        "IV/Share (OE)":       iv_oe,
        "Upside % (perp)":     upside,
        "Implied Growth":      implied_g,
        "Market Cap":          general.get("Market Cap"),
        "EV/Revenue":          general.get("EV/Revenue"),
        "EV/EBITDA":           general.get("EV/EBITDA"),
        "Rev CAGR 3yr %":      general.get("Revenue CAGR 3yr %"),
        "EBITDA Margin %":     general.get("EBITDA Margin %"),
        "FCF Yield %":         general.get("FCF Yield %"),
        "OE Yield %":          general.get("Owner Earnings Yield %"),
        "SBC % Rev":           general.get("SBC % Revenue"),
        "Net Debt":            general.get("Net Debt"),
    }


def _print_comparison(summaries: list):
    if not summaries:
        return
    if RICH:
        console.print(f"\n[bold cyan]{'='*64}[/bold cyan]")
        console.print("[bold white]  COMPARISON TABLE[/bold white]")
        console.print(f"[bold cyan]{'='*64}[/bold cyan]")
        ct = Table(box=box.SIMPLE_HEAVY)
        cols = ["Ticker","Type","Price","IV (perp)","IV (exit)","Upside","Impl.G",
                "Mkt Cap","EV/Rev","FCF Yld","SBC%Rev","Rev CAGR"]
        for c in cols: ct.add_column(c, justify="right")
        for s in summaries:
            color = TYPE_COLORS.get(s["Type"], "white")
            iv_p  = s["IV/Share (perp)"]
            up    = s["Upside % (perp)"]
            up_c  = "green" if (up or 0) > 0 else "red"
            ig    = s["Implied Growth"]
            ct.add_row(
                f"[{color}]{s['Ticker']}[/{color}]",
                f"[{color}]{s['Type']}[/{color}]",
                _fmt_d(s["Price"]),
                _fmt_d(iv_p),
                _fmt_d(s["IV/Share (exit)"]),
                f"[{up_c}]{_fmt_p(up)}[/{up_c}]" if up is not None else "N/A",
                _fmt_p((ig or 0)*100) if ig is not None else "N/A",
                _fmt_d(s["Market Cap"]),
                _fmt_m(s["EV/Revenue"]),
                _fmt_p(s["FCF Yield %"]),
                _fmt_p(s["SBC % Rev"]),
                _fmt_p(s["Rev CAGR 3yr %"]),
            )
        console.print(ct)
    else:
        print(f"\n{'='*64}\n  COMPARISON TABLE\n{'='*64}")
        hdr = f"{'Ticker':<8} {'Type':<10} {'Price':>8} {'IV(perp)':>10} {'Upside':>8} {'ImpG':>7} {'EV/Rev':>7} {'FCFYld':>7}"
        print(hdr)
        print("-" * len(hdr))
        for s in summaries:
            up = s["Upside % (perp)"]
            ig = s["Implied Growth"]
            print(
                f"{s['Ticker']:<8} {s['Type']:<10} "
                f"{_fmt_d(s['Price']):>8} {_fmt_d(s['IV/Share (perp)']):>10} "
                f"{_fmt_p(up):>8} "
                f"{_fmt_p((ig or 0)*100) if ig is not None else 'N/A':>7} "
                f"{_fmt_m(s['EV/Revenue']):>7} {_fmt_p(s['FCF Yield %']):>7}"
            )


def _export_csv(summaries: list, path: Path):
    if not summaries:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\n  Exported CSV → {path}")


def _export_excel(summaries: list, path: Path):
    try:
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "DCF Results"
        if summaries:
            ws.append(list(summaries[0].keys()))
            for row in summaries:
                ws.append(list(row.values()))
        wb.save(path)
        print(f"\n  Exported Excel → {path}")
    except ImportError:
        _warn("openpyxl not installed — run: pip install openpyxl")
        _export_csv(summaries, path.with_suffix(".csv"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args    = sys.argv[1:]
    tickers = [a.upper() for a in args if not a.startswith("--")]
    export  = None
    for i, a in enumerate(args):
        if a == "--export" and i + 1 < len(args):
            export = args[i + 1].lower()

    if not tickers:
        print("Usage: python main.py TICKER1 TICKER2 ... [--export csv|excel]")
        print("  e.g. python main.py AAPL MSFT CRM SNOW LMT XOM JPM")
        sys.exit(1)

    if RICH:
        console.print(Panel.fit(
            "[bold cyan]DCF Valuation Tool[/bold cyan]\n"
            "Perpetuity + Exit Multiple  |  Owner Earnings  |  Bull/Base/Bear  |  Reverse DCF\n"
            "Data: SEC EDGAR XBRL (cached) + Yahoo Finance",
            border_style="cyan"
        ))

    rfr = get_risk_free_rate()
    if RICH: console.print(f"Risk-free rate (10yr UST): [yellow]{rfr:.2%}[/yellow]")
    else: print(f"Risk-free rate (10yr UST): {rfr:.2%}")

    summaries = []
    for i, ticker in enumerate(tickers):
        result = analyze_ticker(ticker, rfr)
        if result:
            summaries.append(result)
        if i < len(tickers) - 1:
            time.sleep(0.5)

    if len(summaries) > 1:
        _print_comparison(summaries)

    if export and summaries:
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"dcf_results_{ts}"
        if export == "excel":
            _export_excel(summaries, Path(stem + ".xlsx"))
        else:
            _export_csv(summaries, Path(stem + ".csv"))


if __name__ == "__main__":
    main()
