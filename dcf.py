"""
DCF valuation engine.

Methods:
  1. Perpetuity (Gordon Growth Model) — default
  2. Exit EV/EBITDA multiple          — alternative terminal value
  Both are computed when exit_ebitda_multiple > 0.

Additional tools:
  reverse_dcf()    — implied revenue growth rate at current price
  build_scenarios()— auto-generate bull / base / bear DCFInputs
  run_scenarios()  — run all three and return results
  sensitivity_table() — WACC × TGR grid
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

EQUITY_RISK_PREMIUM = 0.055
TAX_RATE_DEFAULT    = 0.21

# Default EV/EBITDA exit multiples by company type
EXIT_MULTIPLES = {
    "SAAS":      25.0,
    "GENERAL":   12.0,
    "CYCLICAL":   8.0,
    "DEFENSE":   12.0,
    "BIOTECH":    0.0,   # not meaningful
    "FINANCIAL":  0.0,
    "REIT":       0.0,
}


@dataclass
class DCFInputs:
    ticker: str
    current_fcf: float
    revenue_growth_rates: list
    fcf_margin_stable: float
    revenue_ttm: float
    wacc: float
    terminal_growth_rate: float = 0.025
    projection_years: int = 10
    shares_outstanding: float = 0.0
    net_debt: float = 0.0
    # Owner-earnings adjustment: SBC as fraction of revenue
    sbc_pct_revenue: float = 0.0
    # Exit multiple terminal value (0 = skip)
    ebitda_margin_stable: float = 0.0
    exit_ebitda_multiple: float = 0.0


@dataclass
class DCFResult:
    ticker: str
    # Perpetuity method
    intrinsic_value_equity: float
    intrinsic_value_per_share: float
    pv_fcfs: float
    pv_terminal: float
    terminal_value: float
    wacc: float
    terminal_growth_rate: float
    projected_fcfs: list
    projected_revenues: list
    discount_factors: list
    margin_of_safety_20: float
    margin_of_safety_30: float
    # Exit multiple method (None when not computed)
    iv_per_share_exit: Optional[float] = None
    terminal_value_exit: Optional[float] = None
    # Owner earnings per share (FCF - SBC)
    iv_per_share_owner_earnings: Optional[float] = None


def build_wacc(
    beta: float,
    risk_free_rate: float,
    equity_value: float,
    debt: float,
    cost_of_debt: float,
    tax_rate: float = TAX_RATE_DEFAULT,
    equity_risk_premium: float = EQUITY_RISK_PREMIUM,
) -> float:
    ke    = risk_free_rate + beta * equity_risk_premium
    total = equity_value + debt
    if total <= 0:
        return ke
    return (equity_value / total) * ke + (debt / total) * cost_of_debt * (1 - tax_rate)


def _project(
    revenue_ttm: float,
    fcf_margin_stable: float,
    revenue_growth_rates: list,
    current_fcf: float,
    projection_years: int,
    sbc_pct_revenue: float = 0.0,
) -> tuple[list, list, list]:
    """
    Returns (fcf_series, owner_earnings_series, revenue_series).
    FCF blends from current margin to stable margin.
    Owner earnings = FCF - SBC (SBC expressed as % of projected revenue).
    """
    rates = list(revenue_growth_rates)
    while len(rates) < projection_years:
        rates.append(rates[-1] if rates else 0.05)

    fcfs, oes, revs = [], [], []
    rev            = revenue_ttm
    current_margin = (current_fcf / revenue_ttm) if revenue_ttm else 0.1

    for i in range(projection_years):
        rev = rev * (1 + rates[i])
        t   = (i + 1) / projection_years
        blended = current_margin * (1 - t) + fcf_margin_stable * t
        fcf = rev * blended
        oe  = fcf - rev * sbc_pct_revenue
        fcfs.append(fcf)
        oes.append(oe)
        revs.append(rev)

    return fcfs, oes, revs


def run_dcf(inputs: DCFInputs) -> DCFResult:
    wacc = inputs.wacc
    g    = inputs.terminal_growth_rate

    if wacc <= g:
        raise ValueError(f"WACC ({wacc:.2%}) must exceed terminal growth rate ({g:.2%})")

    fcfs, oes, revs = _project(
        revenue_ttm=inputs.revenue_ttm,
        fcf_margin_stable=inputs.fcf_margin_stable,
        revenue_growth_rates=inputs.revenue_growth_rates,
        current_fcf=inputs.current_fcf,
        projection_years=inputs.projection_years,
        sbc_pct_revenue=inputs.sbc_pct_revenue,
    )

    dfs      = [(1 / (1 + wacc) ** (i + 1)) for i in range(inputs.projection_years)]
    pv_fcfs  = sum(f * d for f, d in zip(fcfs, dfs))

    # --- Terminal value: perpetuity ---
    tv_perp    = fcfs[-1] * (1 + g) / (wacc - g)
    pv_tv_perp = tv_perp * dfs[-1]
    ev_perp    = pv_fcfs + pv_tv_perp
    eq_perp    = ev_perp - inputs.net_debt
    ps_perp    = (eq_perp / inputs.shares_outstanding) if inputs.shares_outstanding > 0 else 0.0

    # --- Terminal value: exit multiple (optional) ---
    ps_exit = tv_exit = None
    if inputs.exit_ebitda_multiple > 0 and inputs.ebitda_margin_stable > 0:
        final_ebitda = revs[-1] * inputs.ebitda_margin_stable
        tv_exit      = final_ebitda * inputs.exit_ebitda_multiple
        pv_tv_exit   = tv_exit * dfs[-1]
        ev_exit      = pv_fcfs + pv_tv_exit
        eq_exit      = ev_exit - inputs.net_debt
        ps_exit      = (eq_exit / inputs.shares_outstanding) if inputs.shares_outstanding > 0 else 0.0

    # --- Owner earnings per share ---
    pv_oes   = sum(o * d for o, d in zip(oes, dfs))
    tv_oe    = oes[-1] * (1 + g) / (wacc - g)
    ev_oe    = pv_oes + tv_oe * dfs[-1]
    eq_oe    = ev_oe - inputs.net_debt
    ps_oe    = (eq_oe / inputs.shares_outstanding) if inputs.shares_outstanding > 0 else None

    return DCFResult(
        ticker=inputs.ticker,
        intrinsic_value_equity=eq_perp,
        intrinsic_value_per_share=ps_perp,
        pv_fcfs=pv_fcfs,
        pv_terminal=pv_tv_perp,
        terminal_value=tv_perp,
        wacc=wacc,
        terminal_growth_rate=g,
        projected_fcfs=fcfs,
        projected_revenues=revs,
        discount_factors=dfs,
        margin_of_safety_20=ps_perp * 0.80,
        margin_of_safety_30=ps_perp * 0.70,
        iv_per_share_exit=ps_exit,
        terminal_value_exit=tv_exit,
        iv_per_share_owner_earnings=ps_oe,
    )


# ---------------------------------------------------------------------------
# Reverse DCF — implied growth rate at current price
# ---------------------------------------------------------------------------

def reverse_dcf(inputs: DCFInputs, current_price: float) -> Optional[float]:
    """
    Binary search for the flat annual revenue growth rate that makes
    intrinsic value per share equal to current_price.
    Returns None if no solution in [-20%, +80%] range.
    """
    if inputs.shares_outstanding <= 0 or current_price <= 0:
        return None

    def iv_at_rate(g: float) -> float:
        inp = DCFInputs(**{
            **inputs.__dict__,
            "revenue_growth_rates": [g] * inputs.projection_years,
        })
        try:
            return run_dcf(inp).intrinsic_value_per_share
        except Exception:
            return 0.0

    lo, hi = -0.20, 0.80
    if iv_at_rate(lo) > current_price or iv_at_rate(hi) < current_price:
        return None   # price is outside solvable range

    for _ in range(60):
        mid = (lo + hi) / 2
        if iv_at_rate(mid) < current_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

def build_scenarios(base: DCFInputs) -> tuple:
    """
    Auto-generate (bull, base, bear) DCFInputs from base.

    Bull: growth +30%, FCF margin +5pp, WACC -1pp, TGR +0.5pp
    Bear: growth -30%, FCF margin -5pp, WACC +1pp, TGR -0.5pp
    """
    def _adjust(growth_mult, wacc_delta, tgr_delta, margin_delta):
        rates = [min(r * growth_mult, 0.80) for r in base.revenue_growth_rates]
        return DCFInputs(**{
            **base.__dict__,
            "revenue_growth_rates": rates,
            "wacc":                 max(base.wacc + wacc_delta, 0.04),
            "terminal_growth_rate": base.terminal_growth_rate + tgr_delta,
            "fcf_margin_stable":    max(base.fcf_margin_stable + margin_delta, 0.01),
        })

    bull = _adjust(1.30, -0.01, +0.005, +0.05)
    bear = _adjust(0.70, +0.01, -0.005, -0.05)
    return bull, base, bear


def run_scenarios(base: DCFInputs) -> dict:
    """Return dict with keys 'bull', 'base', 'bear', each a DCFResult."""
    bull, base_inp, bear = build_scenarios(base)
    results = {}
    for label, inp in [("bull", bull), ("base", base_inp), ("bear", bear)]:
        try:
            results[label] = run_dcf(inp)
        except Exception as e:
            results[label] = None
    return results


# ---------------------------------------------------------------------------
# Sensitivity table
# ---------------------------------------------------------------------------

def sensitivity_table(inputs: DCFInputs, wacc_range=None, tgr_range=None) -> pd.DataFrame:
    if wacc_range is None:
        wacc_range = [inputs.wacc + d for d in (-0.02, -0.01, 0, 0.01, 0.02)]
    if tgr_range is None:
        tgr_range = [0.01, 0.02, inputs.terminal_growth_rate, 0.03, 0.04]

    rows = {}
    for tgr in tgr_range:
        row = {}
        for w in wacc_range:
            try:
                inp = DCFInputs(**{**inputs.__dict__, "wacc": w, "terminal_growth_rate": tgr})
                row[f"{w:.1%}"] = round(run_dcf(inp).intrinsic_value_per_share, 2)
            except Exception:
                row[f"{w:.1%}"] = float("nan")
        rows[f"{tgr:.1%}"] = row

    df = pd.DataFrame(rows).T
    df.index.name = "TGR \\ WACC"
    return df
