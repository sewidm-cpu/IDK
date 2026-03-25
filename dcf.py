"""
DCF valuation engine — projects free cash flows and discounts to perpetuity
using the Gordon Growth Model for terminal value.

  Intrinsic Value = PV(FCF projections) + PV(Terminal Value)
  Terminal Value  = FCF_n * (1 + g) / (WACC - g)
  WACC            = E/(D+E)*Ke + D/(D+E)*Kd*(1-t)
  Ke              = Rf + beta*(Rm - Rf)   [CAPM]
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


EQUITY_RISK_PREMIUM = 0.055   # Damodaran long-run ERP
TAX_RATE_DEFAULT    = 0.21    # US corporate tax


@dataclass
class DCFInputs:
    ticker: str
    current_fcf: float          # trailing twelve-month FCF (USD)
    revenue_growth_rates: list  # list of annual growth rates for projection period
    fcf_margin_stable: float    # FCF margin at end of projection (steady state)
    revenue_ttm: float          # TTM revenue
    wacc: float
    terminal_growth_rate: float = 0.025   # 2.5% perpetuity growth
    projection_years: int = 10
    shares_outstanding: float = 0.0
    net_debt: float = 0.0       # net debt (debt - cash); negative = net cash


@dataclass
class DCFResult:
    ticker: str
    intrinsic_value_equity: float       # total equity value
    intrinsic_value_per_share: float
    pv_fcfs: float
    pv_terminal: float
    terminal_value: float
    wacc: float
    terminal_growth_rate: float
    projected_fcfs: list
    discount_factors: list
    margin_of_safety_20: float          # price at 20% discount to IV
    margin_of_safety_30: float          # price at 30% discount to IV


def build_wacc(
    beta: float,
    risk_free_rate: float,
    equity_value: float,
    debt: float,
    cost_of_debt: float,
    tax_rate: float = TAX_RATE_DEFAULT,
    equity_risk_premium: float = EQUITY_RISK_PREMIUM,
) -> float:
    ke = risk_free_rate + beta * equity_risk_premium
    total = equity_value + debt
    if total <= 0:
        return ke
    we = equity_value / total
    wd = debt / total
    wacc = we * ke + wd * cost_of_debt * (1 - tax_rate)
    return wacc


def project_fcfs(
    revenue_ttm: float,
    fcf_margin_stable: float,
    revenue_growth_rates: list,
    current_fcf: float,
    projection_years: int,
) -> list:
    """
    Project FCF for each year.  Growth rates transition from historical/estimated
    rates toward stable-state FCF margin linearly.
    """
    years = projection_years
    rates = list(revenue_growth_rates)

    # Pad with last rate if shorter than projection window
    while len(rates) < years:
        rates.append(rates[-1] if rates else 0.05)

    fcfs = []
    rev = revenue_ttm
    for i in range(years):
        rev = rev * (1 + rates[i])
        # Blend from current FCF margin toward stable margin
        t = (i + 1) / years
        current_margin = current_fcf / revenue_ttm if revenue_ttm else 0.1
        blended_margin = current_margin * (1 - t) + fcf_margin_stable * t
        fcfs.append(rev * blended_margin)

    return fcfs


def run_dcf(inputs: DCFInputs) -> DCFResult:
    wacc = inputs.wacc
    g = inputs.terminal_growth_rate

    if wacc <= g:
        raise ValueError(f"WACC ({wacc:.2%}) must exceed terminal growth rate ({g:.2%})")

    projected_fcfs = project_fcfs(
        revenue_ttm=inputs.revenue_ttm,
        fcf_margin_stable=inputs.fcf_margin_stable,
        revenue_growth_rates=inputs.revenue_growth_rates,
        current_fcf=inputs.current_fcf,
        projection_years=inputs.projection_years,
    )

    discount_factors = [(1 / (1 + wacc) ** (i + 1)) for i in range(inputs.projection_years)]

    pv_fcfs_list = [fcf * df for fcf, df in zip(projected_fcfs, discount_factors)]
    pv_fcfs = sum(pv_fcfs_list)

    # Terminal value using perpetuity growth (Gordon Growth Model)
    terminal_fcf = projected_fcfs[-1] * (1 + g)
    terminal_value = terminal_fcf / (wacc - g)
    pv_terminal = terminal_value * discount_factors[-1]

    enterprise_value = pv_fcfs + pv_terminal
    equity_value = enterprise_value - inputs.net_debt

    per_share = (equity_value / inputs.shares_outstanding) if inputs.shares_outstanding > 0 else 0.0

    return DCFResult(
        ticker=inputs.ticker,
        intrinsic_value_equity=equity_value,
        intrinsic_value_per_share=per_share,
        pv_fcfs=pv_fcfs,
        pv_terminal=pv_terminal,
        terminal_value=terminal_value,
        wacc=wacc,
        terminal_growth_rate=g,
        projected_fcfs=projected_fcfs,
        discount_factors=discount_factors,
        margin_of_safety_20=per_share * 0.80,
        margin_of_safety_30=per_share * 0.70,
    )


def sensitivity_table(inputs: DCFInputs, wacc_range=None, tgr_range=None) -> pd.DataFrame:
    """Return intrinsic value per share across WACC × terminal growth rate grid."""
    if wacc_range is None:
        wacc_range = [inputs.wacc - 0.02, inputs.wacc - 0.01, inputs.wacc,
                      inputs.wacc + 0.01, inputs.wacc + 0.02]
    if tgr_range is None:
        tgr_range = [0.01, 0.02, inputs.terminal_growth_rate, 0.03, 0.04]

    rows = {}
    for tgr in tgr_range:
        row = {}
        for w in wacc_range:
            try:
                inp = DCFInputs(**{**inputs.__dict__, "wacc": w, "terminal_growth_rate": tgr})
                result = run_dcf(inp)
                row[f"{w:.1%}"] = round(result.intrinsic_value_per_share, 2)
            except Exception:
                row[f"{w:.1%}"] = float("nan")
        rows[f"{tgr:.1%}"] = row

    df = pd.DataFrame(rows).T
    df.index.name = "TGR \\ WACC"
    return df
