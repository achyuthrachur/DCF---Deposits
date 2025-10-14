"""Utilities for comparing single-rate and yield-curve discounting results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.core.pv_calculator import PresentValueCalculator
from src.core.yield_curve import YieldCurve


@dataclass
class YieldCurveImpactSummary:
    """Container for PV comparison diagnostics."""

    pv_single_rate: float
    pv_yield_curve: float
    difference: float
    pct_difference: float
    account_count: int
    sample_size: int


def validate_yield_curve_impact(
    cashflows: pd.DataFrame,
    single_rate: float,
    yield_curve: YieldCurve,
    *,
    sample_accounts: int = 100,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Compare PV calculations using a flat discount rate vs. a full yield curve.

    Parameters
    ----------
    cashflows:
        Cashflow dataframe produced by the engine (requires account_id, month,
        total_cash_flow, ending_balance columns).
    single_rate:
        Annual decimal rate used in the legacy single-rate approach.
    yield_curve:
        YieldCurve instance used for the enhanced method.
    sample_accounts:
        Optional cap on the number of accounts to evaluate (random sample).
    random_seed:
        Optional seed for reproducible sampling.
    """
    if cashflows.empty:
        raise ValueError("cashflows dataframe is empty.")
    required_cols = {"account_id", "month", "total_cash_flow", "ending_balance"}
    missing = required_cols.difference(cashflows.columns)
    if missing:
        raise KeyError(f"Cashflows dataframe missing required columns: {sorted(missing)}")

    accounts = cashflows["account_id"].unique()
    if sample_accounts and len(accounts) > sample_accounts:
        rng = np.random.default_rng(random_seed)
        sampled_accounts = rng.choice(accounts, size=sample_accounts, replace=False)
        cf_sample = cashflows[cashflows["account_id"].isin(sampled_accounts)].copy()
    else:
        cf_sample = cashflows.copy()
        sampled_accounts = accounts

    single_calc = PresentValueCalculator(single_rate)
    curve_calc = PresentValueCalculator(yield_curve)

    pv_single = single_calc.account_level_pv(cf_sample)
    pv_curve = curve_calc.account_level_pv(cf_sample)

    merged = pv_single.merge(
        pv_curve,
        on="account_id",
        suffixes=("_single", "_curve"),
    )
    merged["difference"] = merged["pv_curve"] - merged["pv_single"]
    merged["pct_difference"] = np.where(
        merged["pv_single"] != 0,
        (merged["difference"] / merged["pv_single"]) * 100,
        0.0,
    )

    total_single = float(merged["pv_single"].sum())
    total_curve = float(merged["pv_curve"].sum())
    total_diff = total_curve - total_single
    pct_diff = (total_diff / total_single * 100) if total_single else 0.0

    summary = YieldCurveImpactSummary(
        pv_single_rate=total_single,
        pv_yield_curve=total_curve,
        difference=total_diff,
        pct_difference=pct_diff,
        account_count=len(merged),
        sample_size=len(sampled_accounts),
    )

    print("=" * 80)
    print("VALIDATION: Single Rate vs. Yield Curve Impact")
    print("=" * 80)
    print(f"Sample accounts evaluated: {summary.account_count} (of {len(accounts)})")
    print(f"Total PV (single rate):   ${summary.pv_single_rate:,.0f}")
    print(f"Total PV (yield curve):   ${summary.pv_yield_curve:,.0f}")
    print(f"Difference:               ${summary.difference:,.0f} ({summary.pct_difference:+.2f}%)")
    print("\nDistribution of account-level differences (% change):")
    print(
        f"  Min:  {merged['pct_difference'].min():+.2f}%  "
        f"25th: {merged['pct_difference'].quantile(0.25):+.2f}%  "
        f"Median: {merged['pct_difference'].median():+.2f}%  "
        f"75th: {merged['pct_difference'].quantile(0.75):+.2f}%  "
        f"Max: {merged['pct_difference'].max():+.2f}%"
    )

    return {
        "summary": summary,
        "account_detail": merged,
    }


def calculate_with_logging(
    curve: YieldCurve,
    projection_months: int,
    *,
    log_months: Optional[Iterable[int]] = None,
) -> None:
    """
    Print spot rates and discount factors for selected months to aid debugging.
    """
    months_to_log: List[int] = sorted(set(log_months or [1, 6, 12, 24, projection_months]))
    print("\nYield curve diagnostics:")
    for month in months_to_log:
        rate = curve.get_rate(month)
        discount_factor = curve.get_discount_factor(month)
        print(
            f"  Month {month:>3}: rate={rate * 100:6.3f}%  "
            f"discount_factor={discount_factor:0.6f}"
        )
