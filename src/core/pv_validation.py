"""Dynamic validation helpers for PV calculations."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Union

import numpy as np
import pandas as pd

from ..models.assumptions import SegmentAssumptions
from .discount import DiscountCurve
from .yield_curve import YieldCurve


def calculate_expected_pv_ratio(
    decay_rate: float,
    projection_months: int,
    discount_rate: float,
    avg_interest_rate: float,
) -> Dict[str, float]:
    """Return expected PV ratio and acceptable bounds based on assumptions."""
    years = projection_months / 12 if projection_months else 0.0

    # Remaining balance percentage after decay.
    remaining_pct = (1 - decay_rate) ** years if years > 0 else 1.0

    # Terminal value contribution (remaining balance discounted at projection end).
    terminal_discount = (1 / (1 + discount_rate) ** years) if years > 0 else 1.0
    terminal_contribution = remaining_pct * terminal_discount

    # Runoff (cash flow) contribution assuming average timing at midpoint.
    runoff_pct = 1 - remaining_pct
    avg_time = years / 2.0
    avg_discount = (1 / (1 + discount_rate) ** avg_time) if avg_time > 0 else 1.0
    cashflow_contribution = runoff_pct * avg_discount

    # Rough interest contribution estimate.
    total_interest = avg_interest_rate * years
    interest_contribution = total_interest * avg_discount * 0.5

    expected_ratio = terminal_contribution + cashflow_contribution + interest_contribution

    min_acceptable = max(0.0, terminal_contribution * 0.95)
    max_acceptable = min(1.10, expected_ratio * 1.15)

    return {
        "expected_ratio": expected_ratio,
        "terminal_contribution": terminal_contribution,
        "cashflow_contribution": cashflow_contribution,
        "interest_contribution": interest_contribution,
        "min_acceptable": min_acceptable,
        "max_acceptable": max_acceptable,
    }


def _resolve_discount_curve(
    source: Union[DiscountCurve, YieldCurve, float]
) -> DiscountCurve:
    """Normalise discount curve inputs to a DiscountCurve instance."""
    if isinstance(source, DiscountCurve):
        return source
    if isinstance(source, YieldCurve):
        return DiscountCurve(source)
    if isinstance(source, (int, float)):
        return DiscountCurve.from_single_rate(float(source))
    raise TypeError("discount_curve must be DiscountCurve, YieldCurve, or numeric rate.")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure canonical column names are present for validation."""
    rename_map = {
        "Account_ID": "account_id",
        "Current_Balance": "balance",
        "Balance": "balance",
        "Current_Interest_Rate": "interest_rate",
        "Interest_Rate": "interest_rate",
        "PV": "pv",
    }
    present = {col: rename_map[col] for col in rename_map if col in df.columns}
    return df.rename(columns=present)


def _average_decay_rate(
    assumptions: Mapping[str, SegmentAssumptions],
) -> float:
    """Compute the average decay rate across assumption segments."""
    if not assumptions:
        return 0.0
    values = [seg.decay_rate for seg in assumptions.values()]
    return float(np.mean(values))


def validate_pv_results(
    df_results: pd.DataFrame,
    df_original: pd.DataFrame,
    assumptions: Mapping[str, SegmentAssumptions],
    discount_curve: Union[DiscountCurve, YieldCurve, float],
    projection_months: int,
) -> Dict[str, object]:
    """Validate PV output against dynamic expectations."""
    if df_results.empty:
        return {
            "status": "FAIL",
            "failed_checks": ["no_results"],
            "warnings": ["No PV results supplied"],
        }

    results_norm = _normalise_columns(df_results.copy())
    orig_norm = _normalise_columns(df_original.copy())

    if "account_id" not in results_norm.columns:
        raise KeyError("Results dataframe must include 'account_id'")
    if "account_id" not in orig_norm.columns:
        raise KeyError("Original dataframe must include 'account_id'")

    merged = orig_norm.merge(
        results_norm[["account_id", "pv"]],
        on="account_id",
        how="left",
    )

    merged["pv_ratio"] = merged["pv"] / merged["balance"]

    avg_decay = _average_decay_rate(assumptions)
    avg_rate = float(merged["interest_rate"].mean() or 0.0)

    resolved_curve = _resolve_discount_curve(discount_curve)
    horizon_rate = resolved_curve.rate_for_month(projection_months)

    expected = calculate_expected_pv_ratio(
        decay_rate=avg_decay,
        projection_months=projection_months,
        discount_rate=horizon_rate,
        avg_interest_rate=avg_rate,
    )

    total_balance = float(merged["balance"].sum())
    total_pv = float(merged["pv"].sum())
    portfolio_ratio = (total_pv / total_balance) if total_balance else 0.0

    checks = {
        "no_negative_pv": (merged["pv"] >= 0).all(),
        "no_null_pv": merged["pv"].notna().all(),
        "all_accounts_processed": merged["pv"].notna().sum() == len(orig_norm),
        "terminal_value_included": portfolio_ratio > expected["min_acceptable"],
        "pv_ratio_reasonable": expected["min_acceptable"]
        <= portfolio_ratio
        <= expected["max_acceptable"],
        "no_extreme_outliers": (
            (merged["pv_ratio"] > 0.50) & (merged["pv_ratio"] < 1.20)
        ).all(),
    }

    warnings: list[str] = []
    ratio_diff = abs(portfolio_ratio - expected["expected_ratio"])
    if ratio_diff > 0.10:
        warnings.append(
            f"Portfolio ratio ({portfolio_ratio:.2%}) differs from expected "
            f"({expected['expected_ratio']:.2%}) by >10%."
        )

    if merged["pv_ratio"].std(ddof=0) > 0.20:
        warnings.append(
            f"High variance in PV ratios across accounts (std={merged['pv_ratio'].std(ddof=0):.2%})."
        )

    failed = [name for name, passed in checks.items() if not passed]

    result: Dict[str, object] = {
        "status": "PASS" if not failed else "FAIL",
        "failed_checks": failed,
        "warnings": warnings,
        "portfolio_balance": total_balance,
        "portfolio_pv": total_pv,
        "portfolio_ratio": portfolio_ratio,
        "expected_ratio": expected["expected_ratio"],
        "expected_range": f"{expected['min_acceptable']:.2%} - {expected['max_acceptable']:.2%}",
        "avg_account_ratio": float(merged["pv_ratio"].mean()),
        "accounts_processed": int(len(merged)),
        "expected_details": expected,
    }

    if not checks["terminal_value_included"]:
        result[
            "critical_error"
        ] = (
            "TERMINAL VALUE LIKELY NOT INCLUDED - "
            f"PV ratio ({portfolio_ratio:.2%}) is below minimum expected "
            f"({expected['min_acceptable']:.2%}) based on remaining balance after decay."
        )

    return result
