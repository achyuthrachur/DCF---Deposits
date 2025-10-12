"""Present value calculation utilities."""

from __future__ import annotations

import pandas as pd

from .discount import DiscountCurve


class PresentValueCalculator:
    """Compute discounted values from projected cash flows."""

    def __init__(self, discount_curve: DiscountCurve) -> None:
        self.discount_curve = discount_curve

    def account_level_pv(self, cashflows: pd.DataFrame) -> pd.DataFrame:
        """Return account-level PV based on the provided cash flows."""
        if cashflows.empty:
            return pd.DataFrame(columns=["account_id", "pv"])
        cashflows = cashflows.copy()
        cashflows["discount_factor"] = cashflows["month"].map(
            self.discount_curve.discount_factor
        )
        cashflows["discounted_cf"] = cashflows["total_cash_flow"] * cashflows[
            "discount_factor"
        ]
        grouped = (
            cashflows.groupby("account_id", as_index=False)["discounted_cf"]
            .sum()
            .rename(columns={"discounted_cf": "pv"})
        )
        return grouped

    def portfolio_pv(self, cashflows: pd.DataFrame) -> float:
        """Aggregate discounted cash flows across the portfolio."""
        account_pv = self.account_level_pv(cashflows)
        return float(account_pv["pv"].sum()) if not account_pv.empty else 0.0
