"""Present value calculation utilities."""

from __future__ import annotations

from typing import Union

import pandas as pd

from .discount import DiscountCurve
from .yield_curve import YieldCurve


class PresentValueCalculator:
    """Compute discounted values from projected cash flows."""

    def __init__(
        self,
        discount_source: Union[DiscountCurve, YieldCurve, float],
        *,
        interpolation_method: str = "linear",
    ) -> None:
        if isinstance(discount_source, DiscountCurve):
            self.discount_curve = discount_source
        elif isinstance(discount_source, YieldCurve):
            self.discount_curve = DiscountCurve(discount_source)
        elif isinstance(discount_source, (int, float)):
            self.discount_curve = DiscountCurve.from_single_rate(
                float(discount_source),
                interpolation_method=interpolation_method,
            )
        else:
            raise TypeError(
                "discount_source must be a DiscountCurve, YieldCurve, or numeric rate."
            )

    @property
    def yield_curve(self) -> YieldCurve:
        """Expose the underlying yield curve."""
        return self.discount_curve.yield_curve

    def account_level_pv(self, cashflows: pd.DataFrame) -> pd.DataFrame:
        """Return account-level PV based on the provided cash flows."""
        if cashflows.empty:
            return pd.DataFrame(columns=["account_id", "pv"])
        cashflows = cashflows.copy()
        if "ending_balance" not in cashflows.columns:
            raise KeyError("Cashflow dataframe must include 'ending_balance' column for terminal value.")

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

        terminal = (
            cashflows.sort_values("month")
            .groupby("account_id", as_index=False)
            .agg({"ending_balance": "last", "month": "max"})
        )
        terminal["terminal_df"] = terminal["month"].map(
            self.discount_curve.discount_factor
        )
        terminal["terminal_pv"] = terminal["ending_balance"] * terminal["terminal_df"]

        grouped = grouped.merge(
            terminal[["account_id", "terminal_pv"]],
            on="account_id",
            how="left",
        )
        grouped["pv"] = grouped["pv"] + grouped["terminal_pv"].fillna(0.0)
        grouped = grouped.drop(columns="terminal_pv")

        return grouped

    def portfolio_pv(self, cashflows: pd.DataFrame) -> float:
        """Aggregate discounted cash flows across the portfolio."""
        account_pv = self.account_level_pv(cashflows)
        return float(account_pv["pv"].sum()) if not account_pv.empty else 0.0
