"""Discount curve utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class DiscountCurve:
    """Represents a term structure used for discounting cash flows."""

    annual_rates: Dict[int, float]

    @classmethod
    def from_single_rate(cls, rate: float) -> "DiscountCurve":
        """Create a flat discount curve."""
        if rate < 0 or rate > 0.15:
            raise ValueError("Discount rate must be between 0 and 15%")
        return cls(annual_rates={12: rate})

    @classmethod
    def from_yield_curve(cls, tenor_rates: Dict[int, float]) -> "DiscountCurve":
        """Create a curve from tenor-specific rates."""
        invalid = [tenor for tenor, rate in tenor_rates.items() if rate < 0 or rate > 0.15]
        if invalid:
            raise ValueError(
                "Invalid rate(s) detected for tenors: " + ", ".join(map(str, invalid))
            )
        return cls(annual_rates=dict(sorted(tenor_rates.items())))

    def rate_for_month(self, month: int) -> float:
        """Return the interpolated annualized rate for a given month."""
        if not self.annual_rates:
            raise ValueError("Discount curve is empty")
        tenors, rates = zip(*sorted(self.annual_rates.items()))
        months_array = np.array(tenors, dtype=float)
        rates_array = np.array(rates, dtype=float)
        if month <= months_array[0]:
            return float(rates_array[0])
        if month >= months_array[-1]:
            return float(rates_array[-1])
        return float(np.interp(month, months_array, rates_array))

    def discount_factor(self, month: int) -> float:
        """Return the discount factor for a given month."""
        rate = self.rate_for_month(month)
        return 1.0 / ((1 + rate) ** (month / 12))

    def iter_discount_factors(
        self, months: Iterable[int]
    ) -> Iterable[Tuple[int, float]]:
        """Yield discount factors for the requested months."""
        for month in months:
            yield month, self.discount_factor(month)
