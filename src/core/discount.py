"""Discount curve compatibility wrappers built on top of YieldCurve."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

from .yield_curve import YieldCurve


@dataclass
class DiscountCurve:
    """
    Backwards-compatible wrapper around the new YieldCurve implementation.

    The ALM engine historically referenced DiscountCurve directly; this wrapper
    preserves that interface while delegating calculations to YieldCurve.
    """

    yield_curve: YieldCurve

    @classmethod
    def from_single_rate(
        cls,
        rate: float,
        *,
        interpolation_method: str = "linear",
    ) -> "DiscountCurve":
        """Create a flat discount curve using two anchor tenors."""
        if rate < 0 or rate > 0.15:
            raise ValueError("Discount rate must be between 0 and 15%")
        # Use short and long anchors to satisfy interpolation requirements.
        tenors = [1, 600]
        rates = [rate, rate]
        return cls(YieldCurve(tenors, rates, interpolation_method))

    @classmethod
    def from_yield_curve(
        cls,
        tenor_rates: Dict[int, float],
        *,
        interpolation_method: str = "linear",
    ) -> "DiscountCurve":
        """Create a curve from tenor-specific rates."""
        if not tenor_rates:
            raise ValueError("At least one tenor rate is required.")
        invalid = [
            tenor for tenor, rate in tenor_rates.items() if rate < 0 or rate > 0.15
        ]
        if invalid:
            raise ValueError(
                "Invalid rate(s) detected for tenors: " + ", ".join(map(str, invalid))
            )
        tenors, rates = zip(*sorted(tenor_rates.items()))
        return cls(YieldCurve(tenors, rates, interpolation_method))

    # ---------------------------------------------------------------- Utilities
    @property
    def annual_rates(self) -> Dict[int, float]:
        """Return a tenor -> rate mapping for compatibility with legacy code."""
        return {
            int(tenor): float(rate)
            for tenor, rate in zip(self.yield_curve.tenors, self.yield_curve.rates)
        }

    def rate_for_month(self, month: int) -> float:
        """Return the interpolated annualised rate for a given month."""
        if month <= 0:
            month = 1
        return float(self.yield_curve.get_rate(month))

    def discount_factor(self, month: int) -> float:
        """Return the discount factor for a given month."""
        if month <= 0:
            month = 1
        return float(self.yield_curve.get_discount_factor(month))

    def iter_discount_factors(
        self, months: Iterable[int]
    ) -> Iterable[Tuple[int, float]]:
        """Yield discount factors for the requested months."""
        for month in months:
            yield month, self.discount_factor(month)
