"""Backward-compatible discount curve wrapper built on YieldCurve."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

from .yield_curve import YieldCurve


class DiscountCurve(YieldCurve):
    """Compatibility wrapper around YieldCurve used throughout the codebase."""

    def __init__(
        self,
        tenors: Iterable[int],
        rates: Iterable[float],
        *,
        interpolation_method: str = "linear",
    ) -> None:
        super().__init__(tenors, rates, interpolation_method)
        self.annual_rates: Dict[int, float] = {
            int(t): float(r) for t, r in zip(self.tenors, self.rates)
        }

    @classmethod
    def from_single_rate(cls, rate: float, *, tenor_months: int = 360) -> "DiscountCurve":
        """Create a flat discount curve using a single annual rate."""
        if rate < 0 or rate > 0.20:
            raise ValueError("Discount rate must be between 0% and 20%.")
        tenors = [3, 6, 12, 24, 36, 60, 120, tenor_months]
        rates = [float(rate)] * len(tenors)
        return cls(tenors, rates, interpolation_method="linear")

    @classmethod
    def from_yield_curve(
        cls,
        tenor_rates: Dict[int, float],
        *,
        interpolation_method: str = "linear",
    ) -> "DiscountCurve":
        """Create a curve from tenor-specific annualised rates."""
        if not tenor_rates:
            raise ValueError("Tenor rates cannot be empty.")
        invalid = [
            tenor for tenor, value in tenor_rates.items() if float(value) < 0 or float(value) > 0.20
        ]
        if invalid:
            raise ValueError(
                f"Invalid rate(s) detected for tenors: {', '.join(map(str, invalid))}"
            )
        tenors, rates = zip(*sorted(tenor_rates.items(), key=lambda item: item[0]))
        return cls(tenors, rates, interpolation_method=interpolation_method)

    def rate_for_month(self, month: int | float) -> float:
        """Return the interpolated annualised rate for a given month (compat shim)."""
        return float(self.get_rate(month))

    def discount_factor(self, month: int | float) -> float:
        """Return the discount factor for a given month (compat shim)."""
        return float(self.get_discount_factor(month))

    def iter_discount_factors(self, months: Iterable[int]) -> Iterable[Tuple[int, float]]:
        """Yield discount factors for the requested months (compat shim)."""
        for month in months:
            yield int(month), float(self.get_discount_factor(month))

    def to_dict(self) -> Dict[int, float]:
        """Return tenor-rate mapping (compat shim)."""
        return dict(self.annual_rates)


__all__ = ["DiscountCurve"]
