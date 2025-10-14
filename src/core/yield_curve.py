"""Yield curve utilities with interpolation and shock support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Union

import numpy as np
from scipy.interpolate import interp1d


@dataclass(frozen=True)
class YieldCurve:
    """
    Represents a yield curve with interpolation capability.

    Provides rate lookups, discount factor calculations, and common shock
    transformations used within the ALM engine.
    """

    tenors: np.ndarray
    rates: np.ndarray
    interpolation_method: str = "linear"

    def __init__(
        self,
        tenors: Union[Sequence[float], np.ndarray],
        rates: Union[Sequence[float], np.ndarray],
        interpolation_method: str = "linear",
    ) -> None:
        object.__setattr__(self, "tenors", np.array(tenors, dtype=float))
        object.__setattr__(self, "rates", np.array(rates, dtype=float))
        object.__setattr__(self, "interpolation_method", interpolation_method)

        if self.tenors.shape != self.rates.shape:
            raise ValueError("Tenors and rates must have same length")
        if self.tenors.size < 2:
            raise ValueError("Need at least 2 data points for interpolation")

        # Sort by tenor to ensure monotonic interpolation inputs.
        sort_idx = np.argsort(self.tenors)
        sorted_tenors = self.tenors[sort_idx]
        sorted_rates = self.rates[sort_idx]

        object.__setattr__(self, "tenors", sorted_tenors)
        object.__setattr__(self, "rates", sorted_rates)

        method = interpolation_method.lower()
        if method == "cubic":
            interpolator = interp1d(
                sorted_tenors,
                sorted_rates,
                kind="cubic",
                fill_value="extrapolate",
            )
            object.__setattr__(self, "_interpolation_mode", "standard")
        elif method == "log-linear":
            transformed = np.log1p(sorted_rates)
            interpolator = interp1d(
                sorted_tenors,
                transformed,
                kind="linear",
                fill_value="extrapolate",
            )
            object.__setattr__(self, "_interpolation_mode", "log-linear")
        else:
            interpolator = interp1d(
                sorted_tenors,
                sorted_rates,
                kind="linear",
                fill_value="extrapolate",
            )
            object.__setattr__(self, "_interpolation_mode", "standard")

        object.__setattr__(self, "interpolator", interpolator)

    # ------------------------------------------------------------------ Helpers
    def _interpolate_rates(self, months: np.ndarray) -> np.ndarray:
        """Return interpolated rates honouring the configured interpolation mode."""
        values = self.interpolator(months)
        if getattr(self, "_interpolation_mode", "standard") == "log-linear":
            values = np.expm1(values)
        return np.maximum(values, 0.0)

    # ----------------------------------------------------------------- Public API
    def get_rate(self, months: Union[float, Sequence[float], np.ndarray]) -> Union[float, np.ndarray]:
        """
        Return the interpolated rate for the requested maturity in months.

        Scalar inputs return a float; array inputs return an ndarray.
        """
        months_array = np.atleast_1d(months).astype(float)
        rates = self._interpolate_rates(months_array)
        if rates.shape == (1,) and np.isscalar(months):
            return float(rates[0])
        if rates.shape == (1,) and np.ndim(months) == 0:
            return float(rates[0])
        return rates

    def get_discount_factor(
        self, months: Union[float, Sequence[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Return discount factor(s) for the requested maturity in months.
        """
        months_array = np.atleast_1d(months).astype(float)
        rates = self._interpolate_rates(months_array)
        years = months_array / 12.0
        discount = np.power(1.0 + rates, -years)
        if discount.shape == (1,) and np.ndim(months) == 0:
            return float(discount[0])
        return discount

    def apply_parallel_shock(self, shock_bps: float) -> "YieldCurve":
        """
        Apply a parallel shock in basis points and return a shocked curve.
        """
        shock_decimal = shock_bps / 10000.0
        shocked_rates = np.maximum(self.rates + shock_decimal, 0.0)
        return YieldCurve(
            tenors=self.tenors,
            rates=shocked_rates,
            interpolation_method=self.interpolation_method,
        )

    def apply_non_parallel_shock(self, shocks_by_tenor: Dict[float, float]) -> "YieldCurve":
        """
        Apply tenor-specific shocks (in basis points) to the curve.
        """
        shocked_rates = []
        for tenor, rate in zip(self.tenors, self.rates):
            shock_bps = shocks_by_tenor.get(float(tenor), 0.0)
            shocked_rate = max(0.0, rate + shock_bps / 10000.0)
            shocked_rates.append(shocked_rate)
        return YieldCurve(
            tenors=self.tenors,
            rates=np.array(shocked_rates, dtype=float),
            interpolation_method=self.interpolation_method,
        )

    def iter_discount_factors(self, months: Iterable[int]) -> Iterable[tuple[int, float]]:
        """Yield (month, discount factor) tuples."""
        for month in months:
            yield month, float(self.get_discount_factor(month))

    def to_dict(self) -> Dict[str, object]:
        """Export curve data as a serialisable dictionary."""
        return {
            "tenors": self.tenors.tolist(),
            "rates": self.rates.tolist(),
            "interpolation_method": self.interpolation_method,
        }

    # ---------------------------------------------------------------- Dunder API
    def __repr__(self) -> str:
        parts = [f"  {int(t)}M: {r * 100:.2f}%" for t, r in zip(self.tenors, self.rates)]
        header = f"YieldCurve(interpolation='{self.interpolation_method}')"
        return f"{header}\n" + "\n".join(parts)


# ---------------------------------------------------------------------- Testing
def test_yield_curve() -> bool:
    """Run a quick validation of the YieldCurve implementation."""
    tenors = [3, 6, 12, 24, 36, 60, 84, 120]
    rates = [0.0425, 0.0415, 0.0410, 0.0395, 0.0385, 0.0380, 0.0395, 0.0415]
    curve = YieldCurve(tenors, rates)

    assert abs(curve.get_rate(12) - 0.0410) < 1e-9, "Exact tenor lookup failed"

    rate_18m = curve.get_rate(18)
    expected_rate = (0.0410 + 0.0395) / 2
    assert abs(rate_18m - expected_rate) < 0.001, f"Expected {expected_rate}, got {rate_18m}"

    df_24m = curve.get_discount_factor(24)
    expected_df = 1 / (1 + 0.0395) ** 2
    assert abs(df_24m - expected_df) < 0.0001, f"Expected {expected_df}, got {df_24m}"

    shocked_curve = curve.apply_parallel_shock(100)
    assert abs(shocked_curve.get_rate(12) - 0.0510) < 1e-9, "Parallel shock failed"

    steepener = curve.apply_non_parallel_shock({3: -100, 12: -50, 120: 100})
    assert abs(steepener.get_rate(3) - 0.0325) < 1e-9, "Steepener short-end shock failed"
    assert abs(steepener.get_rate(120) - 0.0515) < 1e-9, "Steepener long-end shock failed"

    return True
