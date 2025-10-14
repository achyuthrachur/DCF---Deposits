"""Yield curve helper supporting multiple interpolation schemes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Sequence

import numpy as np

try:  # SciPy is optional at runtime, but available via requirements.
    from scipy.interpolate import CubicSpline
except Exception:  # pragma: no cover - SciPy may not be installed in some envs
    CubicSpline = None  # type: ignore[assignment]


@dataclass
class YieldCurve:
    """Represents an annualised yield curve sampled at discrete tenors (months)."""

    tenors: Sequence[int]
    rates: Sequence[float]
    interpolation_method: str = "linear"
    metadata: Dict[str, object] = field(default_factory=dict, repr=False)
    _cubic_spline: CubicSpline | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        if len(self.tenors) != len(self.rates):
            raise ValueError("Tenor and rate sequences must be the same length.")
        if len(self.tenors) == 0:
            raise ValueError("Yield curve requires at least one tenor.")

        # Sort and normalise inputs
        pairs = sorted(zip(self.tenors, self.rates), key=lambda item: item[0])
        tenors_arr = np.array([int(max(item[0], 1)) for item in pairs], dtype=float)
        rates_arr = np.array([float(item[1]) for item in pairs], dtype=float)

        if np.any(np.isnan(rates_arr)) or np.any(np.isinf(rates_arr)):
            raise ValueError("Yield curve rates must be finite numbers.")

        self.tenors = tenors_arr
        self.rates = rates_arr
        self.interpolation_method = self.interpolation_method.lower().strip() or "linear"

        if self.interpolation_method not in {"linear", "log-linear", "cubic"}:
            raise ValueError(
                f"Unsupported interpolation method: {self.interpolation_method}"
            )

        if self.interpolation_method == "cubic":
            if CubicSpline is None:
                raise RuntimeError("scipy must be installed for cubic interpolation.")
            if len(self.tenors) < 4:
                raise ValueError(
                    "Cubic interpolation requires at least four tenor points."
                )
            # Natural spline avoids exaggerated end-point curvature.
            self._cubic_spline = CubicSpline(self.tenors, self.rates, bc_type="natural")

    def get_rate(self, months: int | float | Sequence[float]) -> float | np.ndarray:
        """Return the annualised rate(s) corresponding to the requested month(s)."""
        values = np.atleast_1d(np.asarray(months, dtype=float))
        if np.any(values <= 0):
            raise ValueError("Month values must be positive.")

        if self.interpolation_method == "linear":
            interpolated = np.interp(values, self.tenors, self.rates, left=self.rates[0], right=self.rates[-1])
        elif self.interpolation_method == "log-linear":
            log_rates = np.log1p(self.rates)
            interpolated = np.expm1(
                np.interp(values, self.tenors, log_rates, left=log_rates[0], right=log_rates[-1])
            )
        else:  # cubic
            assert self._cubic_spline is not None  # for type-checking
            interpolated = self._cubic_spline(values)

        interpolated = np.maximum(interpolated, 0.0)
        if np.isscalar(months):
            return float(interpolated.item())
        return interpolated

    def get_discount_factor(self, months: int | float | Sequence[float]) -> float | np.ndarray:
        """Return discount factor(s) for the given maturity month(s)."""
        rates = self.get_rate(months)
        periods = np.atleast_1d(np.asarray(months, dtype=float)) / 12.0
        factors = 1.0 / np.power(1.0 + np.atleast_1d(rates), periods)
        if np.isscalar(months):
            return float(factors.item())
        return factors

    def iter_rates(self, months: Iterable[int]) -> Dict[int, float]:
        """Return a mapping of month -> interpolated rate."""
        return {int(month): float(self.get_rate(int(month))) for month in months}

    def to_discount_curve(self) -> Dict[int, float]:
        """Return tenor-rate dictionary suitable for DiscountCurve construction."""
        return {int(month): float(rate) for month, rate in zip(self.tenors, self.rates)}

    def apply_parallel_shock(self, shock_bps: float) -> "YieldCurve":
        """Return a new yield curve with a parallel shock applied (in basis points)."""
        shock_decimal = float(shock_bps) / 10000.0
        shocked_rates = np.maximum(self.rates + shock_decimal, 0.0)
        return YieldCurve(self.tenors, shocked_rates, self.interpolation_method, metadata=dict(self.metadata))

    def apply_non_parallel_shock(self, shocks_by_tenor: Dict[int, float]) -> "YieldCurve":
        """Return a new yield curve with tenor-specific shocks applied (in basis points)."""
        shocked_rates = []
        for tenor, rate in zip(self.tenors, self.rates):
            shock = float(shocks_by_tenor.get(int(tenor), 0.0)) / 10000.0
            shocked_rates.append(max(rate + shock, 0.0))
        return YieldCurve(self.tenors, shocked_rates, self.interpolation_method, metadata=dict(self.metadata))

    def to_dict(self) -> Dict[str, object]:
        """Export curve configuration as a serialisable dictionary."""
        return {
            "tenors": [int(t) for t in self.tenors],
            "rates": [float(r) for r in self.rates],
            "interpolation_method": self.interpolation_method,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        rows = "\n".join(
            f"  {int(tenor):>3}M: {rate * 100:.3f}%" for tenor, rate in zip(self.tenors, self.rates)
        )
        return f"YieldCurve(\n{rows}\n)"


__all__ = ["YieldCurve"]
