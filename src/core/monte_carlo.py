"""Monte Carlo configuration and stochastic path generation utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from math import sqrt
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .discount import DiscountCurve
from .yield_curve import YieldCurve


class MonteCarloLevel(IntEnum):
    """Supported Monte Carlo implementation levels."""

    STATIC_CURVE = 1  # Level 1
    TWO_FACTOR = 2    # Level 2


@dataclass
class VasicekParams:
    """Parameters for the Vasicek short-rate model."""

    mean_reversion: float  # a
    long_term_mean: float  # b
    volatility: float      # sigma (annualised)
    initial_rate: Optional[float] = None  # starting point; filled at runtime if None

    def with_initial(self, rate: float) -> "VasicekParams":
        """Return a copy with the supplied initial rate."""
        return VasicekParams(
            mean_reversion=self.mean_reversion,
            long_term_mean=self.long_term_mean,
            volatility=self.volatility,
            initial_rate=rate,
        )


@dataclass
class MonteCarloConfig:
    """Configuration bundle for Monte Carlo simulations."""

    level: MonteCarloLevel = MonteCarloLevel.STATIC_CURVE
    num_simulations: int = 1000
    projection_months: int = 120
    random_seed: Optional[int] = 42
    short_rate: VasicekParams = field(
        default_factory=lambda: VasicekParams(mean_reversion=0.15, long_term_mean=0.03, volatility=0.01)
    )
    long_rate: Optional[VasicekParams] = field(default=None)
    correlation: float = 0.70
    save_rate_paths: bool = True
    generate_reports: bool = False
    sample_size: int = 100
    interpolation_method: str = "linear"

    def to_metadata(self) -> Dict[str, object]:
        """Serialise into scenario metadata for persistence."""
        payload: Dict[str, object] = {
            "level": int(self.level),
            "num_simulations": int(self.num_simulations),
            "projection_months": int(self.projection_months),
            "random_seed": self.random_seed,
            "short_rate": vars(self.short_rate),
            "correlation": float(self.correlation),
            "save_rate_paths": bool(self.save_rate_paths),
            "generate_reports": bool(self.generate_reports),
            "sample_size": int(self.sample_size),
            "interpolation_method": self.interpolation_method,
        }
        if self.long_rate:
            payload["long_rate"] = vars(self.long_rate)
        return payload

    @classmethod
    def from_metadata(cls, metadata: Dict[str, object]) -> "MonteCarloConfig":
        """Rehydrate a configuration from scenario metadata."""
        level = MonteCarloLevel(int(metadata.get("level", 1)))
        short_dict = dict(metadata.get("short_rate", {}))
        long_dict = metadata.get("long_rate")
        short_params = VasicekParams(
            mean_reversion=float(short_dict.get("mean_reversion", 0.15)),
            long_term_mean=float(short_dict.get("long_term_mean", 0.03)),
            volatility=float(short_dict.get("volatility", 0.01)),
            initial_rate=short_dict.get("initial_rate"),
        )
        long_params = None
        if long_dict is not None:
            long_params = VasicekParams(
                mean_reversion=float(long_dict.get("mean_reversion", 0.05)),
                long_term_mean=float(long_dict.get("long_term_mean", 0.035)),
                volatility=float(long_dict.get("volatility", 0.008)),
                initial_rate=long_dict.get("initial_rate"),
            )
        return MonteCarloConfig(
            level=level,
            num_simulations=int(metadata.get("num_simulations", 1000)),
            projection_months=int(metadata.get("projection_months", 120)),
            random_seed=metadata.get("random_seed"),
            short_rate=short_params,
            long_rate=long_params,
            correlation=float(metadata.get("correlation", 0.70)),
            save_rate_paths=bool(metadata.get("save_rate_paths", True)),
            generate_reports=bool(metadata.get("generate_reports", False)),
            sample_size=int(metadata.get("sample_size", 100)),
            interpolation_method=str(metadata.get("interpolation_method", "linear")),
        )


def generate_vasicek_path(params: VasicekParams, months: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single-factor Vasicek path with monthly time steps."""
    if params.initial_rate is None:
        raise ValueError("VasicekParams.initial_rate must be provided before simulation.")
    dt = 1.0 / 12.0
    path = np.empty(months, dtype=float)
    rate = float(params.initial_rate)
    path[0] = max(rate, 0.0)
    for idx in range(1, months):
        shock = rng.standard_normal()
        rate += params.mean_reversion * (params.long_term_mean - rate) * dt + params.volatility * sqrt(dt) * shock
        rate = max(rate, 0.0)
        path[idx] = rate
    return path


def generate_correlated_vasicek_paths(
    short_params: VasicekParams,
    long_params: VasicekParams,
    correlation: float,
    months: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate correlated short- and long-rate Vasicek paths."""
    if short_params.initial_rate is None or long_params.initial_rate is None:
        raise ValueError("Initial rates must be supplied for correlated Vasicek simulation.")

    dt = 1.0 / 12.0
    chol = np.linalg.cholesky(np.array([[1.0, correlation], [correlation, 1.0]]))

    short_path = np.empty(months, dtype=float)
    long_path = np.empty(months, dtype=float)

    short_rate = max(float(short_params.initial_rate), 0.0)
    long_rate = max(float(long_params.initial_rate), 0.0)
    short_path[0] = short_rate
    long_path[0] = long_rate

    for idx in range(1, months):
        z = rng.standard_normal(2)
        shocks = chol @ z

        short_rate += (
            short_params.mean_reversion * (short_params.long_term_mean - short_rate) * dt
            + short_params.volatility * sqrt(dt) * shocks[0]
        )
        long_rate += (
            long_params.mean_reversion * (long_params.long_term_mean - long_rate) * dt
            + long_params.volatility * sqrt(dt) * shocks[1]
        )

        short_rate = max(short_rate, 0.0)
        long_rate = max(long_rate, 0.0)
        short_path[idx] = short_rate
        long_path[idx] = long_rate

    return short_path, long_path


def interpolate_curve_rates(
    short_rate: float,
    long_rate: float,
    base_curve: DiscountCurve,
    *,
    short_tenor: int = 3,
    long_tenor: int = 120,
) -> Dict[int, float]:
    """
    Blend the base curve with simulated short/long anchors using linear deltas.

    Parameters
    ----------
    short_rate:
        Simulated rate (decimal) for the short tenor (e.g., 3M).
    long_rate:
        Simulated rate (decimal) for the long tenor (e.g., 10Y).
    base_curve:
        Discount curve used as reference.
    short_tenor:
        Tenor in months of the short anchor.
    long_tenor:
        Tenor in months of the long anchor.
    """
    base_short = base_curve.rate_for_month(short_tenor)
    base_long = base_curve.rate_for_month(long_tenor)
    short_delta = short_rate - base_short
    long_delta = long_rate - base_long

    adjusted: Dict[int, float] = {}
    tenors = sorted(base_curve.annual_rates.keys())
    for tenor in tenors:
        base_rate = base_curve.annual_rates[tenor]
        if tenor <= short_tenor:
            delta = short_delta
        elif tenor >= long_tenor:
            delta = long_delta
        else:
            weight = (tenor - short_tenor) / max(long_tenor - short_tenor, 1)
            delta = short_delta + weight * (long_delta - short_delta)
        adjusted_rate = max(base_rate + delta, 0.0)
        adjusted[int(tenor)] = adjusted_rate
    return adjusted


def expand_monthly_zero_rates(
    tenor_rates: Dict[int, float],
    months: int,
    *,
    interpolation_method: str = "linear",
) -> List[float]:
    """
    Derive zero rates for each month using an interpolated yield curve.

    Returns a list of length ``months`` where index ``m`` corresponds to month ``m+1``.
    """
    tenors = list(tenor_rates.keys())
    rates = list(tenor_rates.values())
    curve = YieldCurve(tenors, rates, interpolation_method=interpolation_method)
    return [float(curve.get_rate(month)) for month in range(1, months + 1)]


def build_percentile_table(
    values: Sequence[float],
    *,
    percentiles: Iterable[int] = range(1, 100, 1),
) -> pd.DataFrame:
    """Return a percentile ladder as a dataframe."""
    series = pd.Series(values, dtype=float)
    ladder = [{"percentile": p, "portfolio_pv": float(series.quantile(p / 100.0))} for p in percentiles]
    return pd.DataFrame(ladder)
