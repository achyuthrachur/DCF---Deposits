"""Validation helpers for Monte Carlo simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class ValidationResult:
    """Basic container for validation outcomes."""

    status: str
    failed_checks: Sequence[str]
    warnings: Sequence[str]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "failed_checks": list(self.failed_checks),
            "warnings": list(self.warnings),
        }


def validate_rate_paths(
    short_paths: Sequence[np.ndarray],
    *,
    long_paths: Optional[Sequence[np.ndarray]] = None,
    long_term_mean: Optional[float] = None,
) -> ValidationResult:
    """Run basic sanity checks on simulated rate paths."""
    failed: list[str] = []
    warnings: list[str] = []

    all_paths = list(short_paths)
    if long_paths:
        all_paths += list(long_paths)
    if not all_paths:
        return ValidationResult(status="PASS", failed_checks=failed, warnings=warnings)

    stacked = np.vstack(all_paths)
    if np.any(np.isnan(stacked)) or np.any(np.isinf(stacked)):
        failed.append("nan_or_inf_rates")
    if float(stacked.max(initial=0.0)) > 0.20:
        warnings.append("rates_exceed_20pct")
    if float(stacked.min(initial=0.0)) < 0.0:
        warnings.append("negative_rates_generated")

    if long_term_mean is not None:
        final_mean = float(np.mean(stacked[:, -1]))
        if abs(final_mean - long_term_mean) > 0.015:
            warnings.append("mean_reversion_off_target")

    status = "PASS" if not failed else "FAIL"
    return ValidationResult(status=status, failed_checks=failed, warnings=warnings)


def validate_distribution(
    pv_values: Sequence[float],
    *,
    book_value: Optional[float] = None,
) -> ValidationResult:
    """Validate ordering and dispersion of Monte Carlo PV results."""
    failed: list[str] = []
    warnings: list[str] = []
    if not pv_values:
        failed.append("no_pv_values")
        return ValidationResult(status="FAIL", failed_checks=failed, warnings=warnings)

    series = pd.Series(pv_values, dtype=float)
    percentiles = series.quantile([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    if not percentiles.is_monotonic_increasing:
        failed.append("percentile_ordering")

    std = float(series.std(ddof=1))
    mean = float(series.mean())
    if std <= 0:
        failed.append("non_positive_std")
    elif mean != 0 and std / abs(mean) > 0.20:
        warnings.append("high_volatility")

    if book_value is not None:
        if float(series.min()) < 0.5 * book_value:
            warnings.append("extreme_lower_tail")
        if float(series.max()) > 1.5 * book_value:
            warnings.append("extreme_upper_tail")
        prob_below = float((series < book_value).mean())
        if prob_below < 0.05 or prob_below > 0.95:
            warnings.append("probability_below_book_extreme")

    status = "PASS" if not failed else "FAIL"
    return ValidationResult(status=status, failed_checks=failed, warnings=warnings)


__all__ = ["ValidationResult", "validate_rate_paths", "validate_distribution"]
