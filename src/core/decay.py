"""Decay rate and weighted-average-life conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class DecayResolution:
    """Represents the resolved decay configuration for a segment."""

    annual_decay_rate: float
    monthly_decay_rate: float
    wal_years: float
    mode: str
    priority_used: str
    implied_wal_from_decay: Optional[float]
    implied_decay_from_wal: Optional[float]
    difference_years: Optional[float]
    difference_ratio: Optional[float]
    warning: Optional[str]
    inconsistent: bool


def convert_wal_to_decay_rate(
    target_wal_years: float,
    *,
    tolerance_years: float = 0.01,
    max_months: int = 2400,
    balance_threshold: float = 0.01,
    max_iterations: int = 10_000,
) -> Tuple[float, float, float]:
    """
    Convert a target WAL (in years) into the equivalent annual and monthly decay rate.

    Returns tuple of (annual_decay_rate, monthly_decay_rate, achieved_wal_years).
    """
    if target_wal_years <= 0:
        raise ValueError("Target WAL must be positive.")

    target_wal_months = target_wal_years * 12.0
    low_rate = 0.0001  # 0.01% monthly
    high_rate = 0.10   # 10% monthly

    def wal_for_monthly_rate(monthly_rate: float) -> float:
        runoff_schedule, _ = _generate_decay_schedule(
            monthly_rate,
            max_months=max_months,
            balance_threshold=balance_threshold,
        )
        wal_months = calculate_wal_from_runoffs(runoff_schedule)
        return wal_months

    wal_low = wal_for_monthly_rate(low_rate)
    wal_high = wal_for_monthly_rate(high_rate)

    if wal_low < target_wal_months:
        raise ValueError(
            "Target WAL is too short for the configured search bounds; "
            "increase the lower-rate bound."
        )
    if wal_high > target_wal_months:
        raise ValueError(
            "Target WAL is too long for the configured search bounds; "
            "increase the upper-rate bound."
        )

    iterations = 0
    solved_wal = wal_low
    monthly_rate = low_rate
    tolerance_months = max(tolerance_years * 12.0, 0.01)

    while iterations < max_iterations and (high_rate - low_rate) > 1e-8:
        monthly_rate = (low_rate + high_rate) / 2.0
        solved_wal = wal_for_monthly_rate(monthly_rate)
        if abs(solved_wal - target_wal_months) <= tolerance_months:
            break
        if solved_wal > target_wal_months:
            low_rate = monthly_rate
        else:
            high_rate = monthly_rate
        iterations += 1

    annual_rate = 1.0 - (1.0 - monthly_rate) ** 12
    return annual_rate, monthly_rate, solved_wal / 12.0


def convert_decay_rate_to_wal(
    annual_decay_rate: float,
    *,
    max_months: int = 2400,
    balance_threshold: float = 0.01,
) -> Tuple[float, float]:
    """
    Convert an annual decay rate into the implied WAL (years) and monthly rate.
    """
    if not 0.0 < annual_decay_rate < 1.0:
        raise ValueError("Annual decay rate must be between 0 and 1 (exclusive).")
    monthly_rate = 1.0 - (1.0 - annual_decay_rate) ** (1.0 / 12.0)
    runoff_schedule, _ = _generate_decay_schedule(
        monthly_rate,
        max_months=max_months,
        balance_threshold=balance_threshold,
    )
    wal_months = calculate_wal_from_runoffs(runoff_schedule)
    return wal_months / 12.0, monthly_rate


def calculate_wal_from_runoffs(runoff_schedule: Sequence[float]) -> float:
    """
    Calculate the weighted average life (in months) from a runoff schedule.
    """
    weighted_sum = 0.0
    total_runoff = 0.0
    for index, runoff in enumerate(runoff_schedule, start=1):
        weighted_sum += index * runoff
        total_runoff += runoff
    if total_runoff == 0:
        raise ValueError("Runoff schedule has zero total runoff; WAL undefined.")
    return weighted_sum / total_runoff


def resolve_decay_parameters(
    wal_years: Optional[float],
    annual_decay_rate: Optional[float],
    *,
    priority: str = "auto",
    mismatch_ratio_threshold: float = 0.10,
    tolerance_years: float = 0.01,
) -> DecayResolution:
    """
    Resolve WAL/decay inputs into a consistent decay profile.

    Returns DecayResolution with the final annual/monthly decay rates and derived WAL.
    """
    if wal_years is None and annual_decay_rate is None:
        raise ValueError("Either wal_years or annual_decay_rate must be provided.")

    priority_normalised = priority.lower().strip()
    if priority_normalised not in {"wal", "decay", "auto"}:
        raise ValueError("decay_priority must be one of {'wal', 'decay', 'auto'}.")

    if wal_years is not None and wal_years <= 0:
        raise ValueError("wal_years must be positive.")
    if annual_decay_rate is not None and not 0.0 < annual_decay_rate < 1.0:
        raise ValueError("decay_rate must be between 0 and 1 (exclusive).")

    if wal_years is not None and annual_decay_rate is None:
        annual_rate, monthly_rate, solved_wal_years = convert_wal_to_decay_rate(
            wal_years,
            tolerance_years=tolerance_years,
        )
        return DecayResolution(
            annual_decay_rate=annual_rate,
            monthly_decay_rate=monthly_rate,
            wal_years=solved_wal_years,
            mode="wal_only",
            priority_used="wal",
            implied_wal_from_decay=None,
            implied_decay_from_wal=annual_rate,
            difference_years=None,
            difference_ratio=None,
            warning=None,
            inconsistent=False,
        )

    if annual_decay_rate is not None and wal_years is None:
        implied_wal_years, monthly_rate = convert_decay_rate_to_wal(
            annual_decay_rate,
        )
        return DecayResolution(
            annual_decay_rate=annual_decay_rate,
            monthly_decay_rate=monthly_rate,
            wal_years=implied_wal_years,
            mode="decay_only",
            priority_used="decay",
            implied_wal_from_decay=implied_wal_years,
            implied_decay_from_wal=None,
            difference_years=None,
            difference_ratio=None,
            warning=None,
            inconsistent=False,
        )

    assert wal_years is not None and annual_decay_rate is not None
    implied_wal_years, monthly_rate_from_decay = convert_decay_rate_to_wal(
        annual_decay_rate,
    )
    annual_rate_from_wal, monthly_rate_from_wal, solved_wal_years = convert_wal_to_decay_rate(
        wal_years,
        tolerance_years=tolerance_years,
    )
    difference_years = abs(implied_wal_years - wal_years)
    difference_ratio = (
        difference_years / wal_years if wal_years > 0 else None
    )
    inconsistent = (
        difference_ratio is not None and difference_ratio > mismatch_ratio_threshold
    )

    warning_message: Optional[str] = None
    if inconsistent:
        warning_message = (
            f"Inconsistent WAL/decay inputs: WAL {wal_years:.2f} years implies "
            f"{annual_rate_from_wal * 100:.2f}% annual decay, but supplied decay rate "
            f"{annual_decay_rate * 100:.2f}% produces {implied_wal_years:.2f} year WAL."
        )

    priority_used = priority_normalised
    if priority_normalised == "auto":
        priority_used = "decay" if not inconsistent else "wal"

    if priority_used == "wal":
        annual_rate = annual_rate_from_wal
        monthly_rate = monthly_rate_from_wal
        wal_output = solved_wal_years
    elif priority_used == "decay":
        annual_rate = annual_decay_rate
        monthly_rate = monthly_rate_from_decay
        wal_output = implied_wal_years
    else:
        raise ValueError("Resolved decay priority could not be determined.")

    return DecayResolution(
        annual_decay_rate=annual_rate,
        monthly_decay_rate=monthly_rate,
        wal_years=wal_output,
        mode="both",
        priority_used=priority_used,
        implied_wal_from_decay=implied_wal_years,
        implied_decay_from_wal=annual_rate_from_wal,
        difference_years=difference_years,
        difference_ratio=difference_ratio,
        warning=warning_message,
        inconsistent=inconsistent,
    )


def _generate_decay_schedule(
    monthly_rate: float,
    *,
    max_months: int,
    balance_threshold: float,
    starting_balance: float = 100_000.0,
) -> Tuple[list[float], list[float]]:
    """Generate runoff schedule and balances for a constant monthly decay rate."""
    if monthly_rate <= 0.0 or monthly_rate >= 1.0:
        raise ValueError("monthly_rate must be between 0 and 1 (exclusive).")
    balances: list[float] = []
    runoffs: list[float] = []
    balance = float(starting_balance)
    for _ in range(1, max_months + 1):
        runoff = balance * monthly_rate
        runoffs.append(runoff)
        balances.append(balance)
        balance -= runoff
        if balance <= balance_threshold:
            break
    if balance > balance_threshold:
        runoffs.append(balance)
        balances.append(balance)
    return runoffs, balances


__all__ = [
    "DecayResolution",
    "calculate_wal_from_runoffs",
    "convert_decay_rate_to_wal",
    "convert_wal_to_decay_rate",
    "resolve_decay_parameters",
]
