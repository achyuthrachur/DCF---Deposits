"""Animation helper utilities for Plotly charts."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def build_frame_names(month_count: int) -> List[str]:
    """Return standard frame identifiers for month-by-month animations."""
    return [f"frame_{month:03d}" for month in range(1, month_count + 1)]


def build_time_axis(months: int) -> np.ndarray:
    """Return a numpy array representing time in months."""
    return np.arange(1, months + 1)


def ease_in_out(values: Iterable[float], steps: int = 10) -> List[float]:
    """
    Apply a simple ease-in-out easing to smooth transitions.

    Useful for smoothing gauge movements or statistic updates.
    """
    arr = np.linspace(0, 1, steps)
    easing = arr * arr * (3 - 2 * arr)
    delta = float(values[-1] - values[0])
    return [float(values[0] + delta * e) for e in easing]
