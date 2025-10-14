"""Utilities for downsampling high-density Monte Carlo data for plotting."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def downsample_paths(
    paths: np.ndarray,
    target_paths: int = 300,
) -> np.ndarray:
    """
    Reduce the number of simulation paths to display while preserving structure.

    Parameters
    ----------
    paths:
        2D array of shape (num_simulations, horizon_months).
    target_paths:
        Maximum number of paths to retain for plotting.
    """
    num_simulations = paths.shape[0]
    if num_simulations <= target_paths:
        return paths
    indices = np.linspace(0, num_simulations - 1, target_paths).astype(int)
    return paths[indices]


def summarise_distribution(values: Iterable[float]) -> Tuple[float, float, float, float]:
    """Return mean, std dev, p5, p95 for the provided series."""
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if arr.size > 1 else 0.0)
    p05 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    return mean, std, p05, p95
