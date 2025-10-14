"""Data models related to Monte Carlo simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import time


@dataclass(frozen=True)
class MonteCarloProgressEvent:
    """Represents a single Monte Carlo simulation progress update."""

    scenario_id: str
    simulation_index: int
    total_simulations: int
    present_value: float
    cumulative_mean: float
    cumulative_std: float
    percentile_estimates: Dict[str, float]
    rate_path: Optional[List[float]] = None
    timestamp: float = field(default_factory=time.time)


__all__ = ["MonteCarloProgressEvent"]
