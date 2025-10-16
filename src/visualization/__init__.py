"""Visualization utilities for ALM analytics."""

from __future__ import annotations

import logging
from typing import Callable

from .monte_carlo_plots import (
    extract_monte_carlo_data,
    plot_percentile_ladder,
    plot_portfolio_pv_distribution,
    plot_rate_confidence_fan,
    plot_rate_path_spaghetti,
    create_monte_carlo_dashboard,
    create_rate_path_animation,
)

__all__ = [
    "extract_monte_carlo_data",
    "plot_percentile_ladder",
    "plot_portfolio_pv_distribution",
    "plot_rate_confidence_fan",
    "plot_rate_path_spaghetti",
    "create_monte_carlo_dashboard",
    "create_rate_path_animation",
]


def _missing(name: str, exc: Exception) -> Callable[..., None]:
    def _raiser(*_args, **_kwargs) -> None:
        raise RuntimeError(
            f"Visualization helper '{name}' is unavailable because: {exc}"
        ) from exc

    return _raiser


try:
    from .shock_plots import (
        extract_shock_data,
        plot_shock_rate_paths,
        plot_shock_magnitude,
        plot_shock_tenor_comparison,
        plot_shock_pv_delta,
    )
except Exception as exc:  # pragma: no cover - graceful degradation
    logging.getLogger(__name__).warning(
        "Shock visualization helpers disabled: %s", exc
    )
    extract_shock_data = _missing("extract_shock_data", exc)  # type: ignore[assignment]
    plot_shock_rate_paths = _missing("plot_shock_rate_paths", exc)  # type: ignore[assignment]
    plot_shock_magnitude = _missing("plot_shock_magnitude", exc)  # type: ignore[assignment]
    plot_shock_tenor_comparison = _missing("plot_shock_tenor_comparison", exc)  # type: ignore[assignment]
    plot_shock_pv_delta = _missing("plot_shock_pv_delta", exc)  # type: ignore[assignment]
else:
    __all__.extend(
        [
            "extract_shock_data",
            "plot_shock_rate_paths",
            "plot_shock_magnitude",
            "plot_shock_tenor_comparison",
            "plot_shock_pv_delta",
        ]
    )
