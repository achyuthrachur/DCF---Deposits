"""High-level Plotly utilities for Monte Carlo visualisations."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import plotly.graph_objects as go

from .chart_components import (
    build_convergence_plot,
    build_distribution_figure,
    build_fan_chart,
    build_percentile_ladder,
    build_spaghetti_figure,
    build_yield_curve_animation,
)
from .dashboard_data import extract_monte_carlo_payload
from .monte_carlo_dashboard import create_dashboard_app, prepare_dashboard_data
from .themes import DEFAULT_THEME


def _theme(theme: Optional[str]) -> dict:
    if theme and theme.lower() == "dark":
        from .themes import DARK_THEME

        return DARK_THEME
    return DEFAULT_THEME


def extract_monte_carlo_data(results, scenario_id: str = "monte_carlo") -> Optional[Dict[str, object]]:
    """Backward-compatible wrapper returning structured payload."""
    return extract_monte_carlo_payload(results, scenario_id=scenario_id)


def plot_rate_path_spaghetti(
    rate_paths_sample: Optional[pd.DataFrame],
    rate_summary: pd.DataFrame,
    *,
    theme: str = "light",
) -> go.Figure:
    theme_cfg = _theme(theme)
    if rate_paths_sample is not None:
        cols = [col for col in rate_paths_sample.columns if col.startswith("month_")]
        paths = rate_paths_sample[cols].to_numpy(dtype=float)
    else:
        paths = rate_summary[["mean"]].to_numpy(dtype=float).T
    return build_spaghetti_figure(paths, rate_summary, theme=theme_cfg)


def plot_portfolio_pv_distribution(
    pv_distribution: pd.DataFrame,
    *,
    book_value: Optional[float] = None,
    base_case: Optional[float] = None,
    percentiles: Optional[Dict[str, float]] = None,
    theme: str = "light",
) -> go.Figure:
    return build_distribution_figure(
        pv_distribution,
        book_value=book_value,
        base_case=base_case,
        percentiles=percentiles or {},
        theme=_theme(theme),
        animate=False,
    )


def plot_rate_confidence_fan(
    rate_summary: pd.DataFrame,
    *,
    theme: str = "light",
) -> go.Figure:
    return build_fan_chart(rate_summary, theme=_theme(theme))


def plot_percentile_ladder(
    percentiles: Dict[str, float],
    *,
    book_value: Optional[float] = None,
    base_case: Optional[float] = None,
    theme: str = "light",
) -> go.Figure:
    return build_percentile_ladder(
        percentiles,
        book_value=book_value,
        base_case=base_case,
        theme=_theme(theme),
    )


def create_rate_path_animation(
    curve_records: pd.DataFrame,
    *,
    selected_simulations: Optional[list[int]] = None,
    theme: str = "light",
) -> go.Figure:
    return build_yield_curve_animation(
        curve_records,
        selected_simulations=selected_simulations,
        theme=_theme(theme),
    )


def create_monte_carlo_dashboard(results, scenario_id: str = "monte_carlo", *, theme: str = "light"):
    """Return a Dash app instance for the Monte Carlo dashboard."""
    data = prepare_dashboard_data(results, scenario_id=scenario_id)
    if data is None:
        return None
    return create_dashboard_app(results, scenario_id=scenario_id, theme=theme)


def build_convergence_figure(progress: pd.DataFrame, *, theme: str = "light") -> go.Figure:
    return build_convergence_plot(progress, theme=_theme(theme))


def save_figure(fig: go.Figure, path: str | Path, *, scale: float = 2.0) -> Path:
    """Persist a Plotly figure to disk using Kaleido (PNG)."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), scale=scale, engine="kaleido")
    return output_path
