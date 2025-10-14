"""Correlation heatmap for Monte Carlo rate paths."""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from ..themes import DEFAULT_THEME


def build_correlation_heatmap(
    rate_paths: np.ndarray,
    *,
    theme: Optional[dict] = None,
) -> go.Figure:
    """
    Display correlation matrix between rate tenors.

    Parameters
    ----------
    rate_paths:
        Array of shape (num_simulations, months). Correlation is computed across simulations.
    """
    theme = theme or DEFAULT_THEME
    corr_matrix = np.corrcoef(rate_paths, rowvar=False)
    months = np.arange(1, rate_paths.shape[1] + 1)

    heatmap = go.Heatmap(
        z=corr_matrix,
        x=months,
        y=months,
        colorscale="RdBu",
        zmin=-1,
        zmax=1,
        hovertemplate="Month %{y} vs %{x}<br>ρ %{z:.2f}<extra></extra>",
    )

    fig = go.Figure(heatmap)
    fig.update_layout(
        template=theme["plotly_template"],
        title="Rate Path Correlation Heatmap",
        xaxis=dict(title="Month"),
        yaxis=dict(title="Month"),
        margin=dict(l=80, r=30, t=60, b=60),
    )
    return fig
