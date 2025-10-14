"""Convergence plot showing stability of statistics over simulations."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.graph_objects as go

from ..themes import DEFAULT_THEME


def build_convergence_plot(
    progress: pd.DataFrame,
    *,
    theme: Optional[dict] = None,
) -> go.Figure:
    """
    Build convergence chart for mean/median PV across simulations.
    """
    theme = theme or DEFAULT_THEME
    palette = theme["palette"]
    df = progress.sort_values("simulation")

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=df["simulation"],
            y=df["cumulative_mean"],
            mode="lines",
            line=dict(color=palette["primary_blue"], width=3),
            name="Mean",
            hovertemplate="Simulation %{x}<br>Mean %{y:,.0f}<extra></extra>",
        )
    )

    if {"p50"}.issubset(df.columns):
        figure.add_trace(
            go.Scatter(
                x=df["simulation"],
                y=df["p50"],
                mode="lines",
                line=dict(color=palette["secondary_teal"], width=2, dash="dash"),
                name="Median",
                hovertemplate="Simulation %{x}<br>Median %{y:,.0f}<extra></extra>",
            )
        )

    figure.update_layout(
        template=theme["plotly_template"],
        title="Convergence of Portfolio PV",
        margin=dict(l=60, r=30, t=60, b=40),
        xaxis=dict(title="Simulations completed"),
        yaxis=dict(title="Present Value ($)", tickformat="~s"),
        hovermode="x unified",
    )

    return figure
