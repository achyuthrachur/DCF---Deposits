"""Interactive fan chart (confidence bands over time)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..themes import DEFAULT_THEME


def build_fan_chart(
    rate_summary: pd.DataFrame,
    *,
    theme: Optional[dict] = None,
) -> go.Figure:
    """
    Construct a fan chart with progressive shading for confidence intervals.
    """
    theme = theme or DEFAULT_THEME
    palette = theme["palette"]
    months = rate_summary["month"].to_numpy(dtype=int)
    rate_summary_pct = rate_summary.copy()
    percent_cols = [
        "p01",
        "p05",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    ]
    for col in percent_cols + ["mean", "base_rate"]:
        if col in rate_summary_pct:
            rate_summary_pct[col] = rate_summary_pct[col] * 100.0

    bands = [
        ("p01", "p99", 0.12),
        ("p05", "p95", 0.18),
        ("p10", "p90", 0.24),
        ("p25", "p75", 0.30),
    ]

    figure = go.Figure()

    for lower, upper, opacity in bands:
        figure.add_trace(
            go.Scatter(
                x=months,
                y=rate_summary_pct[upper],
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
                name=f"{upper.upper()}",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=months,
                y=rate_summary_pct[lower],
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba(46, 134, 171, {opacity})",
                name=f"{lower.upper()} to {upper.upper()}",
                hovertemplate=(
                    "Month %{x}<br>"
                    f"{lower.upper()} %{y:.2f}%<extra>{lower.upper()}-{upper.upper()} band</extra>"
                ),
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scatter(
            x=months,
            y=rate_summary_pct["p50"],
            line=dict(color=palette["primary_navy"], width=3),
            name="Median",
            hovertemplate="Month %{x}<br>Median %{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=months,
            y=rate_summary_pct["mean"],
            line=dict(color=palette["secondary_teal"], width=2, dash="dot"),
            name="Mean",
            hovertemplate="Month %{x}<br>Mean %{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=months,
            y=rate_summary_pct["base_rate"],
            line=dict(color=palette["neutral"], width=2, dash="dash"),
            name="Base curve",
            hovertemplate="Month %{x}<br>Base %{y:.2f}%<extra></extra>",
        )
    )

    figure.update_layout(
        template=theme["plotly_template"],
        title="Confidence Bands Over Time",
        hovermode="x unified",
        margin=dict(l=60, r=30, t=60, b=60),
        xaxis=dict(title="Month", dtick=3),
        yaxis=dict(title="Rate (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )

    return figure
