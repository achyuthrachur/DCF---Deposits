"""Interactive rate-path spaghetti plot component."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..themes import DEFAULT_THEME
from ..utils.data_reduction import downsample_paths
from ..utils.animation_helpers import build_frame_names, build_time_axis


def build_spaghetti_figure(
    rate_paths: np.ndarray,
    rate_summary: pd.DataFrame,
    *,
    max_paths: int = 400,
    theme: Optional[dict] = None,
    enable_animation: bool = True,
) -> go.Figure:
    """
    Build the interactive spaghetti plot with confidence bands and animations.

    Parameters
    ----------
    rate_paths:
        2D numpy array with shape (num_simulations, months).
    rate_summary:
        DataFrame containing percentile columns (mean, p05, p95, etc) per month.
    max_paths:
        Maximum number of paths to render for performance.
    theme:
        Theme dictionary from ``visualization.themes``.
    enable_animation:
        Whether to include Plotly animation frames/controls.
    """
    theme = theme or DEFAULT_THEME
    palette = theme["palette"]
    months = rate_paths.shape[1]
    month_axis = build_time_axis(months)
    display_paths = downsample_paths(rate_paths, target_paths=max_paths)

    figure = go.Figure()

    path_color = palette["primary_blue"]

    for idx, path in enumerate(display_paths):
        figure.add_trace(
            go.Scatter(
                x=month_axis,
                y=path * 100.0,
                mode="lines",
                line=dict(color=path_color, width=1),
                opacity=0.18,
                name=f"Path {idx + 1}",
                hovertemplate="Month %{x}<br>Rate %{y:.2f}%<extra>%{fullData.name}</extra>",
                showlegend=False,
            )
        )

    # Confidence bands (95, 90, 75, 50)
    band_specs = [
        ("p01", "p99", 0.1),
        ("p05", "p95", 0.16),
        ("p10", "p90", 0.22),
        ("p25", "p75", 0.28),
    ]

    for lower, upper, opacity in band_specs:
        figure.add_trace(
            go.Scatter(
                x=month_axis,
                y=rate_summary[upper] * 100.0,
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=month_axis,
                y=rate_summary[lower] * 100.0,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba(31, 71, 136, {opacity})",
                name=f"{upper.upper()}-{lower.upper()} band",
                hovertemplate=(
                    "Month %{x}<br>"
                    f"{lower.upper()} %{y:.2f}%<extra>{upper.upper()}-{lower.upper()} band</extra>"
                ),
                showlegend=False,
            )
        )

    # Mean and percentile lines
    figure.add_trace(
        go.Scatter(
            x=month_axis,
            y=rate_summary["mean"] * 100.0,
            mode="lines",
            line=dict(color=palette["primary_navy"], width=3),
            name="Mean",
            hovertemplate="Month %{x}<br>Mean %{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=month_axis,
            y=rate_summary["p50"] * 100.0,
            mode="lines",
            line=dict(color=palette["secondary_teal"], width=2),
            name="Median",
            hovertemplate="Month %{x}<br>Median %{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=month_axis,
            y=rate_summary["p05"] * 100.0,
            mode="lines",
            line=dict(color=palette["negative"], width=2, dash="dash"),
            name="P5",
            hovertemplate="Month %{x}<br>P5 %{y:.2f}%<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=month_axis,
            y=rate_summary["p95"] * 100.0,
            mode="lines",
            line=dict(color=palette["warning"], width=2, dash="dash"),
            name="P95",
            hovertemplate="Month %{x}<br>P95 %{y:.2f}%<extra></extra>",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=month_axis,
            y=rate_summary["base_rate"] * 100.0,
            mode="lines",
            line=dict(color=palette["neutral"], width=2, dash="dot"),
            name="Base curve",
            hovertemplate="Month %{x}<br>Base %{y:.2f}%<extra></extra>",
        )
    )

    figure.update_layout(
        template=theme["plotly_template"],
        hovermode="x unified",
        title="Monte Carlo Rate Path Spaghetti",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(l=60, r=20, t=60, b=60),
        xaxis=dict(
            title="Month",
            dtick=3,
            tick0=0,
            hoverformat=".0f",
            rangeslider=dict(visible=False),
        ),
        yaxis=dict(
            title="Rate (%)",
            hoverformat=".2f",
        ),
    )

    if enable_animation:
        frames = []
        frame_names = build_frame_names(months)
        for frame_idx, name in enumerate(frame_names):
            frames.append(
                go.Frame(
                    name=name,
                    data=[
                        go.Scatter(x=month_axis[: frame_idx + 1], y=trace.y[: frame_idx + 1])
                        if isinstance(trace, go.Scatter) and trace.mode == "lines"
                        else trace
                        for trace in figure.data
                    ],
                )
            )
        figure.update(frames=frames)
        figure.update_layout(
            updatemenus=[
                {
                    "type": "buttons",
                    "showactive": False,
                    "buttons": [
                        {
                            "label": "Play",
                            "method": "animate",
                            "args": [
                                None,
                                {
                                    "frame": {"duration": 120, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 200},
                                },
                            ],
                        },
                        {
                            "label": "Pause",
                            "method": "animate",
                            "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        },
                    ],
                }
            ],
            sliders=[
                {
                    "steps": [
                        {
                            "args": [[name], {"frame": {"duration": 0, "redraw": True}}],
                            "label": str(idx + 1),
                            "method": "animate",
                        }
                        for idx, name in enumerate(frame_names)
                    ],
                    "currentvalue": {"prefix": "Month: "},
                    "pad": {"t": 30},
                }
            ],
        )

    return figure
