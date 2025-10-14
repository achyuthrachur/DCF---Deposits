"""Animated Monte Carlo PV distribution histogram."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..themes import DEFAULT_THEME
from ..utils.animation_helpers import build_frame_names


def _zone_color(value: float, book_value: Optional[float]) -> str:
    if book_value is None:
        return DEFAULT_THEME["palette"]["primary_blue"]
    if value >= book_value:
        return DEFAULT_THEME["palette"]["positive"]
    if value >= 0.9 * book_value:
        return DEFAULT_THEME["palette"]["warning"]
    return DEFAULT_THEME["palette"]["negative"]


def build_distribution_figure(
    pv_distribution: pd.DataFrame,
    *,
    book_value: Optional[float] = None,
    base_case: Optional[float] = None,
    percentiles: Optional[Dict[str, float]] = None,
    theme: Optional[dict] = None,
    animate: bool = True,
) -> go.Figure:
    """
    Create the animated PV distribution histogram with dynamic overlays.
    """
    theme = theme or DEFAULT_THEME
    palette = theme["palette"]
    values = pv_distribution["present_value"].to_numpy(dtype=float)
    simulations = pv_distribution["simulation"].to_numpy(dtype=int)

    nbins = min(60, max(20, int(np.sqrt(values.size))))
    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Histogram(
            x=values,
            nbinsx=nbins,
            opacity=0.85,
            marker=dict(
                color=[_zone_color(v, book_value) for v in values],
                line=dict(color="#424242", width=0.5),
            ),
            hovertemplate=(
                "PV %{x:,.0f}<br>Count %{y}<extra>Simulation Distribution</extra>"
            ),
            name="PV Distribution",
        )
    )

    annotations = []
    shapes = []

    if book_value is not None:
        shapes.append(
            dict(
                type="line",
                x0=book_value,
                x1=book_value,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=palette["secondary_cyan"], width=3, dash="dot"),
            )
        )
        annotations.append(
            dict(
                x=book_value,
                y=1.02,
                xref="x",
                yref="paper",
                text="Book value",
                showarrow=False,
                font=dict(color=palette["secondary_cyan"]),
            )
        )

    if base_case is not None:
        shapes.append(
            dict(
                type="line",
                x0=base_case,
                x1=base_case,
                y0=0,
                y1=1,
                xref="x",
                yref="paper",
                line=dict(color=palette["primary_navy"], width=2, dash="dash"),
            )
        )
        annotations.append(
            dict(
                x=base_case,
                y=1.05,
                xref="x",
                yref="paper",
                text="Base case PV",
                showarrow=False,
                font=dict(color=palette["primary_navy"]),
            )
        )

    if percentiles:
        for label, value in percentiles.items():
            label_clean = label.upper()
            shapes.append(
                dict(
                    type="line",
                    x0=value,
                    x1=value,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color=palette["neutral"], width=1, dash="dot"),
                )
            )
            annotations.append(
                dict(
                    x=value,
                    y=1.08,
                    xref="x",
                    yref="paper",
                    text=label_clean,
                    showarrow=False,
                    font=dict(color=palette["neutral"], size=10),
                )
            )

    hist_fig.update_layout(
        template=theme["plotly_template"],
        title="Present Value Distribution",
        bargap=0.02,
        margin=dict(l=60, r=30, t=60, b=60),
        xaxis=dict(title="Present Value ($)", tickformat="~s"),
        yaxis=dict(title="Simulations"),
        shapes=shapes,
        annotations=annotations,
    )

    if animate:
        frames = []
        frame_names = build_frame_names(values.size)
        cumulative_values = []
        for idx, value in enumerate(values):
            cumulative_values.append(value)
            frames.append(
                go.Frame(
                    name=frame_names[idx],
                    data=[
                        go.Histogram(
                            x=cumulative_values.copy(),
                            nbinsx=nbins,
                            marker=dict(
                                color=[
                                    _zone_color(v, book_value) for v in cumulative_values
                                ]
                            ),
                        )
                    ],
                )
            )
        hist_fig.update(
            frames=frames,
            layout=dict(
                updatemenus=[
                    {
                        "type": "buttons",
                        "buttons": [
                            {
                                "label": "Replay Build-up",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "fromcurrent": True,
                                        "frame": {"duration": 80, "redraw": True},
                                    },
                                ],
                            },
                            {
                                "label": "Pause",
                                "method": "animate",
                                "args": [[None], {"frame": {"duration": 0}}],
                            },
                        ],
                    }
                ],
                sliders=[
                    {
                        "steps": [
                            {
                                "args": [[name], {"frame": {"duration": 0}}],
                                "label": str(simulations[idx]),
                                "method": "animate",
                            }
                            for idx, name in enumerate(frame_names)
                        ],
                        "currentvalue": {"prefix": "Simulations: "},
                    }
                ],
            ),
        )

    return hist_fig
