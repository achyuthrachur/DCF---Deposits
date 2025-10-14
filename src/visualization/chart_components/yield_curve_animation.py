"""Animated yield-curve evolution component."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..themes import DEFAULT_THEME


def build_yield_curve_animation(
    curve_records: pd.DataFrame,
    *,
    selected_simulations: Optional[Sequence[int]] = None,
    theme: Optional[dict] = None,
) -> go.Figure:
    """
    Animate the evolution of the yield curve for selected simulations.

    Parameters
    ----------
    curve_records:
        DataFrame with columns [simulation, month, tenor, rate].
    selected_simulations:
        Iterable of simulation identifiers to display. Defaults to first five.
    """
    theme = theme or DEFAULT_THEME
    palette = theme["palette"]

    if curve_records.empty:
        return go.Figure()

    available_sims = sorted(curve_records["simulation"].unique())
    if not selected_simulations:
        selected_simulations = available_sims[:5]
    else:
        selected_simulations = [sim for sim in selected_simulations if sim in available_sims]
        if not selected_simulations:
            selected_simulations = available_sims[:5]

    base_fig = go.Figure()
    color_cycle = [
        palette["primary_blue"],
        palette["primary_navy"],
        palette["secondary_teal"],
        palette["secondary_cyan"],
        palette["warning"],
    ]

    frames = []
    months = sorted(curve_records["month"].unique())
    tenors = sorted(curve_records["tenor"].unique())

    for sim_idx, simulation in enumerate(selected_simulations):
        start_slice = curve_records[(curve_records["simulation"] == simulation) & (curve_records["month"] == months[0])]
        base_fig.add_trace(
            go.Scatter(
                x=start_slice["tenor"],
                y=start_slice["rate"] * 100.0,
                mode="lines+markers",
                line=dict(color=color_cycle[sim_idx % len(color_cycle)], width=2),
                name=f"Simulation {simulation}",
                hovertemplate="Tenor %{x}M<br>Rate %{y:.2f}%<extra>Sim {simulation}</extra>",
            )
        )

    for month in months:
        frame_data = []
        for sim_idx, simulation in enumerate(selected_simulations):
            slice_df = curve_records[
                (curve_records["simulation"] == simulation) & (curve_records["month"] == month)
            ].sort_values("tenor")
            frame_data.append(
                go.Scatter(
                    x=slice_df["tenor"],
                    y=slice_df["rate"] * 100.0,
                    mode="lines+markers",
                    line=dict(color=color_cycle[sim_idx % len(color_cycle)], width=2),
                    name=f"Simulation {simulation}",
                    hovertemplate="Tenor %{x}M<br>Rate %{y:.2f}%<extra>Sim {simulation}</extra>",
                )
            )
        frames.append(go.Frame(data=frame_data, name=f"month_{month:03d}"))

    base_fig.update(frames=frames)
    base_fig.update_layout(
        template=theme["plotly_template"],
        title="Yield Curve Evolution",
        xaxis=dict(title="Tenor (months)"),
        yaxis=dict(title="Rate (%)"),
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 180, "redraw": True},
                                "transition": {"duration": 200},
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
                        "args": [[f"month_{month:03d}"], {"frame": {"duration": 0}}],
                        "label": f"Month {month}",
                        "method": "animate",
                    }
                    for month in months
                ]
            }
        ],
        margin=dict(l=60, r=30, t=60, b=60),
    )

    return base_fig
