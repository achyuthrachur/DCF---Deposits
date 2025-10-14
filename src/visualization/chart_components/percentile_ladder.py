"""Interactive percentile ladder chart."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import plotly.graph_objects as go

from ..themes import DEFAULT_THEME


def _sorted_percentiles(percentiles: Dict[str, float]) -> list[tuple[str, float]]:
    def key(item: tuple[str, float]) -> float:
        label, _ = item
        try:
            return float(label.strip("pP"))
        except ValueError:
            return 0.0

    return sorted(percentiles.items(), key=key)


def build_percentile_ladder(
    percentiles: Dict[str, float],
    *,
    book_value: Optional[float] = None,
    base_case: Optional[float] = None,
    show_percent_of_book: bool = False,
    theme: Optional[dict] = None,
) -> go.Figure:
    """
    Build a horizontal percentile ladder chart.
    """
    theme = theme or DEFAULT_THEME
    palette = theme["palette"]

    sorted_percentiles = _sorted_percentiles(percentiles)
    labels = [label.upper() for label, _ in sorted_percentiles]
    values = np.array([value for _, value in sorted_percentiles], dtype=float)

    if show_percent_of_book and book_value:
        plot_values = (values / book_value) * 100.0
        xaxis_title = "% of book value"
        hover_suffix = "%"
    else:
        plot_values = values
        xaxis_title = "Present Value ($)"
        hover_suffix = ""

    bars = go.Bar(
        x=plot_values,
        y=labels,
        orientation="h",
        marker=dict(color=palette["primary_blue"]),
        hovertemplate="Percentile %{y}<br>PV %{x:,.2f}" + hover_suffix + "<extra></extra>",
        name="Percentiles",
    )

    figures = [bars]
    shapes = []
    annotations = []

    def _ref_line(value: Optional[float], text: str, color: str, offset: float) -> None:
        if value is None:
            return
        line_value = (value / book_value) * 100.0 if (show_percent_of_book and book_value) else value
        shapes.append(
            dict(
                type="line",
                x0=line_value,
                x1=line_value,
                y0=-0.5,
                y1=len(labels) - 0.5,
                line=dict(color=color, width=2, dash="dash"),
            )
        )
        annotations.append(
            dict(
                x=line_value,
                y=len(labels) + offset,
                text=text,
                showarrow=False,
                font=dict(color=color, size=12),
            )
        )

    _ref_line(book_value, "Book value", palette["secondary_cyan"], 0.2)
    _ref_line(base_case, "Base case PV", palette["primary_navy"], 0.4)

    layout = dict(
        template=theme["plotly_template"],
        title="Percentile Ladder",
        margin=dict(l=120, r=40, t=60, b=40),
        xaxis=dict(title=xaxis_title, tickformat=".2f"),
        yaxis=dict(autorange="reversed"),
        shapes=shapes,
        annotations=annotations,
    )

    fig = go.Figure(data=figures, layout=layout)
    return fig
