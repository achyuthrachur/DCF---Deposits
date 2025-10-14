"""Risk gauge component for VaR visualisation."""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go

from ..themes import DEFAULT_THEME


def build_risk_gauge(
    var_value: float,
    *,
    book_value: Optional[float] = None,
    theme: Optional[dict] = None,
) -> go.Figure:
    """
    Build a speedometer-style gauge showing VaR as a percentage of book value.
    """
    theme = theme or DEFAULT_THEME
    palette = theme["palette"]

    if book_value and book_value != 0:
        var_pct = abs(var_value) / book_value * 100.0
    else:
        var_pct = abs(var_value)

    gauge = go.Indicator(
        mode="gauge+number+delta",
        value=var_pct,
        number={"suffix": "%", "font": {"size": 28}},
        gauge={
            "axis": {"range": [0, max(15, var_pct * 1.2)], "tickwidth": 1, "tickcolor": "#BDBDBD"},
            "bar": {"color": palette["primary_navy"]},
            "bgcolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0, 5], "color": palette["positive"]},
                {"range": [5, 10], "color": palette["warning"]},
                {"range": [10, max(15, var_pct * 1.2)], "color": palette["negative"]},
            ],
        },
        delta={"reference": 5, "relative": False, "increasing": {"color": palette["negative"]}},
        title={"text": "Value at Risk", "font": {"size": 18}},
    )

    fig = go.Figure(gauge)
    fig.update_layout(
        template=theme["plotly_template"],
        margin=dict(l=20, r=20, t=50, b=20),
        height=260,
    )
    return fig
