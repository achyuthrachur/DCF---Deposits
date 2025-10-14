"""Dash-based Monte Carlo results dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from .chart_components import (
    build_convergence_plot,
    build_distribution_figure,
    build_fan_chart,
    build_percentile_ladder,
    build_risk_gauge,
    build_spaghetti_figure,
    build_yield_curve_animation,
)
from .dashboard_data import extract_monte_carlo_payload
from .themes import DARK_THEME, DEFAULT_THEME, LIGHT_THEME


THEME_MAP = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


@dataclass
class DashboardData:
    scenario_id: str
    pv_distribution: pd.DataFrame
    rate_summary: pd.DataFrame
    rate_sample: Optional[pd.DataFrame]
    progress: Optional[pd.DataFrame]
    percentiles: Dict[str, float]
    metadata: Dict[str, object]
    book_value: Optional[float]
    base_case_pv: Optional[float]

    @property
    def pv_values(self) -> np.ndarray:
        return self.pv_distribution["present_value"].to_numpy(dtype=float)

    @property
    def rate_paths(self) -> np.ndarray:
        if self.rate_sample is not None:
            cols = [col for col in self.rate_sample.columns if col.startswith("month_")]
            return self.rate_sample[cols].to_numpy(dtype=float)
        # Fallback to mean path if sample missing
        return np.tile(self.rate_summary["mean"].to_numpy(dtype=float), (10, 1))


def prepare_dashboard_data(
    results,
    scenario_id: str = "monte_carlo",
) -> Optional[DashboardData]:
    payload = extract_monte_carlo_payload(results, scenario_id=scenario_id)
    if payload is None:
        return None

    return DashboardData(
        scenario_id=payload["scenario_id"],
        pv_distribution=payload["pv_distribution"],
        rate_summary=payload["rate_summary"],
        rate_sample=payload["rate_sample"],
        progress=payload["progress"],
        percentiles=payload["percentiles"],
        metadata=payload["metadata"],
        book_value=payload["book_value"],
        base_case_pv=payload["base_case_pv"],
    )


def get_theme(name: str | None) -> dict:
    if not name:
        return DEFAULT_THEME
    return THEME_MAP.get(name.lower(), DEFAULT_THEME)


def _build_stat_cards(data: DashboardData, theme: dict) -> html.Div:
    palette = theme["palette"]
    percentiles = data.percentiles
    cards = [
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Mean", className="fw-bold"),
                    dbc.CardBody(
                        html.H4(f"${percentiles.get('mean', 0):,.0f}", className="card-title")
                    ),
                ],
                className="shadow-sm h-100",
                style={"borderLeft": f"4px solid {palette['primary_blue']}"},
            ),
            md=2,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Median (P50)", className="fw-bold"),
                    dbc.CardBody(
                        html.H4(f\"${percentiles.get('p50', 0):,.0f}\", className=\"card-title\")
                    ),
                ],
                className=\"shadow-sm h-100\",
                style={\"borderLeft\": f\"4px solid {palette['secondary_teal']}\"},
            ),
            md=2,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("P5 / P95", className="fw-bold"),
                    dbc.CardBody(
                        html.Div(
                            [
                                html.H4(f\"${percentiles.get('p5', 0):,.0f}\", className=\"card-title mb-1\"),  # noqa: E501
                                html.H6(f\"${percentiles.get('p95', 0):,.0f}\", className=\"text-muted\"),  # noqa: E501
                            ]
                        )
                    ),
                ],
                className=\"shadow-sm h-100\",
                style={\"borderLeft\": f\"4px solid {palette['warning']}\"},
            ),
            md=2,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Std Dev", className="fw-bold"),
                    dbc.CardBody(
                        html.H4(f\"${percentiles.get('std', 0):,.0f}\", className=\"card-title\")
                    ),
                ],
                className=\"shadow-sm h-100\",
                style={\"borderLeft\": f\"4px solid {palette['neutral']}\"},
            ),
            md=2,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Simulations", className="fw-bold"),
                    dbc.CardBody(
                        html.H4(
                            f\"{int(data.metadata.get('num_simulations', len(data.pv_distribution))):,}\",  # noqa: E501
                            className=\"card-title\",
                        )
                    ),
                ],
                className=\"shadow-sm h-100\",
                style={\"borderLeft\": f\"4px solid {palette['secondary_cyan']}\"},
            ),
            md=2,
        ),
        dbc.Col(
            dbc.Card(
                [
                    dbc.CardHeader("Drift / Volatility", className="fw-bold"),
                    dbc.CardBody(
                        html.Div(
                            [
                                html.H6(
                                    f\"Drift: {data.metadata.get('quarterly_drift', 0)*100:.2f} bps\",
                                    className=\"mb-1\",
                                ),
                                html.H6(
                                    f\"Vol: {data.metadata.get('quarterly_volatility', 0)*100:.2f} bps\",
                                    className=\"mb-0 text-muted\",
                                ),
                            ]
                        )
                    ),
                ],
                className=\"shadow-sm h-100\",
                style={\"borderLeft\": f\"4px solid {palette['primary_navy']}\"},
            ),
            md=2,
        ),
    ]
    return dbc.Row(cards, className="gy-3")


def build_dashboard_layout(
    data: DashboardData,
    *,
    theme_name: str = "light",
) -> html.Div:
    theme = get_theme(theme_name)
    risk_gauge = build_risk_gauge(
        var_value=data.percentiles.get("p5", 0) - data.percentiles.get("mean", 0),
        book_value=data.book_value,
        theme=theme,
    )
    distribution_fig = build_distribution_figure(
        data.pv_distribution,
        book_value=data.book_value,
        base_case=data.base_case_pv,
        percentiles=data.percentiles,
        theme=theme,
    )
    spaghetti_fig = build_spaghetti_figure(
        data.rate_paths,
        data.rate_summary,
        theme=theme,
    )
    fan_chart_fig = build_fan_chart(
        data.rate_summary,
        theme=theme,
    )
    percentile_ladder_fig = build_percentile_ladder(
        data.percentiles,
        book_value=data.book_value,
        base_case=data.base_case_pv,
        theme=theme,
    )
    convergence_fig = (
        build_convergence_plot(data.progress, theme=theme)
        if data.progress is not None
        else go_placeholder("Convergence data unavailable", theme)
    )

    header = dbc.Row(
        [
            dbc.Col(
                [
                    html.H2("Monte Carlo Results Dashboard", className="fw-bold"),
                    html.Div(
                        [
                            html.Span(f"Scenario: {data.scenario_id}", className="me-4"),
                            html.Span(
                                f"Simulations: {data.metadata.get('num_simulations', len(data.pv_distribution)):,}"  # noqa: E501
                            ),
                        ],
                        className="text-muted",
                    ),
                ],
                md=True,
            ),
            dbc.Col(
                dbc.Select(
                    id="theme-select",
                    options=[
                        {"label": "Light theme", "value": "light"},
                        {"label": "Dark theme", "value": "dark"},
                    ],
                    value=theme_name,
                    className="w-auto ms-auto",
                ),
                md=3,
            ),
        ],
        align="center",
        className="mb-4",
    )

    layout = dbc.Container(
        [
            header,
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="risk-gauge", figure=risk_gauge), md=4),
                    dbc.Col(dcc.Graph(id="pv-distribution", figure=distribution_fig), md=8),
                ],
                className="g-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="spaghetti-plot", figure=spaghetti_fig), width=12),
                ],
                className="g-3 mt-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="percentile-ladder", figure=percentile_ladder_fig), md=6),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(dcc.Graph(id="convergence-plot", figure=convergence_fig)),
                            className="h-100 shadow-sm",
                        ),
                        md=6,
                    ),
                ],
                className="g-3 mt-3",
            ),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="fan-chart", figure=fan_chart_fig), width=12),
                ],
                className="g-3 mt-3",
            ),
            html.H4("Key Statistics", className="mt-4"),
            _build_stat_cards(data, theme),
            dcc.Store(id="dashboard-theme", data=theme_name),
            dcc.Store(id="pv-range-filter"),
        ],
        fluid=True,
        className=f"theme-{theme['name']} pb-5",
    )
    return layout


def go_placeholder(message: str, theme: dict):
    """Return a minimal placeholder figure."""
    placeholder = go.Figure()
    placeholder.add_annotation(
        text=message,
        showarrow=False,
        font=dict(color=theme["subtext_color"]),
    )
    placeholder.update_layout(
        template=theme["plotly_template"],
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return placeholder


def register_callbacks(app: Dash, data: DashboardData) -> None:
    @app.callback(
        Output("spaghetti-plot", "figure"),
        Output("pv-distribution", "figure"),
        Output("percentile-ladder", "figure"),
        Output("fan-chart", "figure"),
        Output("convergence-plot", "figure"),
        Output("dashboard-theme", "data"),
        Input("theme-select", "value"),
        Input("pv-range-filter", "data"),
        prevent_initial_call=True,
    )
    def refresh_charts(theme_name: str, pv_range: Optional[Dict[str, float]]):
        theme = get_theme(theme_name)
        filtered_distribution = data.pv_distribution.copy()
        filtered_rate_paths = data.rate_paths.copy()

        if pv_range:
            lower = pv_range.get("min")
            upper = pv_range.get("max")
            mask = (filtered_distribution["present_value"] >= lower) & (
                filtered_distribution["present_value"] <= upper
            )
            sims = filtered_distribution.loc[mask, "simulation"]
            filtered_distribution = filtered_distribution.loc[mask]
            if data.rate_sample is not None:
                filtered_rate_paths = data.rate_sample[
                    data.rate_sample["simulation"].isin(sims)
                ]
                cols = [col for col in filtered_rate_paths.columns if col.startswith("month_")]
                if not filtered_rate_paths.empty:
                    filtered_rate_paths = filtered_rate_paths[cols].to_numpy(dtype=float)
                else:
                    filtered_rate_paths = data.rate_paths

        spaghetti_fig = build_spaghetti_figure(
            filtered_rate_paths,
            data.rate_summary,
            theme=theme,
        )
        distribution_fig = build_distribution_figure(
            filtered_distribution,
            book_value=data.book_value,
            base_case=data.base_case_pv,
            percentiles=data.percentiles,
            theme=theme,
        )
        percentile_fig = build_percentile_ladder(
            data.percentiles,
            book_value=data.book_value,
            base_case=data.base_case_pv,
            theme=theme,
        )
        fan_chart_fig = build_fan_chart(
            data.rate_summary,
            theme=theme,
        )
        convergence_fig = (
            build_convergence_plot(data.progress, theme=theme)
            if data.progress is not None
            else go_placeholder("Convergence data unavailable", theme)
        )

        return (
            spaghetti_fig,
            distribution_fig,
            percentile_fig,
            fan_chart_fig,
            convergence_fig,
            theme_name,
        )

    @app.callback(
        Output("pv-range-filter", "data"),
        Input("pv-distribution", "selectedData"),
        Input("pv-distribution", "clickData"),
        State("pv-range-filter", "data"),
        prevent_initial_call=True,
    )
    def update_filter(selected, clicked, current_state):
        if selected and "points" in selected and selected["points"]:
            xs = [point["x"] for point in selected["points"]]
            return {"min": min(xs), "max": max(xs)}
        if clicked and "points" in clicked and clicked["points"]:
            bin_x = clicked["points"][0]["x"]
            bin_width = clicked["points"][0].get("dx", 0)
            if bin_width:
                return {"min": bin_x - bin_width / 2, "max": bin_x + bin_width / 2}
            return {"min": bin_x, "max": bin_x}
        return current_state


def create_dashboard_app(
    results,
    scenario_id: str = "monte_carlo",
    *,
    theme: str = "light",
) -> Optional[Dash]:
    data = prepare_dashboard_data(results, scenario_id=scenario_id)
    if data is None:
        return None
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.layout = build_dashboard_layout(data, theme_name=theme)
    register_callbacks(app, data)
    return app
