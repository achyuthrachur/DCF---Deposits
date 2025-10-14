"""Live Monte Carlo simulation dashboard (real-time view)."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import List, Optional

import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

from .chart_components import (
    build_convergence_plot,
    build_distribution_figure,
    build_spaghetti_figure,
)
from .themes import DARK_THEME, DEFAULT_THEME, LIGHT_THEME
from .utils.data_reduction import summarise_distribution
from src.models.monte_carlo import MonteCarloProgressEvent


THEME_MAP = {
    "light": LIGHT_THEME,
    "dark": DARK_THEME,
}


@dataclass
class LiveSimulationFeed:
    """Thread-safe container for Monte Carlo progress events."""

    events: List[MonteCarloProgressEvent] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def append(self, event: MonteCarloProgressEvent) -> None:
        with self._lock:
            self.events.append(event)

    def snapshot(self) -> List[MonteCarloProgressEvent]:
        with self._lock:
            return list(self.events)


def _theme(name: Optional[str]) -> dict:
    if not name:
        return DEFAULT_THEME
    return THEME_MAP.get(name.lower(), DEFAULT_THEME)


def create_live_dashboard_app(
    feed: LiveSimulationFeed,
    *,
    theme: str = "light",
) -> Dash:
    theme_obj = _theme(theme)
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )

    app.layout = dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(html.H3("Monte Carlo Simulation In Progress", className="fw-bold")),
                    dbc.Col(
                        dbc.Select(
                            id="live-theme-select",
                            value=theme,
                            options=[
                                {"label": "Light", "value": "light"},
                                {"label": "Dark", "value": "dark"},
                            ],
                        ),
                        width=3,
                    ),
                ],
                align="center",
                className="mt-3",
            ),
            dcc.Interval(id="live-refresh", interval=1000, n_intervals=0),
            dcc.Store(id="live-data-store"),
            dbc.Row(
                [
                    dbc.Col(dcc.Graph(id="live-progress"), md=4),
                    dbc.Col(dcc.Graph(id="live-latest-paths"), md=4),
                    dbc.Col(dcc.Graph(id="live-histogram"), md=4),
                ],
                className="g-3 mt-1",
            ),
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            dbc.Row(
                                [
                                    dbc.Col(html.Div(id="stat-mean"), md=2),
                                    dbc.Col(html.Div(id="stat-median"), md=2),
                                    dbc.Col(html.Div(id="stat-p5"), md=2),
                                    dbc.Col(html.Div(id="stat-p95"), md=2),
                                    dbc.Col(html.Div(id="stat-std"), md=2),
                                    dbc.Col(html.Div(id="stat-var"), md=2),
                                ],
                                className="gy-2",
                            )
                        ),
                        className="shadow-sm",
                    )
                ),
                className="mt-3",
            ),
            dbc.Row(
                dbc.Col(dcc.Graph(id="live-convergence"), width=12),
                className="g-3 mt-3",
            ),
        ],
        fluid=True,
        className=f"theme-{theme_obj['name']} pb-4",
    )

    register_live_callbacks(app, feed)

    return app


def register_live_callbacks(app: Dash, feed: LiveSimulationFeed) -> None:
    @app.callback(
        Output("live-data-store", "data"),
        Input("live-refresh", "n_intervals"),
    )
    def poll_events(_):
        events = feed.snapshot()
        serialised = [
            {
                "simulation": e.simulation_index,
                "total": e.total_simulations,
                "pv": e.present_value,
                "mean": e.cumulative_mean,
                "std": e.cumulative_std,
                "percentiles": e.percentile_estimates,
                "rate_path": e.rate_path,
                "timestamp": e.timestamp,
            }
            for e in events
        ]
        return serialised

    @app.callback(
        Output("live-progress", "figure"),
        Output("live-latest-paths", "figure"),
        Output("live-histogram", "figure"),
        Output("live-convergence", "figure"),
        Output("stat-mean", "children"),
        Output("stat-median", "children"),
        Output("stat-p5", "children"),
        Output("stat-p95", "children"),
        Output("stat-std", "children"),
        Output("stat-var", "children"),
        Input("live-data-store", "data"),
        Input("live-theme-select", "value"),
    )
    def update_live_dashboard(serialised_events, theme_name):
        theme = _theme(theme_name)
        if not serialised_events:
            placeholder = go.Figure()
            placeholder.update_layout(
                template=theme["plotly_template"],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            empty_stats = ["—"] * 6
            return (
                placeholder,
                placeholder,
                placeholder,
                placeholder,
                *empty_stats,
            )

        df = pd.DataFrame(serialised_events)
        latest = df.iloc[-1]
        progress_fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=int(latest["simulation"]),
                gauge={
                    "axis": {"range": [0, int(latest["total"])]},
                    "bar": {"color": theme["palette"]["primary_navy"]},
                },
                number={"suffix": f"/{int(latest['total'])}"},
                title={"text": "Progress"},
            )
        )
        progress_fig.update_layout(
            template=theme["plotly_template"], margin=dict(l=20, r=20, t=50, b=20), height=250
        )

        # Latest paths (last 10)
        rate_paths = [row.get("rate_path") for row in serialised_events[-10:] if row.get("rate_path")]
        if rate_paths:
            rate_array = np.array(rate_paths)
            latest_paths_fig = build_spaghetti_figure(
                rate_array,
                pd.DataFrame(
                    {
                        "month": np.arange(1, rate_array.shape[1] + 1),
                        "mean": rate_array.mean(axis=0) / 100.0,
                        "p05": np.percentile(rate_array / 100.0, 5, axis=0),
                        "p95": np.percentile(rate_array / 100.0, 95, axis=0),
                        "p25": np.percentile(rate_array / 100.0, 25, axis=0),
                        "p75": np.percentile(rate_array / 100.0, 75, axis=0),
                        "p10": np.percentile(rate_array / 100.0, 10, axis=0),
                        "p90": np.percentile(rate_array / 100.0, 90, axis=0),
                        "p01": np.percentile(rate_array / 100.0, 1, axis=0),
                        "p99": np.percentile(rate_array / 100.0, 99, axis=0),
                        "base_rate": rate_array.mean(axis=0) / 100.0,
                        "p50": np.percentile(rate_array / 100.0, 50, axis=0),
                    }
                ),
                theme=theme,
                enable_animation=False,
            )
        else:
            latest_paths_fig = go.Figure()
            latest_paths_fig.update_layout(
                template=theme["plotly_template"],
                annotations=[
                    dict(
                        text="Waiting for first path…",
                        showarrow=False,
                        font=dict(color=theme["subtext_color"]),
                    )
                ],
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )

        pv_values = df["pv"].to_numpy(dtype=float)
        hist_df = pd.DataFrame({"simulation": df["simulation"], "present_value": pv_values})
        hist_fig = build_distribution_figure(
            hist_df,
            book_value=None,
            base_case=None,
            percentiles={},
            theme=theme,
            animate=False,
        )

        convergence_fig = build_convergence_plot(
            pd.DataFrame(
                {
                    "simulation": df["simulation"],
                    "cumulative_mean": df["mean"],
                    "p50": [p.get("p50", m) for p, m in zip(df["percentiles"], df["mean"])],
                }
            ),
            theme=theme,
        )

        mean_val, std_val, p5_val, p95_val = summarise_distribution(pv_values)
        median_val = np.median(pv_values)
        var_val = mean_val - p5_val

        stats = [
            html.Div([html.Small("Mean"), html.H5(f"${mean_val:,.0f}")]),
            html.Div([html.Small("Median"), html.H5(f"${median_val:,.0f}")]),
            html.Div([html.Small("P5"), html.H5(f"${p5_val:,.0f}")]),
            html.Div([html.Small("P95"), html.H5(f"${p95_val:,.0f}")]),
            html.Div([html.Small("Std Dev"), html.H5(f"${std_val:,.0f}")]),
            html.Div([html.Small("VaR"), html.H5(f"${var_val:,.0f}")]),
        ]

        return (
            progress_fig,
            latest_paths_fig,
            hist_fig,
            convergence_fig,
            *stats,
        )
