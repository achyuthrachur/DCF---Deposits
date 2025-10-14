"""Reusable Plotly chart components for Monte Carlo dashboards."""

from .spaghetti_plot import build_spaghetti_figure
from .distribution_histogram import build_distribution_figure
from .fan_chart import build_fan_chart
from .percentile_ladder import build_percentile_ladder
from .risk_gauge import build_risk_gauge
from .convergence_plot import build_convergence_plot
from .correlation_heatmap import build_correlation_heatmap
from .yield_curve_animation import build_yield_curve_animation

__all__ = [
    "build_spaghetti_figure",
    "build_distribution_figure",
    "build_fan_chart",
    "build_percentile_ladder",
    "build_risk_gauge",
    "build_convergence_plot",
    "build_correlation_heatmap",
    "build_yield_curve_animation",
]
