"""Visualization helpers for Monte Carlo ALM analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale


def extract_monte_carlo_data(
    results,
    scenario_id: str = "monte_carlo",
) -> Optional[Dict[str, object]]:
    """Prepare data required for Monte Carlo visualisations.

    Parameters
    ----------
    results:
        EngineResults instance returned by ``ALMEngine.run_analysis``.
    scenario_id:
        Identifier for the Monte Carlo scenario.
    """

    scenario_result = results.scenario_results.get(scenario_id)
    if scenario_result is None:
        return None

    tables = scenario_result.extra_tables
    if not tables:
        return None

    rate_sample = tables.get("rate_paths_sample")
    rate_summary = tables.get("rate_paths_summary")
    if rate_summary is None or rate_summary.empty:
        return None
    pv_distribution = tables.get("monte_carlo_distribution")
    if pv_distribution is None:
        pv_distribution = tables.get("simulation_pv")
    if pv_distribution is None:
        return None

    percentiles_table = tables.get("monte_carlo_percentiles")
    if percentiles_table is None:
        percentiles_table = tables.get("percentiles_table")

    base_case_pv = None
    if results.base_scenario_id and results.base_scenario_id in results.scenario_results:
        base_case_pv = results.scenario_results[results.base_scenario_id].present_value

    metadata = scenario_result.metadata or {}
    percentiles = metadata.get("pv_percentiles", {})
    if not percentiles and percentiles_table is not None:
        try:
            percentiles = {
                f"p{int(row.percentile)}": float(row.portfolio_pv)
                for row in percentiles_table.itertuples(index=False)
            }
        except Exception:  # pragma: no cover - fallback only
            percentiles = {}

    summary_stats = metadata.get("summary", {}) or {}
    if summary_stats:
        enriched = percentiles.copy()
        for key in ("mean", "median", "std", "skew", "kurtosis", "var_95", "cvar_95"):
            value = summary_stats.get(key)
            if value is not None:
                enriched[key] = value
        percentiles = enriched

    value_column = next(
        (
            candidate
            for candidate in ("portfolio_pv", "present_value", "pv", "value")
            if candidate in pv_distribution.columns
        ),
        None,
    )
    if value_column is not None:
        numeric_values = (
            pd.to_numeric(pv_distribution[value_column], errors="coerce")
            .dropna()
        )
        if not numeric_values.empty:
            percentiles = {
                **percentiles,
                "min": float(numeric_values.min()),
                "max": float(numeric_values.max()),
            }

    book_value = metadata.get("book_value")
    if book_value is None and results.validation_summary:
        book_value = results.validation_summary.get("portfolio_balance")

    return {
        "rate_sample": rate_sample,
        "rate_summary": rate_summary,
        "pv_distribution": pv_distribution,
        "percentiles": percentiles,
        "base_case_pv": base_case_pv,
        "book_value": book_value,
        "percentiles_table": percentiles_table,
        "summary": summary_stats,
    }


def _setup_figure(figsize=(10, 6)):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _format_currency_axis(ax, values: Iterable[float], axis: str = "x") -> None:
    """Format currency axes with scalable, compact tick labels."""
    finite_values = [
        abs(float(v))
        for v in values
        if v is not None and not np.isnan(v) and not np.isinf(v)
    ]
    if not finite_values:
        return

    max_value = max(finite_values)
    if max_value >= 1e9:
        scale, suffix, decimals = 1e9, "B", 1
    elif max_value >= 1e6:
        scale, suffix, decimals = 1e6, "M", 1
    elif max_value >= 1e5:
        scale, suffix, decimals = 1e3, "K", 0
    else:
        scale, suffix, decimals = 1.0, "", 0

    def _formatter(val: float, _pos: int) -> str:
        scaled = val / scale
        if decimals:
            formatted = f"{scaled:,.{decimals}f}"
        else:
            formatted = f"{scaled:,.0f}"
        return f"${formatted}{suffix}"

    axis_obj = getattr(ax, f"{axis}axis")
    axis_obj.set_major_formatter(FuncFormatter(_formatter))
    axis_obj.set_major_locator(MaxNLocator(6))
    ax.tick_params(axis=axis, labelsize=10)
    for spine in ("top", "right"):
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)


def _extract_short_rate_paths(
    rate_sample: Optional[pd.DataFrame], months: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return matrix of short-rate paths aligned to provided months."""
    n_months = len(months)
    if rate_sample is None or rate_sample.empty:
        return np.empty((0, n_months)), np.array([], dtype=int)

    # Preferred: long-form table with simulation/month columns.
    required_columns = {"simulation", "month", "short_rate"}
    if required_columns.issubset(rate_sample.columns):
        pivot = (
            rate_sample.pivot(index="simulation", columns="month", values="short_rate")
            .reindex(columns=months, fill_value=np.nan)
            .sort_index()
        )
        return pivot.to_numpy(dtype=float), pivot.index.to_numpy()

    # Fallback: assume one row per simulation with months already across columns.
    data = rate_sample.copy()
    if "simulation" in data.columns:
        labels = data["simulation"].to_numpy()
        data = data.drop(columns="simulation")
    else:
        labels = np.arange(1, len(data) + 1)

    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] < n_months:
        padding = np.full((arr.shape[0], n_months - arr.shape[1]), np.nan)
        arr = np.hstack([arr, padding])
    elif arr.shape[1] > n_months:
        arr = arr[:, :n_months]

    return arr, np.asarray(labels)


def plot_rate_path_spaghetti(
    rate_sample: Optional[pd.DataFrame],
    rate_summary: pd.DataFrame,
    *,
    title: str = "Monte Carlo Interest Rate Paths",
) -> plt.Figure:
    """Create a spaghetti plot of simulated rate paths."""

    if rate_summary is None:
        raise ValueError("Rate summary data is required")

    if "month" in rate_summary.columns:
        months = rate_summary["month"].to_numpy()
    else:
        months = np.arange(1, rate_summary.shape[0] + 1)
    fig, ax = _setup_figure(figsize=(12, 7))

    sample_paths, _ = _extract_short_rate_paths(rate_sample, months)
    for path in sample_paths:
        valid = ~np.isnan(path)
        if np.any(valid):
            ax.plot(
                months[valid],
                path[valid] * 100,
                color="steelblue",
                alpha=0.08,
                linewidth=0.6,
            )

    ax.plot(months, rate_summary["mean"].to_numpy() * 100, color="navy", linewidth=2.5, label="Mean")
    ax.plot(months, rate_summary["p05"].to_numpy() * 100, color="darkred", linestyle="--", linewidth=1.5, label="5th Percentile")
    ax.plot(months, rate_summary["p95"].to_numpy() * 100, color="darkred", linestyle="--", linewidth=1.5, label="95th Percentile")
    ax.fill_between(
        months,
        rate_summary["p05"].to_numpy() * 100,
        rate_summary["p95"].to_numpy() * 100,
        color="salmon",
        alpha=0.25,
        label="90% Confidence Band",
    )

    ax.plot(months, rate_summary["base_rate"].to_numpy() * 100, color="orange", linewidth=2, label="Base Case")

    ax.set_xlabel("Month")
    ax.set_ylabel("Interest Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_rate_confidence_fan(
    rate_summary: pd.DataFrame,
    *,
    title: str = "Interest Rate Confidence Bands",
) -> plt.Figure:
    """Create a fan chart of rate percentiles over time."""

    if rate_summary is None:
        raise ValueError("Rate summary data is required")

    months = rate_summary["month"].to_numpy()
    fig, ax = _setup_figure(figsize=(12, 7))

    bands = [
        ("p01", "p99", 0.1, "#cfe2f3", "1st-99th"),
        ("p05", "p95", 0.2, "#9fc5e8", "5th-95th"),
        ("p10", "p90", 0.3, "#6fa8dc", "10th-90th"),
        ("p25", "p75", 0.4, "#3d85c6", "25th-75th"),
    ]

    for lower, upper, alpha, color, label in bands:
        ax.fill_between(
            months,
            rate_summary[lower].to_numpy() * 100,
            rate_summary[upper].to_numpy() * 100,
            color=color,
            alpha=alpha,
            label=f"{label} percentile band",
        )

    ax.plot(months, rate_summary["p50"].to_numpy() * 100, color="navy", linewidth=2.5, label="Median")
    ax.plot(months, rate_summary["base_rate"].to_numpy() * 100, color="orange", linewidth=2, label="Base Case")

    ax.set_xlabel("Month")
    ax.set_ylabel("Interest Rate (%)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_portfolio_pv_distribution(
    pv_distribution: pd.DataFrame,
    *,
    book_value: Optional[float] = None,
    base_case_pv: Optional[float] = None,
    percentiles: Optional[Dict[str, float]] = None,
    title: str = "Portfolio PV Distribution",
) -> plt.Figure:
    """Plot histogram of portfolio PV outcomes."""

    if pv_distribution is None or pv_distribution.empty:
        raise ValueError("PV distribution data is required")

    value_column = next(
        (
            candidate
            for candidate in ("portfolio_pv", "present_value", "pv", "value")
            if candidate in pv_distribution.columns
        ),
        None,
    )
    if value_column is None:
        raise ValueError("PV distribution table missing portfolio PV column")

    values = pd.to_numeric(pv_distribution[value_column], errors="coerce").to_numpy()
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("PV distribution does not contain finite values")
    fig, ax = _setup_figure(figsize=(12, 6))
    counts, bins, patches = ax.hist(values, bins=60, color="#4f81bd", alpha=0.75, edgecolor="black")

    # Highlight risk zones
    if book_value is not None:
        for patch, left in zip(patches, bins[:-1]):
            if left + patch.get_width() < book_value * 0.95:
                patch.set_facecolor("#c0504d")
            elif left > book_value:
                patch.set_facecolor("#9bbb59")

    def _annotate_line(x_val, label, color):
        ax.axvline(x_val, color=color, linestyle="--", linewidth=2)
        ax.text(x_val, ax.get_ylim()[1] * 0.9, label, rotation=90, color=color, ha="right", va="top")

    if book_value is not None:
        _annotate_line(book_value, "Book Value", "black")
    if base_case_pv is not None:
        _annotate_line(base_case_pv, "Base Case PV", "orange")
    if percentiles:
        for key, color in [("p5", "red"), ("p50", "navy"), ("p95", "green")]:
            if key in percentiles:
                label = {
                    "p5": "5th percentile",
                    "p50": "Median",
                    "p95": "95th percentile",
                }.get(key, key.upper())
                _annotate_line(percentiles[key], label, color)

    ax.set_xlabel("Portfolio PV ($)")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y", linestyle="--")

    reference_values = list(values)
    if book_value is not None:
        reference_values.append(book_value)
    if base_case_pv is not None:
        reference_values.append(base_case_pv)
    if percentiles:
        reference_values.extend(percentiles.values())
    _format_currency_axis(ax, reference_values, axis="x")

    fig.tight_layout()
    return fig


def plot_percentile_ladder(
    percentiles: Dict[str, float],
    *,
    book_value: Optional[float] = None,
    base_case_pv: Optional[float] = None,
    title: str = "PV Percentile Ladder",
) -> plt.Figure:
    """Plot horizontal bar chart of PV percentiles."""

    if not percentiles:
        raise ValueError("Percentile metrics are required")

    order = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
    percentile_labels = {
        "p1": "1st percentile",
        "p5": "5th percentile",
        "p10": "10th percentile",
        "p25": "25th percentile",
        "p50": "Median (50th)",
        "p75": "75th percentile",
        "p90": "90th percentile",
        "p95": "95th percentile",
        "p99": "99th percentile",
    }
    labels = [percentile_labels[s] for s in order if s in percentiles]
    values = [percentiles[s] for s in order if s in percentiles]

    fig, ax = _setup_figure(figsize=(8, 6))
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, color="#4f81bd", alpha=0.75)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    if book_value is not None:
        ax.axvline(book_value, color="black", linestyle="--", linewidth=2, label="Book Value")
    if base_case_pv is not None:
        ax.axvline(base_case_pv, color="orange", linestyle=":", linewidth=2, label="Base Case")

    ax.set_xlabel("Portfolio PV ($)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    if (book_value is not None) or (base_case_pv is not None):
        ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2, axis="x", linestyle="--")

    reference_values = list(values)
    if book_value is not None:
        reference_values.append(book_value)
    if base_case_pv is not None:
        reference_values.append(base_case_pv)
    _format_currency_axis(ax, reference_values, axis="x")

    fig.tight_layout()
    return fig


def create_monte_carlo_dashboard(data: Dict[str, object]) -> plt.Figure:
    """Generate a comprehensive dashboard figure using Monte Carlo outputs."""

    rate_sample = data.get("rate_sample")
    rate_summary = data.get("rate_summary")
    pv_distribution = data.get("pv_distribution")
    percentiles = data.get("percentiles", {})
    base_case_pv = data.get("base_case_pv")
    book_value = data.get("book_value")

    if rate_summary is None or pv_distribution is None:
        raise ValueError("Monte Carlo data incomplete for dashboard")

    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(18, 11))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    # Top-left: Spaghetti plot
    ax1 = fig.add_subplot(gs[0, :2])
    if "month" in rate_summary.columns:
        months = rate_summary["month"].to_numpy()
    else:
        months = np.arange(1, rate_summary.shape[0] + 1)
    sample_paths, _ = _extract_short_rate_paths(rate_sample, months)
    for path in sample_paths:
        valid = ~np.isnan(path)
        if np.any(valid):
            ax1.plot(
                months[valid],
                path[valid] * 100,
                color="steelblue",
                alpha=0.05,
                linewidth=0.6,
            )
    ax1.plot(months, rate_summary["mean"].to_numpy() * 100, color="navy", linewidth=2.5, label="Mean path")
    ax1.fill_between(
        months,
        rate_summary["p05"].to_numpy() * 100,
        rate_summary["p95"].to_numpy() * 100,
        color="#f4cccc",
        alpha=0.35,
        label="90% confidence band",
    )
    ax1.plot(months, rate_summary["base_rate"].to_numpy() * 100, color="orange", linewidth=2, label="Deterministic base")
    ax1.set_title("Interest rate scenarios", fontweight="bold")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Rate (%)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Top-right: Risk gauge with explanation
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    if (book_value is not None) and percentiles:
        p5 = percentiles.get("p5")
        if p5 is not None:
            shortfall = book_value - p5
            shortfall_pct = max(0.0, shortfall / book_value * 100)
            color = "#4f9d69" if shortfall_pct <= 5 else "#f6c343" if shortfall_pct <= 15 else "#c94c4c"
            ax2.text(0.5, 0.7, f"${shortfall:,.0f}", fontsize=24, ha="center", fontweight="bold", color=color)
            ax2.text(0.5, 0.52, f"Shortfall vs. book value (5th percentile)", ha="center", fontsize=11)
            ax2.text(0.5, 0.35, f"Equivalent to {shortfall_pct:.1f}% of current balance", ha="center", fontsize=10)
    ax2.set_title("Risk insight", fontweight="bold")

    # Middle row: PV distribution
    ax3 = fig.add_subplot(gs[1, :2])
    value_column = next(
        (
            candidate
            for candidate in ("portfolio_pv", "present_value", "pv", "value")
            if candidate in pv_distribution.columns
        ),
        None,
    )
    if value_column is None:
        raise ValueError("PV distribution table missing portfolio PV column")
    values = (
        pd.to_numeric(pv_distribution[value_column], errors="coerce")
        .dropna()
        .to_numpy()
    )
    ax3.hist(values, bins=60, color="#4f81bd", alpha=0.75, edgecolor="black")
    if book_value is not None:
        ax3.axvline(book_value, color="black", linestyle="--", linewidth=2, label="Book value")
    if base_case_pv is not None:
        ax3.axvline(base_case_pv, color="orange", linestyle=":", linewidth=2, label="Deterministic base PV")
    if percentiles:
        for key, color, label in [
            ("p5", "red", "5th percentile"),
            ("p50", "navy", "Median"),
            ("p95", "green", "95th percentile"),
        ]:
            if key in percentiles:
                ax3.axvline(percentiles[key], color=color, linestyle="-", linewidth=1.8, label=label)
    ax3.set_title("Distribution of portfolio PV", fontweight="bold")
    ax3.set_xlabel("Present value ($)")
    ax3.set_ylabel("Number of simulations")
    reference_values = list(values)
    show_legend = False
    if book_value is not None:
        reference_values.append(book_value)
        show_legend = True
    if base_case_pv is not None:
        reference_values.append(base_case_pv)
        show_legend = True
    if percentiles:
        show_legend = True
    _format_currency_axis(ax3, reference_values, axis="x")
    if show_legend:
        ax3.legend(loc="best")
    ax3.grid(True, alpha=0.25, axis="y", linestyle="--")

    # Middle-right: Percentile table with descriptions
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    if percentiles:
        rows = [["Statistic", "Value ($)"]]
        for key, label in [
            ("mean", "Mean"),
            ("p50", "Median (50th percentile)"),
            ("p5", "5th percentile"),
            ("p95", "95th percentile"),
            ("min", "Minimum"),
            ("max", "Maximum"),
        ]:
            if key in percentiles:
                rows.append([label, f"{percentiles[key]:,.0f}"])
        table = ax4.table(cellText=rows, cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        for i in range(2):
            table[(0, i)].set_facecolor("#4f81bd")
            table[(0, i)].set_text_props(color="white", fontweight="bold")
    ax4.set_title("Key PV metrics", fontweight="bold")

    # Bottom-left: Percentile ladder
    ax5 = fig.add_subplot(gs[2, 0])
    if percentiles:
        order = ["p1", "p5", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
        percentile_labels = {
            "p1": "1st percentile",
            "p5": "5th percentile",
            "p10": "10th percentile",
            "p25": "25th percentile",
            "p50": "Median (50th)",
            "p75": "75th percentile",
            "p90": "90th percentile",
            "p95": "95th percentile",
            "p99": "99th percentile",
        }
        labels = [percentile_labels[s] for s in order if s in percentiles]
        values = [percentiles[s] for s in order if s in percentiles]
        y_pos = np.arange(len(values))
        bars = ax5.barh(
            y_pos,
            values,
            color="#6aa6d8",
            alpha=0.9,
            edgecolor="#f4f6fb",
            linewidth=0.8,
        )
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(labels)
        if book_value is not None:
            ax5.axvline(book_value, color="#1f1f1f", linestyle="--", linewidth=1.5, label="Book value", zorder=5)
        if base_case_pv is not None:
            ax5.axvline(base_case_pv, color="#f6a21a", linestyle=":", linewidth=1.5, label="Base case PV", zorder=5)
        ax5.set_xlabel("Present value ($)")
        ax5.set_title("Percentile ladder", fontweight="bold")
        reference_values = list(values)
        if book_value is not None:
            reference_values.append(book_value)
        if base_case_pv is not None:
            reference_values.append(base_case_pv)
        _format_currency_axis(ax5, reference_values, axis="x")
        max_value = max(reference_values) if reference_values else 0
        x_max = max_value * 1.08 if max_value else 1
        ax5.set_xlim(0, x_max)
        max_value = max(reference_values) if reference_values else 0
        label_offset = max_value * 0.008 if max_value else 0.02
        for rect, val in zip(bars, values):
            y_center = rect.get_y() + rect.get_height() / 2
            ax5.text(
                min(val + label_offset, x_max),
                y_center,
                f"${val:,.0f}",
                va="center",
                ha="left",
                fontsize=9,
                color="#2b2b2b",
                fontweight="bold",
            )
        if (book_value is not None) or (base_case_pv is not None):
            ax5.legend(loc="lower right", frameon=False)
        ax5.grid(True, alpha=0.2, axis="x", linestyle="--")
        for spine in ("top", "right"):
            ax5.spines[spine].set_visible(False)
        ax5.set_axisbelow(True)
    else:
        ax5.axis("off")

    # Bottom-middle: Scenario comparison (full values)
    ax6 = fig.add_subplot(gs[2, 1])
    if percentiles and base_case_pv:
        bars = [
            ("Base case", base_case_pv, "orange"),
            ("5th percentile", percentiles.get("p5", np.nan), "#c0504d"),
            ("Median", percentiles.get("p50", np.nan), "#4f81bd"),
            ("95th percentile", percentiles.get("p95", np.nan), "#9bbb59"),
        ]
        filtered = [(l, v, c) for l, v, c in bars if not np.isnan(v)]
        if filtered:
            labels, values, colors = zip(*filtered)
            positions = np.arange(len(labels))
            rects = ax6.bar(
                positions,
                values,
                color=colors,
                width=0.55,
                alpha=0.95,
                edgecolor="#f4f6fb",
                linewidth=1.0,
                zorder=3,
            )
            if book_value:
                ax6.axhline(
                    book_value,
                    color="#1f1f1f",
                    linestyle=(0, (6, 4)),
                    linewidth=1.6,
                    label="Book value",
                    zorder=5,
                )
            ax6.bar_label(
                rects,
                labels=[f"${val:,.0f}" for val in values],
                padding=4,
                fontsize=10,
                color="#1f1f1f",
                fontweight="bold",
                rotation=0,
                label_type="edge",
            )
            ax6.set_ylabel("Present value ($)")
            ax6.set_title("Scenarios vs. Monte Carlo percentiles", fontweight="bold")
            ax6.set_xticks(positions)
            ax6.set_xticklabels(labels, rotation=10)
            ax6.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
            candidate_values = list(values)
            if book_value is not None:
                candidate_values.append(book_value)
            max_reference = max(candidate_values)
            ax6.set_ylim(0, max_reference * 1.12 if max_reference else 1)
            ax6.grid(True, alpha=0.2, axis="y", linestyle="--")
            ax6.set_axisbelow(True)
            for spine in ("top", "right"):
                ax6.spines[spine].set_visible(False)
            ax6.spines["left"].set_alpha(0.6)
            ax6.legend(loc="upper left", frameon=False)
        else:
            ax6.axis("off")
    else:
        ax6.axis("off")

    # Bottom-right: Narrative summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    narrative_lines = []
    if percentiles:
        median = percentiles.get("p50")
        p5 = percentiles.get("p5")
        p95 = percentiles.get("p95")
        if (book_value is not None) and (median is not None):
            delta = median - book_value
            direction = "higher" if delta >= 0 else "lower"
            narrative_lines.append(
                f"Median PV: ${median:,.0f} ({abs(delta):,.0f} {direction} than book value)"
            )
        if (p5 is not None) and (p95 is not None):
            narrative_lines.append(
                f"Central 90% range: ${p5:,.0f} to ${p95:,.0f}"
            )
        if percentiles.get("std"):
            narrative_lines.append(
                f"Standard deviation across simulations: ${percentiles['std']:,.0f}"
            )
    if not narrative_lines:
        narrative_lines.append("Monte Carlo summary unavailable.")
    y = 0.9
    ax7.text(0.05, y, "Interpretation", fontsize=12, fontweight="bold")
    for line in narrative_lines:
        y -= 0.12
        ax7.text(0.05, y, line, fontsize=10)

    fig.suptitle("Monte Carlo ALM Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    return fig


def create_rate_path_animation(
    rate_sample: Optional[pd.DataFrame],
    rate_summary: pd.DataFrame,
) -> go.Figure:
    """Create an animated Plotly figure showing rate path evolution."""

    if rate_summary is None:
        raise ValueError("Rate summary data is required for animation")

    months = rate_summary["month"].to_numpy()
    mean_path = rate_summary["mean"].to_numpy() * 100
    base_path = rate_summary["base_rate"].to_numpy() * 100
    p5 = rate_summary["p05"].to_numpy() * 100
    p95 = rate_summary["p95"].to_numpy() * 100
    sample_paths, sample_labels = _extract_short_rate_paths(rate_sample, months)
    sample_paths = sample_paths * 100
    sample_colors: list[str] = []
    if sample_paths.size:
        max_paths = min(sample_paths.shape[0], 150)
        sample_paths = sample_paths[:max_paths]
        sample_labels = sample_labels[:max_paths]
        sort_key = np.nan_to_num(sample_paths[:, -1], nan=-np.inf)
        order = np.argsort(sort_key)
        sample_paths = sample_paths[order]
        sample_labels = sample_labels[order]
        color_positions = np.linspace(0.25, 0.85, sample_paths.shape[0]).tolist()
        sample_colors = list(sample_colorscale("Blues", color_positions))

    fig = go.Figure()

    for idx, path in enumerate(sample_paths):
        initial_series = np.concatenate(
            [path[:1], np.full(len(months) - 1, np.nan)]
        )
        fig.add_trace(
            go.Scatter(
                x=months,
                y=initial_series,
                mode="lines",
                line=dict(color=sample_colors[idx], width=1.3),
                opacity=0.7,
                legendgroup="sim_paths",
                name="Simulation paths" if idx == 0 else f"Simulation {int(sample_labels[idx])}",
                showlegend=idx == 0,
                hovertemplate="Month %{x}<br>Rate %{y:.2f}%<extra></extra>",
            )
        )

    fig.add_trace(
        go.Scatter(
            x=months[:1],
            y=mean_path[:1],
            mode="lines",
            name="Mean path",
            line=dict(color="#003f5c", width=3),
            hovertemplate="Month %{x}<br>Rate %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months[:1],
            y=base_path[:1],
            mode="lines",
            name="Base case",
            line=dict(color="#ffa600", width=2, dash="dash"),
            hovertemplate="Month %{x}<br>Rate %{y:.2f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([months[:1], months[:1][::-1]]),
            y=np.concatenate([p95[:1], p5[:1][::-1]]),
            fill="toself",
            fillcolor="rgba(244,204,204,0.35)",
            legendgroup="band",
            showlegend=True,
            name="5th-95th band",
            line=dict(color="rgba(244,204,204,0.0)"),
            hoverinfo="skip",
        )
    )

    frames: list[go.Frame] = []
    for i in range(1, len(months)):
        frame_traces: list[go.Scatter] = []
        if sample_paths.size:
            for idx, path in enumerate(sample_paths):
                path_values = np.concatenate(
                    [path[: i + 1], np.full(len(months) - i - 1, np.nan)]
                )
                frame_traces.append(
                    go.Scatter(
                        x=months,
                        y=path_values,
                        mode="lines",
                        line=dict(color=sample_colors[idx], width=1.3),
                        opacity=0.7,
                        legendgroup="sim_paths",
                        hovertemplate="Month %{x}<br>Rate %{y:.2f}%<extra></extra>",
                        showlegend=False,
                    )
                )
        frame_traces.append(
            go.Scatter(
                x=months[: i + 1],
                y=mean_path[: i + 1],
                mode="lines",
                line=dict(color="#003f5c", width=3),
            )
        )
        frame_traces.append(
            go.Scatter(
                x=months[: i + 1],
                y=base_path[: i + 1],
                mode="lines",
                line=dict(color="#ffa600", width=2, dash="dash"),
            )
        )
        frame_traces.append(
            go.Scatter(
                x=np.concatenate([months[: i + 1], months[: i + 1][::-1]]),
                y=np.concatenate([p95[: i + 1], p5[: i + 1][::-1]]),
                fill="toself",
                fillcolor="rgba(244,204,204,0.35)",
                line=dict(color="rgba(244,204,204,0.0)"),
                hoverinfo="skip",
            )
        )
        frames.append(go.Frame(data=frame_traces, name=str(months[i])))

    fig.frames = frames
    fig.update_layout(
        title="Evolution of interest rate paths",
        xaxis=dict(
            title="Month",
            dtick=12,
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Interest rate (%)",
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            zeroline=False,
        ),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(method="animate", args=[[str(m)]], label=f"Month {m}")
                    for m in months
                ],
                transition=dict(duration=0),
                x=0.05,
                len=0.9,
            )
        ],
        margin=dict(l=60, r=20, t=60, b=50),
    )

    return fig
def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Persist matplotlib figure to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path






