"""Visualization helpers for Monte Carlo ALM analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import plotly.graph_objects as go


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
    pv_distribution = tables.get("simulation_pv")
    if pv_distribution is None:
        return None

    book_value = None
    if results.validation_summary:
        book_value = results.validation_summary.get("portfolio_balance")

    base_case_pv = None
    if results.base_scenario_id and results.base_scenario_id in results.scenario_results:
        base_case_pv = results.scenario_results[results.base_scenario_id].present_value

    percentiles = scenario_result.metadata.get("pv_percentiles", {})

    return {
        "rate_sample": rate_sample,
        "rate_summary": rate_summary,
        "pv_distribution": pv_distribution,
        "percentiles": percentiles,
        "base_case_pv": base_case_pv,
        "book_value": book_value,
    }


def _setup_figure(figsize=(10, 6)):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def plot_rate_path_spaghetti(
    rate_sample: pd.DataFrame,
    rate_summary: pd.DataFrame,
    *,
    title: str = "Monte Carlo Interest Rate Paths",
) -> plt.Figure:
    """Create a spaghetti plot of simulated rate paths."""

    if rate_summary is None:
        raise ValueError("Rate summary data is required")

    months = np.arange(1, rate_summary.shape[0] + 1)
    fig, ax = _setup_figure(figsize=(12, 7))

    # Convert sample to numpy for plotting
    if rate_sample is not None:
        sample_paths = rate_sample.drop(columns="simulation").to_numpy()
        for path in sample_paths:
            ax.plot(months, path * 100, color="steelblue", alpha=0.08, linewidth=0.6)

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

    values = pv_distribution["present_value"].to_numpy()
    fig, ax = _setup_figure(figsize=(12, 6))
    counts, bins, patches = ax.hist(values, bins=60, color="#4f81bd", alpha=0.75, edgecolor="black")

    # Highlight risk zones
    if book_value:
        for patch, left in zip(patches, bins[:-1]):
            if left + patch.get_width() < book_value * 0.95:
                patch.set_facecolor("#c0504d")
            elif left > book_value:
                patch.set_facecolor("#9bbb59")

    def _annotate_line(x_val, label, color):
        ax.axvline(x_val, color=color, linestyle="--", linewidth=2)
        ax.text(x_val, ax.get_ylim()[1] * 0.9, label, rotation=90, color=color, ha="right", va="top")

    if book_value:
        _annotate_line(book_value, "Book Value", "black")
    if base_case_pv:
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
    ax.grid(True, alpha=0.3, axis="y")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
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
    ax.barh(y_pos, values, color="#4f81bd", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    if book_value:
        ax.axvline(book_value, color="black", linestyle="--", linewidth=2, label="Book Value")
    if base_case_pv:
        ax.axvline(base_case_pv, color="orange", linestyle=":", linewidth=2, label="Base Case")

    ax.set_xlabel("Portfolio PV ($)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    if book_value or base_case_pv:
        ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
    ax.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
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
    months = np.arange(1, rate_summary.shape[0] + 1)
    if rate_sample is not None:
        for path in rate_sample.drop(columns="simulation").to_numpy():
            ax1.plot(months, path * 100, color="steelblue", alpha=0.05, linewidth=0.6)
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
    if book_value and percentiles:
        p5 = percentiles.get("p5")
        if p5:
            shortfall = book_value - p5
            shortfall_pct = max(0.0, shortfall / book_value * 100)
            color = "#4f9d69" if shortfall_pct <= 5 else "#f6c343" if shortfall_pct <= 15 else "#c94c4c"
            ax2.text(0.5, 0.7, f"${shortfall:,.0f}", fontsize=24, ha="center", fontweight="bold", color=color)
            ax2.text(0.5, 0.52, f"Shortfall vs. book value (5th percentile)", ha="center", fontsize=11)
            ax2.text(0.5, 0.35, f"Equivalent to {shortfall_pct:.1f}% of current balance", ha="center", fontsize=10)
    ax2.set_title("Risk insight", fontweight="bold")

    # Middle row: PV distribution
    ax3 = fig.add_subplot(gs[1, :2])
    values = pv_distribution["present_value"].to_numpy()
    ax3.hist(values, bins=60, color="#4f81bd", alpha=0.75, edgecolor="black")
    if book_value:
        ax3.axvline(book_value, color="black", linestyle="--", linewidth=2, label="Book value")
    if base_case_pv:
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
    ax3.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.25, axis="y")

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
        ax5.barh(y_pos, values, color="#4f81bd", alpha=0.75)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(labels)
        if book_value:
            ax5.axvline(book_value, color="black", linestyle="--", linewidth=2, label="Book value")
        if base_case_pv:
            ax5.axvline(base_case_pv, color="orange", linestyle=":", linewidth=2, label="Base case PV")
        ax5.set_xlabel("Present value ($)")
        ax5.set_title("Percentile ladder", fontweight="bold")
        ax5.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
        if book_value or base_case_pv:
            ax5.legend(loc="lower right")
        ax5.grid(True, alpha=0.3, axis="x")
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
            rects = ax6.bar(labels, values, color=colors, alpha=0.85, edgecolor="black")
            if book_value:
                ax6.axhline(book_value, color="black", linestyle="--", linewidth=2, label="Book value")
            for rect, val in zip(rects, values):
                ax6.text(rect.get_x() + rect.get_width() / 2, val, f"${val:,.0f}", ha="center", va="bottom")
            ax6.set_ylabel("Present value ($)")
            ax6.set_title("Scenarios vs. Monte Carlo percentiles", fontweight="bold")
            ax6.set_xticklabels(labels, rotation=15)
            ax6.yaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("${x:,.0f}"))
            ax6.grid(True, alpha=0.3, axis="y")
            ax6.legend(loc="best")
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
        if book_value and median:
            delta = median - book_value
            direction = "higher" if delta >= 0 else "lower"
            narrative_lines.append(
                f"Median PV: ${median:,.0f} ({abs(delta):,.0f} {direction} than book value)"
            )
        if p5 and p95:
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

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=months[:1],
            y=mean_path[:1],
            mode="lines",
            name="Mean path",
            line=dict(color="#003f5c", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=months[:1],
            y=base_path[:1],
            mode="lines",
            name="Base case",
            line=dict(color="#ffa600", width=2, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([months[:1], months[:1][::-1]]),
            y=np.concatenate([p95[:1], p5[:1][::-1]]),
            fill="toself",
            fillcolor="rgba(244,204,204,0.4)",
            line=dict(color="rgba(244,204,204,0.0)"),
            hoverinfo="skip",
            name="5th-95th band",
        )
    )

    frames = []
    for i in range(1, len(months)):
        frame_data = [
            go.Scatter(x=months[: i + 1], y=mean_path[: i + 1], mode="lines", line=dict(color="#003f5c", width=3)),
            go.Scatter(x=months[: i + 1], y=base_path[: i + 1], mode="lines", line=dict(color="#ffa600", width=2, dash="dash")),
            go.Scatter(
                x=np.concatenate([months[: i + 1], months[: i + 1][::-1]]),
                y=np.concatenate([p95[: i + 1], p5[: i + 1][::-1]]),
                fill="toself",
                fillcolor="rgba(244,204,204,0.4)",
                line=dict(color="rgba(244,204,204,0.0)"),
                hoverinfo="skip",
            ),
        ]
        frames.append(go.Frame(data=frame_data, name=str(months[i])))

    fig.frames = frames
    fig.update_layout(
        title="Evolution of interest rate paths",
        xaxis_title="Month",
        yaxis_title="Interest rate (%)",
        template="plotly_white",
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True}]),
                    dict(label="Pause", method="animate", args=[[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}]),
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
    )

    return fig
def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Persist matplotlib figure to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path
