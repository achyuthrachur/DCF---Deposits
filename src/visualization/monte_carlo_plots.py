"""Visualization helpers for Monte Carlo ALM analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


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

    if rate_sample is None or rate_summary is None:
        raise ValueError("Rate sample and summary data are required")

    months = np.arange(1, rate_summary.shape[0] + 1)
    fig, ax = _setup_figure(figsize=(12, 7))

    # Convert sample to numpy for plotting
    sample_paths = rate_sample.drop(columns="simulation").to_numpy()
    for path in sample_paths:
        ax.plot(months, path * 100, color="steelblue", alpha=0.08, linewidth=0.6)

    ax.plot(months, rate_summary["mean"].to_numpy() * 100, color="navy", linewidth=2.5, label="Mean")
    ax.plot(months, rate_summary["p05"].to_numpy() * 100, color="darkred", linestyle="--", linewidth=1.5, label="P5 / P95")
    ax.plot(months, rate_summary["p95"].to_numpy() * 100, color="darkred", linestyle="--", linewidth=1.5)
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
        ("p01", "p99", 0.1, "#cfe2f3"),
        ("p05", "p95", 0.2, "#9fc5e8"),
        ("p10", "p90", 0.3, "#6fa8dc"),
        ("p25", "p75", 0.4, "#3d85c6"),
    ]

    for lower, upper, alpha, color in bands:
        ax.fill_between(
            months,
            rate_summary[lower].to_numpy() * 100,
            rate_summary[upper].to_numpy() * 100,
            color=color,
            alpha=alpha,
            label=f"{lower.upper()} - {upper.upper()}",
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

    values = pv_distribution["present_value"].to_numpy() / 1e6
    fig, ax = _setup_figure(figsize=(12, 6))
    counts, bins, patches = ax.hist(values, bins=60, color="#4f81bd", alpha=0.75, edgecolor="black")

    # Highlight risk zones
    if book_value:
        book_millions = book_value / 1e6
        for patch, left in zip(patches, bins[:-1]):
            if left + patch.get_width() < book_millions * 0.95:
                patch.set_facecolor("#c0504d")
            elif left > book_millions:
                patch.set_facecolor("#9bbb59")

    def _annotate_line(x_val, label, color):
        ax.axvline(x_val, color=color, linestyle="--", linewidth=2)
        ax.text(x_val, ax.get_ylim()[1] * 0.9, label, rotation=90, color=color, ha="right", va="top")

    if book_value:
        _annotate_line(book_value / 1e6, "Book Value", "black")
    if base_case_pv:
        _annotate_line(base_case_pv / 1e6, "Base Case PV", "orange")
    if percentiles:
        for key, color in [("p5", "red"), ("p50", "navy"), ("p95", "green")]:
            if key in percentiles:
                _annotate_line(percentiles[key] / 1e6, key.upper(), color)

    ax.set_xlabel("Portfolio PV ($ millions)")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
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
    labels = [s.upper() for s in order if s in percentiles]
    values = [percentiles[s] / 1e6 for s in order if s in percentiles]

    fig, ax = _setup_figure(figsize=(8, 6))
    y_pos = np.arange(len(values))
    ax.barh(y_pos, values, color="#4f81bd", alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)

    if book_value:
        ax.axvline(book_value / 1e6, color="black", linestyle="--", linewidth=2, label="Book Value")
    if base_case_pv:
        ax.axvline(base_case_pv / 1e6, color="orange", linestyle=":", linewidth=2, label="Base Case")

    ax.set_xlabel("Portfolio PV ($ millions)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    if book_value or base_case_pv:
        ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="x")
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
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Top-left: Spaghetti plot
    ax1 = fig.add_subplot(gs[0, :2])
    months = np.arange(1, rate_summary.shape[0] + 1)
    if rate_sample is not None:
        for path in rate_sample.drop(columns="simulation").to_numpy():
            ax1.plot(months, path * 100, color="steelblue", alpha=0.06, linewidth=0.6)
    ax1.plot(months, rate_summary["mean"].to_numpy() * 100, color="navy", linewidth=2.5, label="Mean")
    ax1.fill_between(
        months,
        rate_summary["p05"].to_numpy() * 100,
        rate_summary["p95"].to_numpy() * 100,
        color="salmon",
        alpha=0.25,
        label="90% Confidence",
    )
    ax1.plot(months, rate_summary["base_rate"].to_numpy() * 100, color="orange", linewidth=2, label="Base Case")
    ax1.set_title("Interest Rate Paths", fontweight="bold")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Rate (%)")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    # Top-right: Risk gauge
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    if book_value and percentiles:
        p5 = percentiles.get("p5")
        if p5:
            shortfall_pct = max(0.0, (book_value - p5) / book_value * 100)
            color = "#9bbb59"
            if shortfall_pct > 15:
                color = "#c0504d"
            elif shortfall_pct > 5:
                color = "#f1c232"
            ax2.text(0.5, 0.65, f"{shortfall_pct:.1f}%", fontsize=42, ha="center", fontweight="bold", color=color)
            ax2.text(0.5, 0.35, "P5 Shortfall vs. Book", ha="center", fontsize=12)
    ax2.set_title("Risk Gauge", fontweight="bold")

    # Middle row: PV distribution
    ax3 = fig.add_subplot(gs[1, :2])
    values = pv_distribution["present_value"].to_numpy() / 1e6
    ax3.hist(values, bins=50, color="#4f81bd", alpha=0.75, edgecolor="black")
    if book_value:
        ax3.axvline(book_value / 1e6, color="black", linestyle="--", linewidth=2, label="Book Value")
    if base_case_pv:
        ax3.axvline(base_case_pv / 1e6, color="orange", linestyle=":", linewidth=2, label="Base Case")
    if percentiles:
        for key, color in [("p5", "red"), ("p50", "navy"), ("p95", "green")]:
            if key in percentiles:
                ax3.axvline(percentiles[key] / 1e6, color=color, linestyle="-", linewidth=1.8, label=key.upper())
    ax3.set_title("Portfolio PV Distribution", fontweight="bold")
    ax3.set_xlabel("PV ($ millions)")
    ax3.set_ylabel("Frequency")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3, axis="y")

    # Middle-right: Percentile table
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    if percentiles:
        rows = [["Percentile", "Value ($M)"]]
        for key in ["p1", "p5", "p10", "p50", "p90", "p95", "p99"]:
            if key in percentiles:
                rows.append([key.upper(), f"{percentiles[key] / 1e6:,.0f}"])
        table = ax4.table(cellText=rows, cellLoc="left", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.6)
        for i in range(2):
            table[(0, i)].set_facecolor("#4f81bd")
            table[(0, i)].set_text_props(color="white", fontweight="bold")
    ax4.set_title("PV Percentiles", fontweight="bold")

    # Bottom-left: Percentile ladder
    ax5 = fig.add_subplot(gs[2, 0])
    if percentiles:
        y_labels = ["P1", "P5", "P10", "P25", "P50", "P75", "P90", "P95", "P99"]
        vals = [percentiles.get(k.lower(), np.nan) / 1e6 for k in y_labels]
        mask = ~np.isnan(vals)
        y = np.arange(len(y_labels))[mask]
        ax5.barh(y, np.array(vals)[mask], color="#4f81bd", alpha=0.7)
        if book_value:
            ax5.axvline(book_value / 1e6, color="black", linestyle="--", linewidth=2, label="Book")
        if base_case_pv:
            ax5.axvline(base_case_pv / 1e6, color="orange", linestyle=":", linewidth=2, label="Base")
        ax5.set_yticks(y)
        ax5.set_yticklabels(np.array(y_labels)[mask])
        ax5.set_xlabel("PV ($ millions)")
        ax5.set_title("Percentile Ladder", fontweight="bold")
        if book_value or base_case_pv:
            ax5.legend(loc="lower right")
        ax5.grid(True, alpha=0.3, axis="x")
    else:
        ax5.axis("off")

    # Bottom-middle: Scenario comparison
    ax6 = fig.add_subplot(gs[2, 1])
    if percentiles and base_case_pv:
        bars = [
            ("Base", base_case_pv / 1e6, "orange"),
            ("P5", percentiles.get("p5", np.nan) / 1e6, "#c0504d"),
            ("P50", percentiles.get("p50", np.nan) / 1e6, "#4f81bd"),
            ("P95", percentiles.get("p95", np.nan) / 1e6, "#9bbb59"),
        ]
        labels, values, colors = zip(*[(l, v, c) for l, v, c in bars if not np.isnan(v)])
        rects = ax6.bar(labels, values, color=colors, alpha=0.8, edgecolor="black")
        if book_value:
            ax6.axhline(book_value / 1e6, color="black", linestyle="--", linewidth=2)
        for rect in rects:
            height = rect.get_height()
            ax6.text(rect.get_x() + rect.get_width() / 2, height, f"${height:,.0f}M", ha="center", va="bottom")
        ax6.set_ylabel("PV ($ millions)")
        ax6.set_title("Scenario Comparison", fontweight="bold")
        ax6.grid(True, alpha=0.3, axis="y")
    else:
        ax6.axis("off")

    # Bottom-right: Distribution stats
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")
    if percentiles:
        stats = [
            f"Simulations: {len(pv_distribution):,}",
            f"Mean: ${percentiles['mean']/1e6:,.0f}M",
            f"Median: ${percentiles.get('p50', np.nan)/1e6:,.0f}M",
            f"Std Dev: ${percentiles['std']/1e6:,.0f}M",
            f"P5: ${percentiles.get('p5', np.nan)/1e6:,.0f}M",
            f"P95: ${percentiles.get('p95', np.nan)/1e6:,.0f}M",
        ]
        ax7.text(0.05, 0.9, "Key Metrics", fontsize=12, fontweight="bold")
        for idx, line in enumerate(stats):
            ax7.text(0.05, 0.7 - idx * 0.12, line, fontsize=10)

    fig.suptitle("Monte Carlo ALM Dashboard", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def save_figure(fig: plt.Figure, output_path: Path) -> Path:
    """Persist matplotlib figure to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path

