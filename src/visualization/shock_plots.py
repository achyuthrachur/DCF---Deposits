"""Visualization helpers for deterministic shock scenarios."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from ..models.results import EngineResults


def _setup_figure(figsize=(10, 6)):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _format_currency_axis(ax, values, axis: str = "x") -> None:
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


def extract_shock_data(results: EngineResults, scenario_id: str) -> Optional[Dict[str, object]]:
    scenario_result = results.scenario_results.get(scenario_id)
    if scenario_result is None:
        return None

    method = (scenario_result.metadata or {}).get("method")
    if method in {None, "base", "monte_carlo"}:
        return None

    curve_df = scenario_result.extra_tables.get("curve_comparison")
    if curve_df is None or curve_df.empty:
        return None

    tenor_df = scenario_result.extra_tables.get("curve_tenor_comparison")
    base_result = (
        results.scenario_results.get(results.base_scenario_id)
        if results.base_scenario_id
        else None
    )
    base_pv = float(base_result.present_value) if base_result else None

    scenario_label = (
        (scenario_result.metadata or {}).get("description")
        or scenario_id.replace("_", " ").title()
    )
    scenario_pv = float(scenario_result.present_value)
    delta = None
    delta_pct = None
    if base_pv is not None and base_pv != 0:
        delta = scenario_pv - base_pv
        delta_pct = delta / base_pv

    shock_stats = (scenario_result.metadata or {}).get("shock_stats", {})
    abs_max_bps = float(shock_stats.get("abs_max_bps", 0.0))
    mean_bps = float(shock_stats.get("mean_bps", 0.0))

    return {
        "scenario_id": scenario_id,
        "scenario_label": scenario_label,
        "method": method,
        "curve_comparison": curve_df,
        "tenor_comparison": tenor_df,
        "scenario_pv": scenario_pv,
        "base_pv": base_pv,
        "delta": delta,
        "delta_pct": delta_pct,
        "abs_max_bps": abs_max_bps,
        "mean_bps": mean_bps,
    }


def plot_shock_rate_paths(curve_df: pd.DataFrame, scenario_label: str) -> plt.Figure:
    fig, ax = _setup_figure(figsize=(12, 6))
    months = curve_df["month"].to_numpy(dtype=float)
    base_rates = curve_df["base_rate"].to_numpy(dtype=float) * 100
    shocked_rates = curve_df["scenario_rate"].to_numpy(dtype=float) * 100

    ax.plot(months, base_rates, color="#4f81bd", linewidth=2.2, label="Base curve")
    ax.plot(months, shocked_rates, color="#c0504d", linewidth=2.2, label=scenario_label)
    ax.fill_between(
        months,
        base_rates,
        shocked_rates,
        where=shocked_rates >= base_rates,
        interpolate=True,
        color="#f4cccc",
        alpha=0.3,
        label="Shock uplift",
    )
    ax.fill_between(
        months,
        base_rates,
        shocked_rates,
        where=shocked_rates < base_rates,
        interpolate=True,
        color="#c9daf8",
        alpha=0.3,
        label="Shock reduction",
    )

    ax.set_title(f"Rate Path Comparison – {scenario_label}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Projection Month")
    ax.set_ylabel("Interest Rate (%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_shock_magnitude(curve_df: pd.DataFrame, scenario_label: str) -> plt.Figure:
    fig, ax = _setup_figure(figsize=(12, 4))
    months = curve_df["month"].to_numpy(dtype=float)
    shock_bps = curve_df["shock_bps"].to_numpy(dtype=float)

    ax.axhline(0, color="#666666", linewidth=1.2, linestyle="--")
    ax.plot(months, shock_bps, color="#6d9eeb", linewidth=2)
    ax.fill_between(
        months, 0, shock_bps, where=shock_bps >= 0, color="#9fc5e8", alpha=0.5
    )
    ax.fill_between(
        months, 0, shock_bps, where=shock_bps < 0, color="#f4cccc", alpha=0.5
    )

    ax.set_title(f"Shock Magnitude Over Time – {scenario_label}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Projection Month")
    ax.set_ylabel("Shock (basis points)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_shock_tenor_comparison(
    tenor_df: pd.DataFrame, scenario_label: str
) -> Optional[plt.Figure]:
    if tenor_df is None or tenor_df.empty:
        return None

    fig, ax = _setup_figure(figsize=(10, 5))
    tenor_years = tenor_df["tenor_years"].to_numpy(dtype=float)
    base_rates = tenor_df["base_rate"].to_numpy(dtype=float) * 100
    scenario_rates = tenor_df["scenario_rate"].to_numpy(dtype=float) * 100

    ax.plot(
        tenor_years,
        base_rates,
        color="#4f81bd",
        linewidth=2,
        marker="o",
        label="Base curve",
    )
    ax.plot(
        tenor_years,
        scenario_rates,
        color="#c0504d",
        linewidth=2,
        marker="o",
        label=scenario_label,
    )

    ax.set_title("Curve Comparison by Tenor", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tenor (years)")
    ax.set_ylabel("Interest Rate (%)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def plot_shock_pv_delta(
    base_pv: float,
    scenario_pv: float,
    scenario_label: str,
    delta: Optional[float],
    delta_pct: Optional[float],
) -> Optional[plt.Figure]:
    if base_pv is None:
        return None

    fig, ax = _setup_figure(figsize=(6.5, 4.5))
    labels = ["Base case", scenario_label]
    values = [base_pv, scenario_pv]
    colors = ["#4f81bd", "#c0504d"]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="black")
    _format_currency_axis(ax, values, axis="y")
    ax.set_ylabel("Present Value ($)")
    ax.set_title("Portfolio PV Comparison", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.25)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"${value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    if delta is not None:
        annotation = f"Δ {delta:+,.0f}"
        if delta_pct is not None:
            annotation += f" ({delta_pct * 100:+.2f}%)"
        ax.text(
            0.5,
            max(values) * 1.02,
            annotation,
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="#333333",
        )

    fig.tight_layout()
    return fig


def plot_shock_group_summary(
    results: EngineResults,
    scenario_ids: Sequence[str],
    *,
    title: str,
) -> Optional[plt.Figure]:
    """Plot PV comparison across a group of deterministic shock scenarios."""
    if not scenario_ids:
        return None

    base_result = (
        results.scenario_results.get(results.base_scenario_id)
        if results.base_scenario_id
        else None
    )
    base_pv = float(base_result.present_value) if base_result else None

    rows: list[Dict[str, object]] = []
    if base_result is not None:
        base_label = (
            base_result.metadata.get("description")
            if base_result.metadata
            else None
        )
        rows.append(
            {
                "label": base_label or "Base case",
                "pv": base_pv,
                "delta": 0.0,
                "color": "#4f81bd",
            }
        )

    for scenario_id in scenario_ids:
        scenario = results.scenario_results.get(scenario_id)
        if scenario is None:
            continue
        label = (
            scenario.metadata.get("description")
            if scenario.metadata
            else None
        )
        pv = float(scenario.present_value)
        delta = pv - base_pv if base_pv is not None else None
        color = "#c0504d" if delta is not None and delta < 0 else "#9bbb59"
        rows.append(
            {
                "label": label or scenario_id.replace("_", " ").title(),
                "pv": pv,
                "delta": delta,
                "color": color,
            }
        )

    if not rows:
        return None

    labels = [row["label"] for row in rows]
    values = [row["pv"] for row in rows]
    colors = [row["color"] for row in rows]

    fig, ax = _setup_figure(figsize=(10.5, 6))
    positions = np.arange(len(labels))
    ax.bar(positions, values, color=colors, edgecolor="black", alpha=0.85)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")

    for idx, row in enumerate(rows):
        ax.text(
            positions[idx],
            values[idx],
            f"${values[idx]:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
        delta = row.get("delta")
        if delta is not None and abs(delta) > 1e-6:
            ax.text(
                positions[idx],
                values[idx] * 0.98,
                f"{delta:+,.0f}",
                ha="center",
                va="top",
                fontsize=9,
                color="#333333",
            )

    ax.set_ylabel("Present Value ($)")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    _format_currency_axis(ax, values, axis="y")
    fig.tight_layout()
    return fig
