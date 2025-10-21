"""Shared helpers for formatting matplotlib figures."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple, Callable, Dict

import matplotlib.pyplot as plt

NumberFormatter = Callable[[float], str]
LineSpec = Tuple[float, str, Dict[str, object]]


def _default_currency_formatter(value: float) -> str:
    return f"${value:,.0f}"


def label_horizontal_bars(
    ax: plt.Axes,
    bars: Iterable[plt.Rectangle],
    *,
    formatter: Optional[NumberFormatter] = None,
    margin_ratio: float = 0.15,
    inside_ratio: float = 0.25,
    padding_ratio: float = 0.04,
    text_kwargs: Optional[Dict[str, object]] = None,
) -> None:
    """Annotate horizontal bars with value labels and adjust x-limits."""
    formatter = formatter or _default_currency_formatter
    text_kwargs = text_kwargs.copy() if text_kwargs else {"fontsize": 9, "fontweight": "bold", "color": "#203040"}

    bars = list(bars)
    values = [float(bar.get_width()) for bar in bars]
    max_value = max(values, default=0.0)

    x_min, x_max = ax.get_xlim()
    target_max = max(x_max, max_value * (1.0 + margin_ratio))
    if target_max == 0:
        target_max = 1.0
    ax.set_xlim(x_min, target_max)

    inside_threshold = max_value * inside_ratio
    pad = target_max * padding_ratio

    for bar, value in zip(bars, values):
        y_center = bar.get_y() + bar.get_height() / 2.0
        if value >= inside_threshold:
            text_x = max(value - pad, 0.0)
            ha = "right"
        else:
            text_x = min(value + pad, target_max)
            ha = "left"
        ax.text(
            text_x,
            y_center,
            formatter(value),
            ha=ha,
            va="center",
            clip_on=True,
            **text_kwargs,
        )


def label_vertical_bars(
    ax: plt.Axes,
    bars: Iterable[plt.Rectangle],
    *,
    formatter: Optional[NumberFormatter] = None,
    margin_ratio: float = 0.12,
    inside_ratio: float = 0.25,
    padding_ratio: float = 0.035,
    text_kwargs: Optional[Dict[str, object]] = None,
) -> None:
    """Annotate vertical bars with value labels and adjust y-limits."""
    formatter = formatter or _default_currency_formatter
    text_kwargs = text_kwargs.copy() if text_kwargs else {"fontsize": 10, "fontweight": "bold", "color": "#1f1f1f"}

    bars = list(bars)
    values = [float(bar.get_height()) for bar in bars]
    max_value = max(values, default=0.0)

    y_min, y_max = ax.get_ylim()
    target_max = max(y_max, max_value * (1.0 + margin_ratio))
    if target_max == 0:
        target_max = 1.0
    ax.set_ylim(y_min, target_max)

    inside_threshold = max_value * inside_ratio
    pad = target_max * padding_ratio

    for bar, value in zip(bars, values):
        x_center = bar.get_x() + bar.get_width() / 2.0
        if value >= inside_threshold:
            text_y = max(value - pad, y_min)
            va = "top"
        else:
            text_y = min(value + pad, target_max)
            va = "bottom"
        ax.text(
            x_center,
            text_y,
            formatter(value),
            ha="center",
            va=va,
            clip_on=True,
            **text_kwargs,
        )


def annotate_reference_lines(
    ax: plt.Axes,
    lines: Sequence[LineSpec],
    *,
    orientation: str = "vertical",
    pad_ratio: float = 0.02,
    text_kwargs: Optional[Dict[str, object]] = None,
) -> None:
    """Attach labels to reference lines with automatic padding."""
    if not lines:
        return

    text_kwargs = text_kwargs.copy() if text_kwargs else {"fontsize": 9, "fontweight": "bold"}

    if orientation == "vertical":
        y_min, y_max = ax.get_ylim()
        span = max(y_max - y_min, 1e-6)
        offset = span * pad_ratio
        y_base = y_max - offset
        for position, label, style in lines:
            ax.text(
                position,
                y_base,
                label,
                ha="center",
                va="bottom",
                clip_on=True,
                color=style.get("color"),
                **text_kwargs,
            )
            secondary = style.get("secondary_label")
            if secondary:
                ax.text(
                    position,
                    y_base - offset,
                    secondary,
                    ha="center",
                    va="top",
                    clip_on=True,
                    color=style.get("color"),
                    alpha=0.85,
                    **{k: v for k, v in text_kwargs.items() if k != "fontweight"},
                )
    else:
        x_min, x_max = ax.get_xlim()
        span = max(x_max - x_min, 1e-6)
        offset = span * pad_ratio
        x_base = x_max - offset
        for position, label, style in lines:
            ax.text(
                x_base,
                position,
                label,
                ha="right",
                va="center",
                clip_on=True,
                color=style.get("color"),
                **text_kwargs,
            )
            secondary = style.get("secondary_label")
            if secondary:
                ax.text(
                    x_base - offset,
                    position,
                    secondary,
                    ha="left",
                    va="center",
                    clip_on=True,
                    color=style.get("color"),
                    alpha=0.85,
                    **{k: v for k, v in text_kwargs.items() if k != "fontweight"},
                )
