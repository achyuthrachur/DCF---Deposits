"""Numeric helper functions shared across the application."""

from __future__ import annotations

from typing import Optional


def decimalize(value: Optional[float]) -> Optional[float]:
    """Convert percentage-based inputs to decimals while preserving None."""
    if value is None:
        return None
    if value > 1.5:
        return float(value / 100.0)
    return float(value)


__all__ = ["decimalize"]
