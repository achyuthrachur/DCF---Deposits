"""Unit tests for the YieldCurve helper."""

from __future__ import annotations

import math

import numpy as np

from src.core.yield_curve import YieldCurve


def test_exact_tenor_lookup() -> None:
    curve = YieldCurve([3, 6, 12, 24], [0.0425, 0.0415, 0.0410, 0.0395])
    assert math.isclose(curve.get_rate(12), 0.0410, abs_tol=1e-9)


def test_linear_interpolation() -> None:
    curve = YieldCurve([12, 24], [0.0410, 0.0395])
    interpolated = curve.get_rate(18)
    expected = (0.0410 + 0.0395) / 2.0
    assert math.isclose(interpolated, expected, rel_tol=1e-6)


def test_discount_factor() -> None:
    curve = YieldCurve([24], [0.0395])
    df = curve.get_discount_factor(24)
    expected = 1.0 / ((1.0 + 0.0395) ** 2)
    assert math.isclose(df, expected, rel_tol=1e-6)


def test_parallel_shock() -> None:
    curve = YieldCurve([12], [0.0410])
    shocked = curve.apply_parallel_shock(100)
    assert math.isclose(shocked.get_rate(12), 0.0510, rel_tol=1e-9)


def test_non_parallel_shock() -> None:
    curve = YieldCurve([3, 120], [0.0425, 0.0415])
    shocks = {3: -100, 120: 100}
    shocked = curve.apply_non_parallel_shock(shocks)
    assert math.isclose(shocked.get_rate(3), 0.0325, rel_tol=1e-9)
    assert math.isclose(shocked.get_rate(120), 0.0515, rel_tol=1e-9)
