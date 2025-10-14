"""Factory helpers for standard interest rate scenarios."""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..models.scenario import ScenarioDefinition, ScenarioSet, ScenarioType
from .discount import DiscountCurve
from .yield_curve import YieldCurve
from .monte_carlo import MonteCarloConfig


def _serialise_curve(curve: YieldCurve) -> Dict[str, object]:
    return curve.to_dict()


def _shock_vector_from_tenors(
    shocks_by_tenor: Dict[int, float], projection_months: int
) -> Dict[int, float]:
    """Interpolate tenor-based shocks (in bps) to monthly decimal adjustments."""
    if not shocks_by_tenor:
        return {month: 0.0 for month in range(1, projection_months + 1)}
    tenors, shocks = zip(*sorted(shocks_by_tenor.items()))
    # Ensure at least two points for interpolation.
    if len(tenors) == 1:
        tenors = (tenors[0], projection_months)
        shocks = (shocks[0], shocks[0])
    shock_curve = YieldCurve(tenors, [s / 10000.0 for s in shocks])
    return {
        month: float(shock_curve.get_rate(month))
        for month in range(1, projection_months + 1)
    }


def build_base_case(
    projection_months: int,
    base_curve: Optional[YieldCurve] = None,
) -> ScenarioDefinition:
    """Base case scenario with no shocks."""
    metadata: Dict[str, Any] = {}
    if base_curve is not None:
        metadata["yield_curve"] = _serialise_curve(base_curve)
    return ScenarioDefinition(
        scenario_id="base",
        scenario_type=ScenarioType.BASE,
        shock_vector={month: 0.0 for month in range(1, projection_months + 1)},
        description="Base case with current rates",
        metadata=metadata,
    )


def build_parallel_shock(
    shock_bps: int,
    projection_months: int,
    scenario_id: str,
    base_curve: Optional[YieldCurve] = None,
) -> ScenarioDefinition:
    """Create an immediate parallel rate shock scenario."""
    shock_decimal = shock_bps / 10000.0
    metadata: Dict[str, Any] = {}
    if base_curve is not None:
        metadata["yield_curve"] = _serialise_curve(base_curve.apply_parallel_shock(shock_bps))
    return ScenarioDefinition(
        scenario_id=scenario_id,
        scenario_type=ScenarioType.PARALLEL,
        shock_vector={month: shock_decimal for month in range(1, projection_months + 1)},
        description=f"{shock_bps:+} bps parallel shock",
        metadata=metadata,
    )


def build_curve_shock(
    scenario_id: str,
    description: str,
    scenario_type: ScenarioType,
    shocks_by_tenor: Dict[int, float],
    projection_months: int,
    base_curve: Optional[YieldCurve] = None,
) -> ScenarioDefinition:
    """Create a scenario that applies tenor-specific shocks."""
    metadata: Dict[str, Any] = {}
    if base_curve is not None:
        metadata["yield_curve"] = _serialise_curve(
            base_curve.apply_non_parallel_shock(shocks_by_tenor)
        )
    shock_vector = _shock_vector_from_tenors(shocks_by_tenor, projection_months)
    return ScenarioDefinition(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        shock_vector=shock_vector,
        description=description,
        metadata=metadata,
    )


def build_monte_carlo(
    projection_months: int,
    config: MonteCarloConfig,
) -> ScenarioDefinition:
    """Create a Monte Carlo scenario placeholder with configuration metadata."""
    metadata = config.to_metadata()
    metadata["projection_months"] = projection_months
    return ScenarioDefinition(
        scenario_id="monte_carlo",
        scenario_type=ScenarioType.MONTE_CARLO,
        shock_vector={},
        description="Monte Carlo stochastic rate simulations",
        metadata=metadata,
    )


def assemble_standard_scenarios(
    selections: Dict[str, bool],
    projection_months: int,
    monte_carlo_config: Optional[MonteCarloConfig] = None,
    *,
    base_discount_curve: Optional[DiscountCurve] = None,
) -> ScenarioSet:
    """Assemble a scenario set based on user selections."""
    base_curve = base_discount_curve.yield_curve if base_discount_curve else None
    scenario_set = ScenarioSet()
    if selections.get("base", True):
        scenario_set.add(build_base_case(projection_months, base_curve))

    parallel_shocks = {
        "parallel_100": 100,
        "parallel_200": 200,
        "parallel_300": 300,
        "parallel_400": 400,
        "parallel_minus_100": -100,
        "parallel_minus_200": -200,
        "parallel_minus_300": -300,
        "parallel_minus_400": -400,
    }
    for scenario_id, shock in parallel_shocks.items():
        if selections.get(scenario_id, False):
            scenario_set.add(
                build_parallel_shock(
                    shock_bps=shock,
                    projection_months=projection_months,
                    scenario_id=scenario_id,
                    base_curve=base_curve,
                )
            )

    non_parallel_definitions: Dict[str, Dict[str, Any]] = {
        "steepener": {
            "description": "Yield curve steepener (short down / long up)",
            "scenario_type": ScenarioType.STEEPENER,
            "shocks": {
                3: -100,
                6: -75,
                12: -50,
                24: 0,
                36: 25,
                60: 50,
                84: 75,
                120: 100,
            },
        },
        "flattener": {
            "description": "Yield curve flattener (short up / long down)",
            "scenario_type": ScenarioType.FLATTENER,
            "shocks": {
                3: 100,
                6: 75,
                12: 50,
                24: 0,
                36: -25,
                60: -50,
                84: -75,
                120: -100,
            },
        },
        "short_shock_up": {
            "description": "Front-end shock (+200 bps at 3M tapering to flat by 10Y)",
            "scenario_type": ScenarioType.CUSTOM,
            "shocks": {
                3: 200,
                6: 175,
                12: 150,
                24: 100,
                36: 50,
                60: 25,
                84: 10,
                120: 0,
            },
        },
    }

    for scenario_id, cfg in non_parallel_definitions.items():
        if selections.get(scenario_id, False):
            scenario_set.add(
                build_curve_shock(
                    scenario_id=scenario_id,
                    description=cfg["description"],
                    scenario_type=cfg["scenario_type"],
                    shocks_by_tenor=cfg["shocks"],
                    projection_months=projection_months,
                    base_curve=base_curve,
                )
            )

    if selections.get("monte_carlo", False) and monte_carlo_config:
        scenario_set.add(build_monte_carlo(projection_months, monte_carlo_config))

    return scenario_set


def generate_standard_scenarios(base_curve: YieldCurve) -> Dict[str, YieldCurve]:
    """
    Convenience helper returning shocked curves keyed by scenario name.
    """
    scenarios: Dict[str, YieldCurve] = {"base": base_curve}
    for shock in [100, 200, 300, 400, -100, -200, -300, -400]:
        scenarios[f"parallel_{shock:+d}"] = base_curve.apply_parallel_shock(shock)

    steepener_shocks = {
        3: -100,
        6: -75,
        12: -50,
        24: 0,
        36: 25,
        60: 50,
        84: 75,
        120: 100,
    }
    flattener_shocks = {tenor: -value for tenor, value in steepener_shocks.items()}
    short_shock_up = {
        3: 200,
        6: 175,
        12: 150,
        24: 100,
        36: 50,
        60: 25,
        84: 10,
        120: 0,
    }

    scenarios["steepener"] = base_curve.apply_non_parallel_shock(steepener_shocks)
    scenarios["flattener"] = base_curve.apply_non_parallel_shock(flattener_shocks)
    scenarios["short_shock_up"] = base_curve.apply_non_parallel_shock(short_shock_up)

    return scenarios
