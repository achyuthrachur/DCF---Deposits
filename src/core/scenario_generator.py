"""Factory helpers for standard interest rate scenarios."""

from __future__ import annotations

from typing import Dict, Optional

from ..models.scenario import ScenarioDefinition, ScenarioSet, ScenarioType
from .discount import DiscountCurve
from .monte_carlo import MonteCarloConfig


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


def _build_curve_scenario(
    scenario_id: str,
    scenario_type: ScenarioType,
    base_curve: DiscountCurve,
    shocked_curve: DiscountCurve,
    projection_months: int,
    description: str,
) -> ScenarioDefinition:
    """Create a scenario definition using the difference between two curves."""
    shock_vector = {
        month: float(shocked_curve.get_rate(month) - base_curve.get_rate(month))
        for month in range(1, projection_months + 1)
    }
    metadata = {
        "base_curve": base_curve.to_dict(),
        "scenario_curve": shocked_curve.to_dict(),
        "source": "generated",
    }
    return ScenarioDefinition(
        scenario_id=scenario_id,
        scenario_type=scenario_type,
        shock_vector=shock_vector,
        description=description,
        metadata=metadata,
    )


def _parallel_shock_curve(
    base_curve: DiscountCurve,
    shock_bps: int,
) -> DiscountCurve:
    shocked = base_curve.apply_parallel_shock(shock_bps)
    return DiscountCurve(shocked.tenors, shocked.rates, interpolation_method=shocked.interpolation_method)


def _non_parallel_shock_curve(
    base_curve: DiscountCurve,
    shocks_by_tenor: Dict[int, int],
) -> DiscountCurve:
    shocked = base_curve.apply_non_parallel_shock(shocks_by_tenor)
    return DiscountCurve(shocked.tenors, shocked.rates, interpolation_method=shocked.interpolation_method)


def assemble_standard_scenarios(
    selections: Dict[str, bool],
    projection_months: int,
    monte_carlo_config: Optional[MonteCarloConfig] = None,
    *,
    base_discount_curve: Optional[DiscountCurve] = None,
) -> ScenarioSet:
    """Assemble a scenario set based on user selections."""
    scenario_set = ScenarioSet()
    if base_discount_curve is None:
        base_discount_curve = DiscountCurve.from_single_rate(0.03)

    if selections.get("base", True):
        scenario_set.add(
            _build_curve_scenario(
                scenario_id="base",
                scenario_type=ScenarioType.BASE,
                base_curve=base_discount_curve,
                shocked_curve=base_discount_curve,
                projection_months=projection_months,
                description="Base case with current rates",
            )
        )

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
            shocked_curve = _parallel_shock_curve(base_discount_curve, shock)
            scenario_set.add(
                _build_curve_scenario(
                    scenario_id=scenario_id,
                    scenario_type=ScenarioType.PARALLEL,
                    base_curve=base_discount_curve,
                    shocked_curve=shocked_curve,
                    projection_months=projection_months,
                    description=f"{shock:+} bps parallel shock",
                )
            )

    # Optional non-parallel scenarios
    steepener = {
        3: -100,
        6: -75,
        12: -50,
        24: 0,
        36: 25,
        60: 50,
        84: 75,
        120: 100,
    }
    if selections.get("steepener", False):
        shocked_curve = _non_parallel_shock_curve(base_discount_curve, steepener)
        scenario_set.add(
            _build_curve_scenario(
                scenario_id="steepener",
                scenario_type=ScenarioType.STEEPENER,
                base_curve=base_discount_curve,
                shocked_curve=shocked_curve,
                projection_months=projection_months,
                description="Curve steepener (short down, long up)",
            )
        )

    if selections.get("flattener", False):
        flattener = {tenor: -shock for tenor, shock in steepener.items()}
        shocked_curve = _non_parallel_shock_curve(base_discount_curve, flattener)
        scenario_set.add(
            _build_curve_scenario(
                scenario_id="flattener",
                scenario_type=ScenarioType.FLATTENER,
                base_curve=base_discount_curve,
                shocked_curve=shocked_curve,
                projection_months=projection_months,
                description="Curve flattener (short up, long down)",
            )
        )

    if selections.get("short_shock_up", False):
        short_rate_shock = {
            3: 200,
            6: 175,
            12: 150,
            24: 100,
            36: 50,
            60: 25,
            84: 10,
            120: 0,
        }
        shocked_curve = _non_parallel_shock_curve(base_discount_curve, short_rate_shock)
        scenario_set.add(
            _build_curve_scenario(
                scenario_id="short_shock_up",
                scenario_type=ScenarioType.CUSTOM,
                base_curve=base_discount_curve,
                shocked_curve=shocked_curve,
                projection_months=projection_months,
                description="Short-rate focussed shock (+200 bps front-end)",
            )
        )

    if selections.get("monte_carlo", False) and monte_carlo_config:
        scenario_set.add(build_monte_carlo(projection_months, monte_carlo_config))

    return scenario_set
