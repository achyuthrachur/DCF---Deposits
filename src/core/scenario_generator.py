"""Factory helpers for standard interest rate scenarios."""

from __future__ import annotations

from typing import Dict, Iterable, List

from ..models.scenario import ScenarioDefinition, ScenarioSet, ScenarioType


def build_parallel_shock(
    shock_bps: int, projection_months: int, scenario_id: str
) -> ScenarioDefinition:
    """Create an immediate parallel rate shock scenario."""
    shock_decimal = shock_bps / 10000
    shock_vector = {month: shock_decimal for month in range(1, projection_months + 1)}
    return ScenarioDefinition(
        scenario_id=scenario_id,
        scenario_type=ScenarioType.PARALLEL,
        shock_vector=shock_vector,
        description=f"{shock_bps:+} bps parallel shock",
    )


def build_base_case(projection_months: int) -> ScenarioDefinition:
    """Base case scenario with no shocks."""
    return ScenarioDefinition(
        scenario_id="base",
        scenario_type=ScenarioType.BASE,
        shock_vector={month: 0.0 for month in range(1, projection_months + 1)},
        description="Base case with current rates",
    )


def assemble_standard_scenarios(
    selections: Dict[str, bool], projection_months: int
) -> ScenarioSet:
    """Assemble a scenario set based on user selections."""
    scenario_set = ScenarioSet()
    if selections.get("base", True):
        scenario_set.add(build_base_case(projection_months))

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
                )
            )
    return scenario_set
