"""Scenario data models."""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ScenarioType(str, Enum):
    """Enumeration of supported scenario archetypes."""

    BASE = "base"
    PARALLEL = "parallel"
    RAMP = "ramp"
    STEEPENER = "steepener"
    FLATTENER = "flattener"
    CUSTOM = "custom"
    MONTE_CARLO = "monte_carlo"


class ScenarioDefinition(BaseModel):
    """Defines how market rates evolve relative to the base curve."""

    scenario_id: str = Field(..., description="Unique scenario identifier")
    scenario_type: ScenarioType = Field(..., description="Scenario classification")
    shock_vector: Dict[int, float] = Field(
        default_factory=dict,
        description=(
            "Mapping of month offset (1-based) to rate shock in decimal form. "
            "Positive values indicate rising rate adjustments."
        ),
    )
    description: Optional[str] = Field(
        None, description="Human-readable scenario description"
    )
    metadata: Dict[str, str] = Field(
        default_factory=dict, description="Additional configuration details"
    )

    def rate_adjustment(self, month_index: int) -> float:
        """Return the incremental rate adjustment for a given month."""
        return self.shock_vector.get(month_index, self.shock_vector.get(0, 0.0))


class ScenarioSet(BaseModel):
    """Container for multiple scenarios to be evaluated."""

    scenarios: List[ScenarioDefinition] = Field(
        default_factory=list, description="Scenarios to evaluate"
    )

    def add(self, scenario: ScenarioDefinition) -> None:
        """Register a new scenario."""
        if any(s.scenario_id == scenario.scenario_id for s in self.scenarios):
            raise ValueError(f"Scenario {scenario.scenario_id!r} already exists")
        self.scenarios.append(scenario)

    def get(self, scenario_id: str) -> ScenarioDefinition:
        """Fetch a scenario by identifier."""
        for scenario in self.scenarios:
            if scenario.scenario_id == scenario_id:
                return scenario
        raise KeyError(f"Scenario {scenario_id!r} not found")
