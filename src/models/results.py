"""Result data models for reporting."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ScenarioResult(BaseModel):
    """Holds computed cash flows and PV metrics for a single scenario."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_id: str = Field(..., description="Scenario identifier")
    cashflows: pd.DataFrame = Field(
        ...,
        description=(
            "Monthly cash flow detail with columns: "
            "['account_id', 'month', 'principal', 'interest', 'balance', 'rate']"
        ),
    )
    present_value: float = Field(..., description="Portfolio PV for the scenario")
    account_level_pv: pd.DataFrame = Field(
        ...,
        description="Account-level PV detail with columns ['account_id', 'pv']",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extra_tables: Dict[str, pd.DataFrame] = Field(
        default_factory=dict,
        description="Additional data tables (e.g., distribution summaries)",
    )


class EngineResults(BaseModel):
    """Aggregates all scenario results and summary metrics."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario_results: Dict[str, ScenarioResult] = Field(
        default_factory=dict, description="Results keyed by scenario id"
    )
    base_scenario_id: Optional[str] = Field(
        default=None, description="Scenario used as the base case for comparisons"
    )

    def add_result(self, result: ScenarioResult) -> None:
        """Store a scenario result."""
        self.scenario_results[result.scenario_id] = result

    def summary_frame(self) -> pd.DataFrame:
        """Return a summary table of PVs and relative impacts."""
        if not self.scenario_results:
            return pd.DataFrame()
        rows = []
        base_pv = None
        if self.base_scenario_id and self.base_scenario_id in self.scenario_results:
            base_pv = self.scenario_results[self.base_scenario_id].present_value
        for scenario_id, result in self.scenario_results.items():
            pv = result.present_value
            delta = None
            delta_pct = None
            if base_pv is not None:
                delta = pv - base_pv
                delta_pct = delta / base_pv if base_pv else None
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "present_value": pv,
                    "vs_base": delta,
                    "vs_base_pct": delta_pct,
                }
            )
        return pd.DataFrame(rows)
