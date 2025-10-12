"""Report generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from ..models.results import EngineResults


class ReportGenerator:
    """Export engine outputs to CSV files."""

    def __init__(self, output_dir: str | Path = "output") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_summary(self, results: EngineResults, filename: str = "eve_summary.csv") -> Path:
        """Export the scenario level PV summary."""
        path = self.output_dir / filename
        summary = results.summary_frame()
        summary.to_csv(path, index=False)
        return path

    def export_cashflows(
        self,
        results: EngineResults,
        scenario_id: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Export detailed cash flows for a specific scenario."""
        result = results.scenario_results.get(scenario_id)
        if result is None:
            raise KeyError(f"Scenario {scenario_id!r} not present in engine results.")
        filename = filename or f"cashflows_{scenario_id}.csv"
        path = self.output_dir / filename
        result.cashflows.to_csv(path, index=False)
        return path

    def export_account_pv(
        self,
        results: EngineResults,
        scenario_id: str,
        filename: Optional[str] = None,
    ) -> Path:
        """Export account level PV detail."""
        result = results.scenario_results.get(scenario_id)
        if result is None:
            raise KeyError(f"Scenario {scenario_id!r} not present in engine results.")
        filename = filename or f"account_pv_{scenario_id}.csv"
        path = self.output_dir / filename
        result.account_level_pv.to_csv(path, index=False)
        return path
