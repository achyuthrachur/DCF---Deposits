"""Report generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import json

import pandas as pd

from ..models.results import EngineResults
from ..visualization import (
    create_monte_carlo_dashboard,
    extract_monte_carlo_data,
    plot_percentile_ladder,
    plot_portfolio_pv_distribution,
    plot_rate_confidence_fan,
    plot_rate_path_spaghetti,
)
from ..visualization.monte_carlo_plots import save_figure


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

    @staticmethod
    def sample_cashflows(
        cashflows: pd.DataFrame,
        sample_size: int = 20,
        random_state: Optional[int] = None,
    ) -> pd.DataFrame:
        """Return a sampled subset of cashflows prioritising higher balances."""
        if sample_size <= 0:
            return cashflows
        unique_accounts = cashflows["account_id"].nunique()
        if unique_accounts <= sample_size:
            return cashflows

        starting_balances = (
            cashflows.sort_values("month")
            .groupby("account_id", as_index=False)
            .first()[["account_id", "beginning_balance"]]
        )
        candidate_count = min(
            unique_accounts,
            max(sample_size * 3, sample_size),
        )
        top_candidates = starting_balances.nlargest(
            candidate_count, "beginning_balance"
        )
        sampled_ids = top_candidates["account_id"].sample(
            n=min(sample_size, len(top_candidates)),
            replace=False,
            random_state=random_state,
        )
        return cashflows[cashflows["account_id"].isin(sampled_ids.tolist())]

    def export_cashflows(
        self,
        results: EngineResults,
        scenario_id: str,
        filename: Optional[str] = None,
        sample_size: int = 20,
        random_state: Optional[int] = None,
    ) -> Path:
        """Export detailed cash flows for a specific scenario."""
        result = results.scenario_results.get(scenario_id)
        if result is None:
            raise KeyError(f"Scenario {scenario_id!r} not present in engine results.")
        filename = filename or f"cashflows_{scenario_id}.csv"
        path = self.output_dir / filename
        sampled = self.sample_cashflows(
            result.cashflows, sample_size=sample_size, random_state=random_state
        )
        sampled.to_csv(path, index=False)
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

    def export_validation_summary(
        self,
        results: EngineResults,
        filename: str = "validation_summary.json",
    ) -> Optional[Path]:
        """Export validation summary if available."""
        if not results.validation_summary:
            return None
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(results.validation_summary, fh, indent=2)
        return path

    def export_monte_carlo_visuals(
        self,
        results: EngineResults,
        scenario_id: str = "monte_carlo",
        prefix: str = "monte_carlo",
    ) -> Dict[str, Path]:
        """Generate core Monte Carlo charts and persist them to disk."""

        data = extract_monte_carlo_data(results, scenario_id=scenario_id)
        if not data:
            return {}

        output_paths: Dict[str, Path] = {}

        try:
            fig = plot_rate_path_spaghetti(data["rate_sample"], data["rate_summary"])
            output_paths["rate_spaghetti"] = save_figure(
                fig, self.output_dir / f"{prefix}_rate_spaghetti.png"
            )

            fig = plot_rate_confidence_fan(data["rate_summary"])
            output_paths["rate_fan"] = save_figure(
                fig, self.output_dir / f"{prefix}_rate_fan.png"
            )

            fig = plot_portfolio_pv_distribution(
                data["pv_distribution"],
                book_value=data.get("book_value"),
                base_case_pv=data.get("base_case_pv"),
                percentiles=data.get("percentiles"),
            )
            output_paths["pv_distribution"] = save_figure(
                fig, self.output_dir / f"{prefix}_pv_distribution.png"
            )

            fig = plot_percentile_ladder(
                data.get("percentiles", {}),
                book_value=data.get("book_value"),
                base_case_pv=data.get("base_case_pv"),
            )
            output_paths["pv_percentiles"] = save_figure(
                fig, self.output_dir / f"{prefix}_percentiles.png"
            )

            fig = create_monte_carlo_dashboard(data)
            output_paths["dashboard"] = save_figure(
                fig, self.output_dir / f"{prefix}_dashboard.png"
            )
        except Exception:  # pragma: no cover - visual generation is best-effort
            return output_paths

        return output_paths
