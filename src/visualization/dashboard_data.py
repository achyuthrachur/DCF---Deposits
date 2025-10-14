"""Data preparation helpers for Monte Carlo dashboards."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from src.models.results import EngineResults


def extract_monte_carlo_payload(
    results: EngineResults,
    scenario_id: str = "monte_carlo",
) -> Optional[Dict[str, object]]:
    """
    Extract Monte Carlo specific tables from ``EngineResults`` for visualisation.
    """
    scenario_result = results.scenario_results.get(scenario_id)
    if scenario_result is None:
        return None

    tables = scenario_result.extra_tables or {}
    pv_distribution = tables.get("simulation_pv")
    rate_summary = tables.get("rate_paths_summary")
    rate_sample = tables.get("rate_paths_sample")
    progress = tables.get("simulation_progress")
    if pv_distribution is None or rate_summary is None:
        return None

    book_value = None
    if results.validation_summary:
        book_value = results.validation_summary.get("portfolio_balance")

    base_case_pv = None
    if results.base_scenario_id and results.base_scenario_id in results.scenario_results:
        base_case_pv = results.scenario_results[results.base_scenario_id].present_value

    percentiles = scenario_result.metadata.get("pv_percentiles", {})

    return {
        "scenario_id": scenario_id,
        "pv_distribution": pv_distribution,
        "rate_summary": rate_summary,
        "rate_sample": rate_sample,
        "progress": progress,
        "percentiles": percentiles,
        "metadata": scenario_result.metadata,
        "book_value": book_value,
        "base_case_pv": base_case_pv,
    }
