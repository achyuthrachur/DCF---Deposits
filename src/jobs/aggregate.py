"""Aggregate multiple EngineResults objects (Monte Carlo batching support)."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.models.results import EngineResults, ScenarioResult


def _is_monte_carlo(result: ScenarioResult) -> bool:
    method = (result.metadata or {}).get("method")
    return method == "monte_carlo"


def _get_num_sims(result: ScenarioResult) -> int:
    meta = result.metadata or {}
    config = meta.get("config") or meta.get("scenario_details", {}).get("config")
    if isinstance(config, dict):
        try:
            return int(config.get("num_simulations", 0))
        except Exception:
            return 0
    return 0


def merge_engine_results(results_list: List[EngineResults]) -> EngineResults:
    """Merge results by concatenating Monte Carlo distributions and recomputing stats.

    Deterministic scenarios are taken from the first results object.
    """
    if not results_list:
        raise ValueError("merge_engine_results requires at least one result")

    base = results_list[0]
    merged = EngineResults(
        scenario_results={},
        base_scenario_id=base.base_scenario_id,
        validation_summary=base.validation_summary,
        parameter_summary=base.parameter_summary,
    )

    # Index all Monte Carlo scenarios by id
    all_ids = set()
    for r in results_list:
        all_ids.update(r.scenario_results.keys())

    for scenario_id in sorted(all_ids):
        # Collect all scenario results for this id
        slices: List[ScenarioResult] = []
        for r in results_list:
            if scenario_id in r.scenario_results:
                slices.append(r.scenario_results[scenario_id])
        if not slices:
            continue
        first = slices[0]
        if not _is_monte_carlo(first):
            # Use the first deterministic scenario as-is
            merged.add_result(first)
            continue

        # Merge Monte Carlo distributions
        # Distribution table
        dist_frames: List[pd.DataFrame] = []
        num_sims_total = 0
        expected_cashflows_weighted: List[Tuple[pd.DataFrame, int]] = []
        book_value = None
        for s in slices:
            extra = s.extra_tables or {}
            dist = extra.get("monte_carlo_distribution") or extra.get("simulation_pv")
            if isinstance(dist, pd.DataFrame) and not dist.empty:
                dist_frames.append(dist[["portfolio_pv"]].copy())
            n_sims = _get_num_sims(s)
            if n_sims <= 0 and isinstance(dist, pd.DataFrame):
                n_sims = int(dist.shape[0])
            num_sims_total += n_sims
            if isinstance(s.cashflows, pd.DataFrame) and not s.cashflows.empty:
                expected_cashflows_weighted.append((s.cashflows.copy(), n_sims))
            if book_value is None:
                book_value = (first.metadata or {}).get("book_value")

        if not dist_frames:
            # Fallback: use first
            merged.add_result(first)
            continue

        combined_dist = pd.concat(dist_frames, ignore_index=True)
        combined_dist.index = np.arange(1, len(combined_dist) + 1)
        combined_dist.insert(0, "simulation", combined_dist.index)
        pv_series = combined_dist["portfolio_pv"].astype(float)
        mean_pv = float(pv_series.mean())
        percentiles = {
            "p1": float(pv_series.quantile(0.01)),
            "p5": float(pv_series.quantile(0.05)),
            "p10": float(pv_series.quantile(0.10)),
            "p25": float(pv_series.quantile(0.25)),
            "p50": float(pv_series.quantile(0.50)),
            "p75": float(pv_series.quantile(0.75)),
            "p90": float(pv_series.quantile(0.90)),
            "p95": float(pv_series.quantile(0.95)),
            "p99": float(pv_series.quantile(0.99)),
        }
        summary_row = {
            "mean": mean_pv,
            "median": percentiles["p50"],
            "p1": percentiles["p1"],
            "p5": percentiles["p5"],
            "p10": percentiles["p10"],
            "p25": percentiles["p25"],
            "p75": percentiles["p75"],
            "p90": percentiles["p90"],
            "p95": percentiles["p95"],
            "p99": percentiles["p99"],
            "std": float(pv_series.std(ddof=1)) if pv_series.size > 1 else 0.0,
            "skew": float(pv_series.skew()),
            "kurtosis": float(pv_series.kurt()),
            "var_95": None,
            "cvar_95": None,
            "prob_below_book": None,
        }
        if book_value is not None:
            try:
                bv = float(book_value)
                summary_row["var_95"] = bv - percentiles["p5"]
                cvar_mask = pv_series <= percentiles["p5"]
                summary_row["cvar_95"] = float(pv_series[cvar_mask].mean()) if cvar_mask.any() else percentiles["p5"]
                summary_row["prob_below_book"] = float((pv_series < bv).mean())
            except Exception:
                pass

        summary_df = pd.DataFrame([summary_row])
        percentiles_df = pd.DataFrame(
            {"percentile": list(range(5, 100, 5)),
             "portfolio_pv": [float(pv_series.quantile(p/100.0)) for p in range(5, 100, 5)]}
        )

        # Weighted expected cashflows across batches (by num simulations)
        if expected_cashflows_weighted:
            # Assume identical months index
            months = expected_cashflows_weighted[0][0]["month"].to_numpy()
            cols = [
                "beginning_balance",
                "ending_balance",
                "principal",
                "interest",
                "total_cash_flow",
                "deposit_rate",
                "monthly_decay_rate",
            ]
            accum = None
            total_weight = 0
            for df, weight in expected_cashflows_weighted:
                sub = df.set_index("month")[cols].astype(float)
                sub *= float(max(weight, 1))
                if accum is None:
                    accum = sub
                else:
                    accum = accum.add(sub, fill_value=0.0)
                total_weight += max(weight, 1)
            accum = accum / float(max(total_weight, 1))
            expected_cashflows = accum.reset_index().rename(columns={"index": "month"})
        else:
            expected_cashflows = first.cashflows

        metadata = dict(first.metadata or {})
        config = dict(metadata.get("config", {}))
        config["num_simulations"] = int(num_sims_total)
        metadata["config"] = config
        metadata["summary"] = summary_row
        # Keep other metadata fields (e.g., pv_percentiles) minimal; optional

        extra_tables = dict(first.extra_tables or {})
        extra_tables["monte_carlo_distribution"] = combined_dist
        extra_tables["monte_carlo_summary"] = summary_df
        extra_tables["monte_carlo_percentiles"] = percentiles_df
        extra_tables["simulation_pv"] = combined_dist
        extra_tables["summary_table"] = summary_df
        extra_tables["percentiles_table"] = percentiles_df

        merged.add_result(
            ScenarioResult(
                scenario_id=scenario_id,
                cashflows=expected_cashflows,
                present_value=mean_pv,
                account_level_pv=summary_df,
                metadata=metadata,
                extra_tables=extra_tables,
            )
        )

    return merged

