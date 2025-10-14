"""High-level orchestration for the NMD ALM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .core.cashflow_projector import CashflowProjector, ProjectionSettings
from .core.data_loader import DataLoader, LoadResult
from .core.discount import DiscountCurve
from .core.pv_calculator import PresentValueCalculator
from .core.pv_validation import validate_pv_results
from .core.scenario_generator import assemble_standard_scenarios
from .core.validator import ValidationError
from .models.account import AccountRecord
from .models.assumptions import AssumptionSet, SegmentAssumptions
from .models.results import EngineResults, ScenarioResult
from .models.scenario import ScenarioDefinition, ScenarioSet, ScenarioType


@dataclass
class DiscountConfig:
    """Captures discount rate configuration details."""

    method: str
    curve: DiscountCurve


class ALMEngine:
    """Primary entry point for configuring and running the ALM projection engine."""

    def __init__(self) -> None:
        self.accounts: List[AccountRecord] = []
        self._assumptions = AssumptionSet(segmentation_method="all")
        self.segmentation_method = "all"
        self.projection_months = 120
        self._market_rate_path: Optional[Sequence[float]] = None
        self._discount_config: Optional[DiscountConfig] = None
        self._scenario_set = ScenarioSet()
        self.field_map: Dict[str, str] = {}
        self._monte_carlo_config: Optional[Dict[str, Any]] = None
        self._source_dataframe: Optional["pd.DataFrame"] = None

    # --------------------------------------------------------------------- Data
    def load_data(
        self,
        file_path: str,
        field_map: Dict[str, str],
        optional_fields: Optional[Iterable[str]] = None,
        dtypes: Optional[Dict[str, str]] = None,
    ) -> LoadResult:
        """Load account data from CSV using the provided field mapping."""
        loader = DataLoader()
        result = loader.load_accounts(
            file_path=file_path,
            field_map=field_map,
            optional_fields=optional_fields,
            dtypes=dtypes,
        )
        self.accounts = result.accounts
        self.field_map = dict(field_map)
        self._source_dataframe = result.dataframe.copy()
        return result

    def load_dataframe(
        self,
        dataframe: "pd.DataFrame",
        field_map: Dict[str, str],
        optional_fields: Optional[Iterable[str]] = None,
    ) -> LoadResult:
        """Load account data from an existing dataframe."""
        # Delayed import to avoid hard dependency unless required.
        try:
            import pandas as pd  # type: ignore  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ValidationError("pandas is required to load dataframes.") from exc

        loader = DataLoader()
        result = loader.load_accounts_from_dataframe(
            dataframe=dataframe,
            field_map=field_map,
            optional_fields=optional_fields,
        )
        self.accounts = result.accounts
        self.field_map = dict(field_map)
        self._source_dataframe = result.dataframe.copy()
        return result

    # ------------------------------------------------------------- Segmentation
    def set_segmentation(self, method: str) -> None:
        """Configure the segmentation strategy."""
        valid_methods = {"all", "by_account_type", "by_customer_segment", "cross"}
        if method not in valid_methods:
            raise ValueError(f"Unsupported segmentation method: {method}")
        self.segmentation_method = method
        self._assumptions.segmentation_method = method

    # --------------------------------------------------------------- Assumptions
    def set_assumptions(
        self,
        segment_key: str,
        decay_rate: float,
        wal_years: float,
        beta_up: float,
        beta_down: float,
        notes: Optional[str] = None,
    ) -> None:
        """Set the assumptions for a specific segment."""
        assumption = SegmentAssumptions(
            segment_key=segment_key,
            decay_rate=decay_rate,
            wal_years=wal_years,
            beta_up=beta_up,
            beta_down=beta_down,
            notes=notes,
        )
        self._assumptions.add(assumption)

    def assumption_set(self) -> AssumptionSet:
        """Return the current assumption set."""
        return self._assumptions

    # ---------------------------------------------------------- Discount Rates
    def set_discount_single_rate(self, rate: float) -> None:
        """Set a flat discount rate."""
        curve = DiscountCurve.from_single_rate(rate)
        self._discount_config = DiscountConfig(method="single", curve=curve)

    def set_discount_yield_curve(self, tenor_rates: Dict[int, float]) -> None:
        """Set a term structure discount curve."""
        curve = DiscountCurve.from_yield_curve(tenor_rates)
        self._discount_config = DiscountConfig(method="yield_curve", curve=curve)

    def discount_curve(self) -> DiscountCurve:
        """Return the configured discount curve."""
        if not self._discount_config:
            raise ValueError("Discount curve has not been configured.")
        return self._discount_config.curve

    # ------------------------------------------------------------ Market Rates
    def set_base_market_rate_path(
        self, rates: Union[float, Sequence[float]]
    ) -> None:
        """Set the base market rate path used for scenario comparisons."""
        if isinstance(rates, (int, float)):
            self._market_rate_path = [float(rates)] * self.projection_months
        else:
            if len(rates) == 0:
                raise ValueError("Rate path cannot be empty")
            self._market_rate_path = tuple(float(r) for r in rates)

    def market_rate_path(self) -> Sequence[float]:
        """Return the base market rate path."""
        if self._market_rate_path is None:
            raise ValueError("Base market rate path has not been configured.")
        if len(self._market_rate_path) < self.projection_months:
            last_rate = self._market_rate_path[-1]
            extension = [last_rate] * (self.projection_months - len(self._market_rate_path))
            return tuple(self._market_rate_path) + tuple(extension)
        return self._market_rate_path

    # ---------------------------------------------------------------- Scenarios
    def configure_standard_scenarios(self, selections: Dict[str, bool]) -> None:
        """Configure the scenario set using standard selections."""
        self._scenario_set = assemble_standard_scenarios(
            selections=selections,
            projection_months=self.projection_months,
            monte_carlo_config=self._monte_carlo_config,
        )

    def add_scenario(self, scenario: ScenarioDefinition) -> None:
        """Register a custom scenario."""
        self._scenario_set.add(scenario)

    def scenario_set(self) -> ScenarioSet:
        """Return the scenario set, defaulting to base case if empty."""
        if not self._scenario_set.scenarios:
            self.configure_standard_scenarios({"base": True})
        return self._scenario_set

    def set_monte_carlo_config(
        self,
        *,
        num_simulations: int,
        quarterly_volatility: float,
        quarterly_drift: float = 0.0,
        random_seed: Optional[int] = None,
        symmetric_shocks: bool = False,
        pair_opposite_drift: bool = False,
        use_fixed_magnitude: bool = False,
    ) -> None:
        """Configure quarterly-based Monte Carlo simulation parameters."""

        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if quarterly_volatility < 0:
            raise ValueError("quarterly_volatility must be non-negative")

        self._monte_carlo_config = {
            "num_simulations": int(num_simulations),
            "quarterly_volatility": float(quarterly_volatility),
            "quarterly_drift": float(quarterly_drift),
            "random_seed": int(random_seed) if random_seed is not None else None,
            "symmetric_shocks": bool(symmetric_shocks),
            "pair_opposite_drift": bool(pair_opposite_drift),
            "use_fixed_magnitude": bool(use_fixed_magnitude),
        }

    # --------------------------------------------------------------- Execution
    def run_analysis(
        self,
        projection_months: Optional[int] = None,
        progress_callback: Optional[
            callable
        ] = None,
    ) -> EngineResults:
        """Execute the projection engine for all configured scenarios."""
        if projection_months:
            self.projection_months = projection_months
        self._validate_ready_state()

        settings = ProjectionSettings(
            projection_months=self.projection_months,
            segmentation_method=self.segmentation_method,
            base_market_rates=self.market_rate_path(),
        )
        projector = CashflowProjector(self._assumptions)
        discount_curve = self.discount_curve()
        pv_calculator = PresentValueCalculator(discount_curve)

        scenario_plan = self.scenario_set().scenarios
        total_accounts = max(len(self.accounts), 1)

        # Determine total progress steps (per account, per simulation).
        scenario_step_info: List[tuple[ScenarioDefinition, int]] = []
        total_steps = 0
        for scenario in scenario_plan:
            if scenario.scenario_type == ScenarioType.MONTE_CARLO:
                num_sim = max(
                    1, int(scenario.metadata.get("num_simulations", 1) or 1)
                )
                steps = total_accounts * num_sim
            else:
                steps = total_accounts
            scenario_step_info.append((scenario, steps))
            total_steps += steps
        total_steps = max(total_steps, 1)
        total_scenarios = len(scenario_plan)

        current_step = 0

        def emit_progress(step: int, message: str) -> None:
            if progress_callback:
                safe_step = max(0, min(step, total_steps))
                try:
                    progress_callback(safe_step, total_steps, message)
                except Exception:  # pragma: no cover - guard rail
                    pass

        results = EngineResults(base_scenario_id="base")
        emit_progress(0, "Initializing scenario projections...")

        for index, (scenario, scenario_steps) in enumerate(
            scenario_step_info, start=1
        ):
            scenario_label = scenario.description or scenario.scenario_id
            emit_progress(
                current_step,
                f"Scenario {index}/{total_scenarios}: {scenario_label}",
            )

            if scenario.scenario_type == ScenarioType.MONTE_CARLO:
                step_offset = current_step

                def simulation_progress(sim_idx: int, sim_total: int) -> None:
                    progress_value = step_offset + min(
                        scenario_steps, total_accounts * sim_idx
                    )
                    emit_progress(
                        progress_value,
                        f"Scenario {index}/{total_scenarios}: simulation {sim_idx}/{sim_total}",
                    )

                scenario_result = self._run_monte_carlo(
                    scenario=scenario,
                    projector=projector,
                    settings=settings,
                    pv_calculator=pv_calculator,
                    progress_callback=lambda s, _t, m: emit_progress(s, m),
                    step_offset=current_step,
                    total_steps=total_steps,
                    total_accounts=total_accounts,
                    scenario_index=index,
                    total_scenarios=total_scenarios,
                    simulation_callback=simulation_progress,
                )
                current_step += scenario_steps
            else:
                step_offset = current_step

                def account_progress(
                    account_idx: int,
                    account_total: int,
                    account_id: str,
                    scenario_id: str,
                ) -> None:
                    progress_value = step_offset + account_idx
                    emit_progress(
                        progress_value,
                        (
                            f"Scenario {index}/{total_scenarios}: "
                            f"account {account_idx}/{account_total} ({account_id})"
                        ),
                    )

                cashflows = projector.project(
                    self.accounts,
                    scenario,
                    settings,
                    account_progress=account_progress,
                )
                account_pv = pv_calculator.account_level_pv(cashflows)
                portfolio_pv = pv_calculator.portfolio_pv(cashflows)
                scenario_result = ScenarioResult(
                    scenario_id=scenario.scenario_id,
                    cashflows=cashflows,
                    present_value=portfolio_pv,
                    account_level_pv=account_pv,
                    metadata={"method": scenario.scenario_type.value},
                )
                current_step += scenario_steps
            results.add_result(scenario_result)

        if (
            results.base_scenario_id
            and results.base_scenario_id in results.scenario_results
            and self._source_dataframe is not None
        ):
            try:
                base_result = results.scenario_results[results.base_scenario_id]
                discount_rate = float(
                    self.discount_curve().rate_for_month(self.projection_months)
                )
                validation = validate_pv_results(
                    df_results=base_result.account_level_pv,
                    df_original=self._source_dataframe,
                    assumptions=self._assumptions.segments,
                    discount_rate=discount_rate,
                    projection_months=self.projection_months,
                )
                results.validation_summary = validation
            except Exception as exc:  # pragma: no cover - safeguard
                results.validation_summary = {
                    "status": "ERROR",
                    "failed_checks": ["validation_exception"],
                    "warnings": [f"Validation failed: {exc}"],
                }

        return results

    # ----------------------------------------------------------------- Helpers
    def _validate_ready_state(self) -> None:
        """Ensure the engine has sufficient configuration to run."""
        if not self.accounts:
            raise ValidationError("No account data loaded.")
        if not self._assumptions.segments:
            raise ValidationError("No assumptions configured.")
        try:
            self.market_rate_path()
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc
        if not self._discount_config:
            raise ValidationError("Discount rate configuration missing.")
        if self._monte_carlo_config is None:
            # Nothing to validate unless Monte Carlo requested later.
            return

    def _run_monte_carlo(
        self,
        scenario: ScenarioDefinition,
        projector: CashflowProjector,
        settings: ProjectionSettings,
        pv_calculator: PresentValueCalculator,
        *,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        step_offset: int = 0,
        total_steps: int = 1,
        total_accounts: int = 1,
        scenario_index: int = 1,
        total_scenarios: int = 1,
        simulation_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ScenarioResult:
        """Execute Monte Carlo simulations and aggregate results."""
        import pandas as pd

        config = scenario.metadata
        num_sim = int(config.get("num_simulations", 0))
        if num_sim <= 0:
            raise ValidationError("Monte Carlo configuration requires num_simulations > 0")
        if "quarterly_volatility" in config:
            vol_q = float(config.get("quarterly_volatility", 0.0))
            drift_q = float(config.get("quarterly_drift", 0.0))
        else:  # Backwards compatibility with monthly inputs
            vol_q = float(config.get("monthly_volatility", 0.0)) * np.sqrt(3.0)
            drift_q = float(config.get("monthly_drift", 0.0)) * 3.0
        seed = config.get("random_seed")

        symmetric_shocks = bool(config.get("symmetric_shocks", False))
        pair_opposite_drift = bool(config.get("pair_opposite_drift", False))
        use_fixed_magnitude = bool(config.get("use_fixed_magnitude", False))

        rng = np.random.default_rng(seed)
        months = settings.projection_months
        quarter_count = int(np.ceil(months / 3))
        month_index = np.arange(1, months + 1)
        quarter_index = np.arange(1, quarter_count + 1)
        base_path = np.array(settings.base_market_rates[:months], dtype=float)

        principal_sum = np.zeros(months)
        interest_sum = np.zeros(months)
        total_sum = np.zeros(months)
        pv_values = np.zeros(num_sim)
        rate_paths = np.zeros((num_sim, months))

        def build_shocks_matrix() -> np.ndarray:
            if use_fixed_magnitude:
                signs = rng.choice([-1.0, 1.0], size=(num_sim, quarter_count))
                base = np.abs(vol_q) * signs
                if pair_opposite_drift:
                    drift_signs = np.where((np.arange(num_sim) % 2) == 0, 1.0, -1.0).reshape(-1, 1)
                    drift_part = drift_signs * np.abs(drift_q)
                else:
                    drift_part = np.full((num_sim, quarter_count), drift_q)
                return drift_part + base

            if symmetric_shocks:
                half = (num_sim + 1) // 2
                Z = rng.standard_normal(size=(half, quarter_count))
                pos = drift_q + vol_q * Z
                neg = drift_q - vol_q * Z
                mat = np.vstack([pos, neg])[:num_sim]
            else:
                mat = rng.normal(loc=drift_q, scale=vol_q, size=(num_sim, quarter_count))

            if pair_opposite_drift:
                drift_signs = np.where((np.arange(num_sim) % 2) == 0, 1.0, -1.0).reshape(-1, 1)
                mat = (mat - drift_q) + drift_signs * np.abs(drift_q)
            return mat

        quarterly_shocks_matrix = build_shocks_matrix()

        for idx in range(num_sim):
            quarterly_shocks = quarterly_shocks_matrix[idx]
            cumulative_quarters = np.cumsum(quarterly_shocks)
            monthly_levels = np.repeat(cumulative_quarters, 3)[:months]
            shock_vector = {
                int(month): float(shock)
                for month, shock in zip(month_index, monthly_levels)
            }
            sim_scenario = ScenarioDefinition(
                scenario_id=f"{scenario.scenario_id}_sim_{idx + 1}",
                scenario_type=ScenarioType.CUSTOM,
                shock_vector=shock_vector,
                description="Monte Carlo simulation path",
            )
            sim_offset = step_offset + idx * total_accounts
            rate_paths[idx] = np.maximum(0.0, base_path + monthly_levels)

            if progress_callback or simulation_callback:
                message = (
                    f"Scenario {scenario_index}/{total_scenarios}: "
                    f"simulation {idx + 1}/{num_sim}"
                )
                if progress_callback:
                    try:
                        progress_callback(sim_offset, total_steps, message)
                    except Exception:  # pragma: no cover
                        pass
                if simulation_callback:
                    try:
                        simulation_callback(idx + 1, num_sim)
                    except Exception:  # pragma: no cover
                        pass

            cashflows = projector.project(
                self.accounts,
                sim_scenario,
                settings,
            )

            if progress_callback:
                unique_accounts = cashflows["account_id"].unique()
                for account_idx, account_id in enumerate(unique_accounts, start=1):
                    current = sim_offset + account_idx
                    message = (
                        f"Scenario {scenario_index}/{total_scenarios}: "
                        f"simulation {idx + 1}/{num_sim}, "
                        f"account {account_idx}/{len(unique_accounts)} ({account_id})"
                    )
                    try:
                        progress_callback(current, total_steps, message)
                    except Exception:  # pragma: no cover
                        pass

            if progress_callback:
                safe_step = max(0, min(total_steps, sim_offset + total_accounts))
                try:
                    progress_callback(
                        safe_step,
                        total_steps,
                        f"Scenario {scenario_index}/{total_scenarios}: simulation {idx + 1}/{num_sim}",
                    )
                except Exception:  # pragma: no cover
                    pass

            pv_values[idx] = pv_calculator.portfolio_pv(cashflows)
            monthly_totals = (
                cashflows.groupby("month")[["principal", "interest", "total_cash_flow"]]
                .sum()
                .reindex(month_index, fill_value=0.0)
            )
            principal_sum += monthly_totals["principal"].to_numpy()
            interest_sum += monthly_totals["interest"].to_numpy()
            total_sum += monthly_totals["total_cash_flow"].to_numpy()

        expected_cashflows = pd.DataFrame(
            {
                "month": month_index,
                "expected_principal": principal_sum / num_sim,
                "expected_interest": interest_sum / num_sim,
                "expected_total": total_sum / num_sim,
            }
        )

        pv_series = pd.Series(pv_values)
        pv_stats = pd.DataFrame(
            {
                "statistic": ["mean", "std", "p05", "p50", "p95", "min", "max"],
                "value": [
                    float(pv_series.mean()),
                    float(pv_series.std(ddof=1) if num_sim > 1 else 0.0),
                    float(pv_series.quantile(0.05)),
                    float(pv_series.quantile(0.50)),
                    float(pv_series.quantile(0.95)),
                    float(pv_series.min()),
                    float(pv_series.max()),
                ],
            }
        )

        extra_tables = {
            "simulation_pv": pd.DataFrame(
                {"simulation": np.arange(1, num_sim + 1), "present_value": pv_values}
            )
        }

        sample_size = min(num_sim, 500)
        sample_indices = (
            np.linspace(0, num_sim - 1, sample_size).astype(int)
            if sample_size > 0
            else np.array([], dtype=int)
        )
        if sample_size:
            sample_df = pd.DataFrame(
                rate_paths[sample_indices],
                columns=[f"month_{i+1}" for i in range(months)],
            )
            sample_df.insert(0, "simulation", sample_indices + 1)
            extra_tables["rate_paths_sample"] = sample_df

        rate_summary = pd.DataFrame({
            "month": month_index,
            "mean": rate_paths.mean(axis=0),
            "p01": np.percentile(rate_paths, 1, axis=0),
            "p05": np.percentile(rate_paths, 5, axis=0),
            "p10": np.percentile(rate_paths, 10, axis=0),
            "p25": np.percentile(rate_paths, 25, axis=0),
            "p50": np.percentile(rate_paths, 50, axis=0),
            "p75": np.percentile(rate_paths, 75, axis=0),
            "p90": np.percentile(rate_paths, 90, axis=0),
            "p95": np.percentile(rate_paths, 95, axis=0),
            "p99": np.percentile(rate_paths, 99, axis=0),
            "min": rate_paths.min(axis=0),
            "max": rate_paths.max(axis=0),
            "base_rate": base_path,
        })
        extra_tables["rate_paths_summary"] = rate_summary
        pv_percentiles = {
            "mean": float(pv_series.mean()),
            "std": float(pv_series.std(ddof=1) if num_sim > 1 else 0.0),
            "min": float(pv_series.min()),
            "max": float(pv_series.max()),
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

        metadata = {
            "method": scenario.scenario_type.value,
            "num_simulations": num_sim,
            "quarterly_volatility": vol_q,
            "quarterly_drift": drift_q,
            "random_seed": seed,
            "symmetric_shocks": symmetric_shocks,
            "pair_opposite_drift": pair_opposite_drift,
            "use_fixed_magnitude": use_fixed_magnitude,
            "pv_percentiles": pv_percentiles,
        }

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            cashflows=expected_cashflows,
            present_value=float(pv_series.mean()),
            account_level_pv=pv_stats,
            metadata=metadata,
            extra_tables=extra_tables,
        )
