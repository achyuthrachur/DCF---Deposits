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
        monthly_volatility: float,
        monthly_drift: float = 0.0,
        random_seed: Optional[int] = None,
    ) -> None:
        """Configure parameters for Monte Carlo simulations (volatility in decimal form)."""
        if num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if monthly_volatility < 0:
            raise ValueError("monthly_volatility must be non-negative")
        self._monte_carlo_config = {
            "num_simulations": int(num_simulations),
            "monthly_volatility": float(monthly_volatility),
            "monthly_drift": float(monthly_drift),
            "random_seed": int(random_seed) if random_seed is not None else None,
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
                        (
                            f"Scenario {index}/{total_scenarios}: "
                            f"simulation {sim_idx}/{sim_total}"
                        ),
                    )

                scenario_result = self._run_monte_carlo(
                    scenario=scenario,
                    projector=projector,
                    settings=settings,
                    pv_calculator=pv_calculator,
                    progress_callback=emit_progress,
                    step_offset=current_step,
                    total_steps=total_steps,
                    total_accounts=total_accounts,
                    scenario_index=index,
                    total_scenarios=total_scenarios,
                    simulation_callback=simulation_progress,
                )
                current_step += scenario_steps
                emit_progress(
                    current_step,
                    f"Scenario {index}/{total_scenarios}: {scenario_label} complete",
                )
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
                emit_progress(
                    current_step,
                    f"Scenario {index}/{total_scenarios}: {scenario_label} complete",
                )
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
        vol = float(config.get("monthly_volatility", 0.0))
        drift = float(config.get("monthly_drift", 0.0))
        seed = config.get("random_seed")

        rng = np.random.default_rng(seed)
        months = settings.projection_months
        month_index = np.arange(1, months + 1)

        principal_sum = np.zeros(months)
        interest_sum = np.zeros(months)
        total_sum = np.zeros(months)
        pv_values = np.zeros(num_sim)

        for idx in range(num_sim):
            monthly_shocks = rng.normal(loc=drift, scale=vol, size=months)
            cumulative_shock = np.cumsum(monthly_shocks)
            shock_vector = {int(month): float(shock) for month, shock in zip(month_index, cumulative_shock)}
            sim_scenario = ScenarioDefinition(
                scenario_id=f"{scenario.scenario_id}_sim_{idx + 1}",
                scenario_type=ScenarioType.CUSTOM,
                shock_vector=shock_vector,
                description="Monte Carlo simulation path",
            )
            sim_offset = step_offset + idx * total_accounts

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

            def account_progress(
                account_idx: int,
                account_total: int,
                account_id: str,
                scenario_id: str,
            ) -> None:
                if not progress_callback:
                    return
                current = sim_offset + account_idx
                message = (
                    f"Scenario {scenario_index}/{total_scenarios}: "
                    f"simulation {idx + 1}/{num_sim}, "
                    f"account {account_idx}/{account_total} ({account_id})"
                )
                try:
                    progress_callback(current, total_steps, message)
                except Exception:  # pragma: no cover
                    pass

            cashflows = projector.project(
                self.accounts,
                sim_scenario,
                settings,
                account_progress=account_progress,
            )

            if progress_callback:
                try:
                    progress_callback(
                        min(total_steps, sim_offset + total_accounts),
                        total_steps,
                        (
                            f"Scenario {scenario_index}/{total_scenarios}: "
                            f"simulation {idx + 1}/{num_sim} complete"
                        ),
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
        metadata = {
            "method": scenario.scenario_type.value,
            "num_simulations": num_sim,
            "monthly_volatility": vol,
            "monthly_drift": drift,
            "random_seed": seed,
        }

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            cashflows=expected_cashflows,
            present_value=float(pv_series.mean()),
            account_level_pv=pv_stats,
            metadata=metadata,
            extra_tables=extra_tables,
        )
