"""High-level orchestration for the NMD ALM engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .core.cashflow_projector import CashflowProjector, ProjectionSettings
from .core.data_loader import DataLoader, LoadResult
from .core.discount import DiscountCurve
from .core.fred_loader import FREDYieldCurveLoader
from .core.monte_carlo import (
    MonteCarloConfig,
    MonteCarloLevel,
    VasicekParams,
    build_percentile_table,
    generate_correlated_vasicek_paths,
    generate_vasicek_path,
    interpolate_curve_rates,
)
from .core.yield_curve import YieldCurve
from .core.monte_carlo_validation import validate_distribution, validate_rate_paths
from .core.pv_calculator import PresentValueCalculator
from .core.pv_validation import validate_pv_results
from .core.scenario_generator import assemble_standard_scenarios
from .core.validator import ValidationError
from .models.account import AccountRecord
from .models.assumptions import AssumptionSet, SegmentAssumptions
from .models.monte_carlo import MonteCarloProgressEvent
from .models.results import EngineResults, ScenarioResult
from .models.scenario import ScenarioDefinition, ScenarioSet, ScenarioType


@dataclass
class DiscountConfig:
    """Captures discount rate configuration details."""

    method: str
    curve: DiscountCurve
    source: str = "manual"
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        self._monte_carlo_config: Optional[MonteCarloConfig] = None
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
        deposit_beta_up: float,
        deposit_beta_down: float,
        repricing_beta_up: float,
        repricing_beta_down: float,
        notes: Optional[str] = None,
    ) -> None:
        """Set the assumptions for a specific segment."""
        assumption = SegmentAssumptions(
            segment_key=segment_key,
            decay_rate=decay_rate,
            wal_years=wal_years,
            deposit_beta_up=deposit_beta_up,
            deposit_beta_down=deposit_beta_down,
            repricing_beta_up=repricing_beta_up,
            repricing_beta_down=repricing_beta_down,
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
        self._discount_config = DiscountConfig(
            method="single",
            curve=curve,
            source="single_rate",
            metadata={"rate": float(rate)},
        )
        self._default_market_rates_if_missing()

    def set_discount_yield_curve(
        self,
        tenor_rates: Dict[int, float],
        *,
        interpolation_method: str = "linear",
        source: str = "manual",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a term structure discount curve."""
        curve = DiscountCurve.from_yield_curve(
            tenor_rates,
            interpolation_method=interpolation_method,
        )
        payload = {
            "tenor_rates": {int(k): float(v) for k, v in tenor_rates.items()},
            "interpolation_method": interpolation_method,
        }
        if metadata:
            payload.update(metadata)
        self._discount_config = DiscountConfig(
            method="yield_curve",
            curve=curve,
            source=source,
            metadata=payload,
        )
        self._default_market_rates_if_missing()

    def set_discount_curve(self, curve: YieldCurve, *, source: str = "manual") -> None:
        """Set discount curve from a pre-built YieldCurve instance."""
        discount_curve = DiscountCurve(curve.tenors, curve.rates, interpolation_method=curve.interpolation_method)
        self._discount_config = DiscountConfig(
            method="yield_curve",
            curve=discount_curve,
            source=source,
            metadata=curve.metadata or {},
        )
        self._default_market_rates_if_missing()

    def set_discount_curve_from_fred(
        self,
        api_key: str,
        *,
        interpolation_method: str = "linear",
        target_date: Optional[Union[str, "datetime"]] = None,
    ) -> YieldCurve:
        """Fetch the latest Treasury curve from FRED and register it."""
        loader = FREDYieldCurveLoader(api_key)
        curve = loader.get_current_yield_curve(
            interpolation_method=interpolation_method,
            target_date=target_date,
        )
        self.set_discount_curve(curve, source="fred")
        return curve

    def _default_market_rates_if_missing(self) -> None:
        """Populate base market rate path from the discount curve if none supplied."""
        if self._market_rate_path is not None or not self._discount_config:
            return
        self._market_rate_path = tuple(
            float(self._discount_config.curve.get_rate(month))
            for month in range(1, self.projection_months + 1)
        )

    def discount_curve(self) -> DiscountCurve:
        """Return the configured discount curve."""
        if not self._discount_config:
            raise ValueError("Discount curve has not been configured.")
        return self._discount_config.curve

    def discount_configuration(self) -> DiscountConfig:
        """Return the discount configuration details."""
        if not self._discount_config:
            raise ValueError("Discount curve has not been configured.")
        return self._discount_config

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
        base_curve = self._discount_config.curve if self._discount_config else None
        self._scenario_set = assemble_standard_scenarios(
            selections=selections,
            projection_months=self.projection_months,
            monte_carlo_config=self._monte_carlo_config,
            base_discount_curve=base_curve,
        )

    def add_scenario(self, scenario: ScenarioDefinition) -> None:
        """Register a custom scenario."""
        self._scenario_set.add(scenario)

    def scenario_set(self) -> ScenarioSet:
        """Return the scenario set, defaulting to base case if empty."""
        if not self._scenario_set.scenarios:
            self.configure_standard_scenarios({"base": True})
        return self._scenario_set

    def set_monte_carlo_config(self, config: MonteCarloConfig) -> None:
        """Register Monte Carlo configuration (levels 1 & 2)."""

        if config.num_simulations <= 0:
            raise ValueError("num_simulations must be positive")
        if config.level == MonteCarloLevel.TWO_FACTOR and config.long_rate is None:
            raise ValueError("Two-factor Monte Carlo requires long-rate parameters.")
        if config.level not in {MonteCarloLevel.STATIC_CURVE, MonteCarloLevel.TWO_FACTOR}:
            raise ValueError(f"Unsupported Monte Carlo level: {config.level}")
        config.projection_months = self.projection_months
        self._monte_carlo_config = config

    # --------------------------------------------------------------- Execution
    def run_analysis(
        self,
        projection_months: Optional[int] = None,
        progress_callback: Optional[
            Callable[[int, int, str], None]
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
        *,
        pv_calculator: Optional[PresentValueCalculator] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        step_offset: int = 0,
        total_steps: int = 1,
        total_accounts: int = 1,
        scenario_index: int = 1,
        total_scenarios: int = 1,
        simulation_callback: Optional[Callable[[int, int], None]] = None,
        progress_observer: Optional[Callable[[MonteCarloProgressEvent], None]] = None,
    ) -> ScenarioResult:
        """Execute Monte Carlo simulations (levels 1 & 2) and aggregate results."""

        config = MonteCarloConfig.from_metadata(dict(scenario.metadata or {}))
        months = min(config.projection_months, settings.projection_months)
        if months <= 0:
            raise ValidationError("Monte Carlo projection horizon must be positive.")

        base_curve = self.discount_curve()
        base_market_rates = np.array(settings.base_market_rates[:months], dtype=float)
        if base_market_rates.size < months:
            last_rate = float(base_market_rates[-1])
            extension = np.full(months - base_market_rates.size, last_rate, dtype=float)
            base_market_rates = np.concatenate([base_market_rates, extension])
        months_range = np.arange(1, months + 1)

        short_params = config.short_rate
        short_initial = (
            short_params.initial_rate
            if short_params.initial_rate is not None
            else float(base_market_rates[0])
        )
        short_params = short_params.with_initial(short_initial)

        long_params = None
        if config.level == MonteCarloLevel.TWO_FACTOR:
            if config.long_rate is None:
                raise ValidationError(
                    "Two-factor Monte Carlo requires long-rate parameters."
                )
            long_initial = (
                config.long_rate.initial_rate
                if config.long_rate.initial_rate is not None
                else float(base_curve.rate_for_month(120))
            )
            long_params = config.long_rate.with_initial(long_initial)

        rng = np.random.default_rng(config.random_seed)
        mc_settings = ProjectionSettings(
            projection_months=months,
            segmentation_method=settings.segmentation_method,
            base_market_rates=tuple(base_market_rates.tolist()),
        )

        pv_values = np.zeros(config.num_simulations, dtype=float)
        principal_sum = np.zeros(months, dtype=float)
        interest_sum = np.zeros(months, dtype=float)
        total_sum = np.zeros(months, dtype=float)

        progress_records: List[Dict[str, float]] = []
        sample_indices = (
            np.linspace(
                0,
                config.num_simulations - 1,
                min(config.sample_size, config.num_simulations),
            ).astype(int)
            if config.save_rate_paths and config.sample_size > 0
            else np.array([], dtype=int)
        )
        sample_set = set(sample_indices.tolist())
        rate_sample_rows: List[Dict[str, float]] = []

        validation_short_paths: List[np.ndarray] = []
        validation_long_paths: List[np.ndarray] = []
        validation_sample_limit = min(200, config.num_simulations)
        all_short_paths: List[np.ndarray] = []
        all_long_paths: List[np.ndarray] = []

        for sim_index in range(config.num_simulations):
            message = (
                f"Scenario {scenario_index}/{total_scenarios}: "
                f"simulation {sim_index + 1}/{config.num_simulations}"
            )
            progress_value = min(
                step_offset + sim_index * total_accounts, total_steps
            )
            if progress_callback:
                try:
                    progress_callback(progress_value, total_steps, message)
                except Exception:  # pragma: no cover
                    pass
            if simulation_callback:
                try:
                    simulation_callback(sim_index + 1, config.num_simulations)
                except Exception:  # pragma: no cover
                    pass

            if config.level == MonteCarloLevel.STATIC_CURVE:
                short_path = generate_vasicek_path(short_params, months, rng)
                long_path = None
                monthly_discount_rates = [
                    float(base_curve.rate_for_month(month)) for month in months_range
                ]
                discount_curve = base_curve
            else:
                assert long_params is not None
                short_path, long_path = generate_correlated_vasicek_paths(
                    short_params, long_params, config.correlation, months, rng
                )
                monthly_discount_rates: List[float] = []
                for month_idx in range(months):
                    tenor_rates = interpolate_curve_rates(
                        short_rate=float(short_path[month_idx]),
                        long_rate=float(long_path[month_idx]),
                        base_curve=base_curve,
                        short_tenor=3,
                        long_tenor=120,
                    )
                    yc = YieldCurve(
                        list(tenor_rates.keys()),
                        list(tenor_rates.values()),
                        interpolation_method=config.interpolation_method,
                    )
                    monthly_discount_rates.append(float(yc.get_rate(month_idx + 1)))
                    if config.save_rate_paths and sim_index in sample_set:
                        rate_sample_rows.append(
                            {
                                "simulation": sim_index + 1,
                                "month": month_idx + 1,
                                "short_rate": float(short_path[month_idx]),
                                "long_rate": float(long_path[month_idx]),
                                "curve_3m": float(yc.get_rate(3)),
                                "curve_12m": float(yc.get_rate(12)),
                                "curve_60m": float(yc.get_rate(60)),
                                "curve_120m": float(yc.get_rate(120)),
                            }
                        )

                discount_curve = DiscountCurve.from_yield_curve(
                    {
                        int(month): float(rate)
                        for month, rate in zip(months_range, monthly_discount_rates)
                    },
                    interpolation_method=config.interpolation_method,
                )

            if (
                config.save_rate_paths
                and config.level == MonteCarloLevel.STATIC_CURVE
                and sim_index in sample_set
            ):
                for month_idx in range(months):
                    rate_sample_rows.append(
                        {
                            "simulation": sim_index + 1,
                            "month": month_idx + 1,
                            "short_rate": float(short_path[month_idx]),
                            "discount_rate": float(
                                monthly_discount_rates[month_idx]
                            ),
                        }
                    )

            all_short_paths.append(short_path.copy())
            if config.level == MonteCarloLevel.TWO_FACTOR and long_path is not None:
                all_long_paths.append(long_path.copy())

            if len(validation_short_paths) < validation_sample_limit:
                validation_short_paths.append(short_path.copy())
                if (
                    config.level == MonteCarloLevel.TWO_FACTOR
                    and long_path is not None
                ):
                    validation_long_paths.append(long_path.copy())

            shock_vector = {
                int(month): float(short_path[month - 1] - base_market_rates[month - 1])
                for month in months_range
            }
            sim_scenario = ScenarioDefinition(
                scenario_id=f"{scenario.scenario_id}_sim_{sim_index + 1}",
                scenario_type=ScenarioType.CUSTOM,
                shock_vector=shock_vector,
                description="Monte Carlo simulation path",
            )

            cashflows = projector.project(
                self.accounts,
                sim_scenario,
                mc_settings,
            )

            pv_calculator = PresentValueCalculator(discount_curve)
            pv_value = pv_calculator.portfolio_pv(cashflows)
            pv_values[sim_index] = pv_value

            monthly_totals = (
                cashflows.groupby("month")[["principal", "interest", "total_cash_flow"]]
                .sum()
                .reindex(months_range, fill_value=0.0)
            )
            principal_sum += monthly_totals["principal"].to_numpy()
            interest_sum += monthly_totals["interest"].to_numpy()
            total_sum += monthly_totals["total_cash_flow"].to_numpy()

            cumulative_series = pd.Series(pv_values[: sim_index + 1])
            cumulative_mean = float(cumulative_series.mean())
            cumulative_std = (
                float(cumulative_series.std(ddof=1)) if sim_index > 0 else 0.0
            )
            percentile_snapshot = {
                "p05": float(cumulative_series.quantile(0.05)),
                "p50": float(cumulative_series.quantile(0.50)),
                "p95": float(cumulative_series.quantile(0.95)),
            }
            progress_records.append(
                {
                    "simulation": sim_index + 1,
                    "present_value": float(pv_value),
                    "cumulative_mean": cumulative_mean,
                    "cumulative_std": cumulative_std,
                    **percentile_snapshot,
                }
            )
            if progress_observer:
                try:
                    progress_observer(
                        MonteCarloProgressEvent(
                            scenario_id=scenario.scenario_id,
                            simulation_index=sim_index + 1,
                            total_simulations=config.num_simulations,
                            present_value=float(pv_value),
                            cumulative_mean=cumulative_mean,
                            cumulative_std=cumulative_std,
                            percentile_estimates=percentile_snapshot,
                            rate_path=short_path.tolist(),
                        )
                    )
                except Exception:  # pragma: no cover
                    pass

            if progress_callback:
                completed_step = min(
                    step_offset + (sim_index + 1) * total_accounts, total_steps
                )
                try:
                    progress_callback(
                        completed_step,
                        total_steps,
                        (
                            f"Scenario {scenario_index}/{total_scenarios}: "
                            f"simulation {sim_index + 1}/{config.num_simulations}"
                        ),
                    )
                except Exception:  # pragma: no cover
                    pass

        expected_cashflows = pd.DataFrame(
            {
                "month": months_range,
                "expected_principal": principal_sum / config.num_simulations,
                "expected_interest": interest_sum / config.num_simulations,
                "expected_total": total_sum / config.num_simulations,
            }
        )

        rate_summary_df: Optional[pd.DataFrame] = None
        if all_short_paths:
            stacked_short = np.vstack(all_short_paths)
            quantile_levels = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
            quantile_labels = [
                "p01",
                "p05",
                "p10",
                "p25",
                "p50",
                "p75",
                "p90",
                "p95",
                "p99",
            ]
            short_quantiles = np.quantile(stacked_short, quantile_levels, axis=0)
            summary_payload: Dict[str, np.ndarray] = {
                "month": months_range,
                "base_rate": base_market_rates,
                "mean": stacked_short.mean(axis=0),
            }
            for label, series_values in zip(quantile_labels, short_quantiles):
                summary_payload[label] = series_values

            if all_long_paths:
                stacked_long = np.vstack(all_long_paths)
                long_quantiles = np.quantile(stacked_long, quantile_levels, axis=0)
                summary_payload["long_mean"] = stacked_long.mean(axis=0)
                for label, series_values in zip(quantile_labels, long_quantiles):
                    summary_payload[f"long_{label}"] = series_values

            rate_summary_df = pd.DataFrame(summary_payload)

        pv_series = pd.Series(pv_values)
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
        mean_pv = float(pv_series.mean())
        std_pv = float(pv_series.std(ddof=1)) if config.num_simulations > 1 else 0.0
        skew_pv = float(pv_series.skew())
        kurtosis_pv = float(pv_series.kurt())

        book_value = None
        if (
            self._source_dataframe is not None
            and "balance" in self._source_dataframe.columns
        ):
            book_value = float(self._source_dataframe["balance"].sum())

        var_95 = book_value - percentiles["p5"] if book_value is not None else None
        cvar_mask = pv_series <= percentiles["p5"]
        cvar_95 = float(pv_series[cvar_mask].mean()) if cvar_mask.any() else percentiles["p5"]
        prob_below_book = (
            float((pv_series < book_value).mean()) if book_value is not None else None
        )

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
            "std": std_pv,
            "skew": skew_pv,
            "kurtosis": kurtosis_pv,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "prob_below_book": prob_below_book,
        }
        summary_df = pd.DataFrame([summary_row])

        distribution_df = pd.DataFrame(
            {
                "simulation": np.arange(1, config.num_simulations + 1),
                "portfolio_pv": pv_values,
            }
        )
        percentiles_df = build_percentile_table(pv_values, percentiles=range(5, 100, 5))

        rate_validation = validate_rate_paths(
            validation_short_paths,
            long_paths=validation_long_paths if validation_long_paths else None,
            long_term_mean=config.short_rate.long_term_mean,
        )
        dist_validation = validate_distribution(pv_values, book_value=book_value)

        summary_metadata = {
            "method": scenario.scenario_type.value,
            "level": int(config.level),
            "config": config.to_metadata(),
            "summary": summary_row,
            "pv_percentiles": percentiles,
            "book_value": book_value,
        }

        extra_tables: Dict[str, pd.DataFrame] = {
            "simulation_pv": distribution_df,
            "summary_table": summary_df,
            "percentiles_table": percentiles_df,
            "monte_carlo_distribution": distribution_df,
            "monte_carlo_summary": summary_df,
            "monte_carlo_percentiles": percentiles_df,
        }
        if rate_sample_rows:
            extra_tables["rate_paths_sample"] = pd.DataFrame(rate_sample_rows)
        if rate_summary_df is not None:
            extra_tables["rate_paths_summary"] = rate_summary_df
        if progress_records:
            extra_tables["simulation_progress"] = pd.DataFrame(progress_records)

        extra_tables["validation"] = pd.DataFrame(
            [
                {
                    "rates_status": rate_validation.status,
                    "distribution_status": dist_validation.status,
                    "rate_warnings": "|".join(rate_validation.warnings),
                    "distribution_warnings": "|".join(dist_validation.warnings),
                }
            ]
        )

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            cashflows=expected_cashflows,
            present_value=mean_pv,
            account_level_pv=summary_df,
            metadata=summary_metadata,
            extra_tables=extra_tables,
        )
