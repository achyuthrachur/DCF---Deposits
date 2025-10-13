"""High-level orchestration for the NMD ALM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .core.cashflow_projector import CashflowProjector, ProjectionSettings
from .core.data_loader import DataLoader, LoadResult
from .core.discount import DiscountCurve
from .core.pv_calculator import PresentValueCalculator
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
        total_scenarios = len(scenario_plan)

        results = EngineResults(base_scenario_id="base")
        for index, scenario in enumerate(scenario_plan, start=1):
            if progress_callback:
                try:
                    progress_callback(index, total_scenarios, scenario)
                except Exception:  # pragma: no cover - guard rail
                    pass
            if scenario.scenario_type == ScenarioType.MONTE_CARLO:
                scenario_result = self._run_monte_carlo(
                    scenario=scenario,
                    projector=projector,
                    settings=settings,
                    pv_calculator=pv_calculator,
                )
            else:
                cashflows = projector.project(self.accounts, scenario, settings)
                account_pv = pv_calculator.account_level_pv(cashflows)
                portfolio_pv = pv_calculator.portfolio_pv(cashflows)
                scenario_result = ScenarioResult(
                    scenario_id=scenario.scenario_id,
                    cashflows=cashflows,
                    present_value=portfolio_pv,
                    account_level_pv=account_pv,
                    metadata={"method": scenario.scenario_type.value},
                )
            results.add_result(scenario_result)
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
            cashflows = projector.project(self.accounts, sim_scenario, settings)
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
