"""High-level orchestration for the NMD ALM engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

from .core.cashflow_projector import CashflowProjector, ProjectionSettings
from .core.data_loader import DataLoader, LoadResult
from .core.discount import DiscountCurve
from .core.pv_calculator import PresentValueCalculator
from .core.scenario_generator import assemble_standard_scenarios
from .core.validator import ValidationError
from .models.account import AccountRecord
from .models.assumptions import AssumptionSet, SegmentAssumptions
from .models.results import EngineResults, ScenarioResult
from .models.scenario import ScenarioDefinition, ScenarioSet


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
            selections=selections, projection_months=self.projection_months
        )

    def add_scenario(self, scenario: ScenarioDefinition) -> None:
        """Register a custom scenario."""
        self._scenario_set.add(scenario)

    def scenario_set(self) -> ScenarioSet:
        """Return the scenario set, defaulting to base case if empty."""
        if not self._scenario_set.scenarios:
            self.configure_standard_scenarios({"base": True})
        return self._scenario_set

    # --------------------------------------------------------------- Execution
    def run_analysis(self, projection_months: Optional[int] = None) -> EngineResults:
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

        results = EngineResults(base_scenario_id="base")
        for scenario in self.scenario_set().scenarios:
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
