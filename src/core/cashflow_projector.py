"""Monthly cash flow projection engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..models.account import AccountRecord
from ..models.assumptions import AssumptionSet, SegmentAssumptions
from ..models.scenario import ScenarioDefinition


@dataclass
class ProjectionSettings:
    """Settings required to run a projection."""

    projection_months: int = 240
    segmentation_method: str = "all"
    base_market_rates: Sequence[float] = (0.0,)
    materiality_threshold: float = 1_000.0
    max_projection_months: int = 600


class CashflowProjector:
    """Generate detailed account-level cashflows for a scenario."""

    def __init__(self, assumption_set: AssumptionSet) -> None:
        self.assumption_set = assumption_set

    def project(
        self,
        accounts: Iterable[AccountRecord],
        scenario: ScenarioDefinition,
        settings: ProjectionSettings,
        account_progress: Optional[
            "Callable[[int, int, str, str], None]"
        ] = None,
    ) -> pd.DataFrame:
        """Compute monthly cash flows for the supplied accounts."""
        rows: List[dict] = []
        horizon_months = max(settings.projection_months, settings.max_projection_months)
        base_rates = self._extend_rate_path(
            settings.base_market_rates, horizon_months
        )
        account_list = list(accounts)
        total_accounts = len(account_list)
        for idx, account in enumerate(account_list, start=1):
            segment_key = account.key(
                segmentation_key=self._segmentation_key(settings.segmentation_method)
            )
            assumptions = self.assumption_set.get(segment_key)
            rows.extend(
                self._project_account(
                    account=account,
                    assumptions=assumptions,
                    scenario=scenario,
                    projection_months=settings.projection_months,
                    max_projection_months=settings.max_projection_months,
                    materiality_threshold=settings.materiality_threshold,
                    base_rates=base_rates,
                )
            )
            if account_progress:
                account_progress(
                    idx,
                    total_accounts,
                    account.account_id,
                    scenario.scenario_id,
                )
        return pd.DataFrame(rows)

    def _project_account(
        self,
        account: AccountRecord,
        assumptions: SegmentAssumptions,
        scenario: ScenarioDefinition,
        projection_months: int,
        max_projection_months: int,
        materiality_threshold: float,
        base_rates: np.ndarray,
    ) -> List[dict]:
        """Project a single account over the requested horizon."""
        monthly_decay = assumptions.monthly_decay_rate()
        balance = account.balance
        deposit_rate = account.interest_rate
        initial_market_rate = base_rates[0]
        previous_scenario_rate = initial_market_rate
        max_months = max(projection_months, max_projection_months)
        records: List[dict] = []
        for month in range(1, max_months + 1):
            base_rate = base_rates[month - 1]
            shock = scenario.rate_adjustment(month)
            repricing_beta = (
                assumptions.repricing_beta_up if shock >= 0 else assumptions.repricing_beta_down
            )
            market_delta = shock * repricing_beta
            scenario_rate = max(0.0, base_rate + market_delta)
            market_change = scenario_rate - previous_scenario_rate if month > 1 else scenario_rate - initial_market_rate
            previous_deposit_rate = deposit_rate
            deposit_rate = self._adjust_deposit_rate(
                previous_rate=deposit_rate,
                market_rate_change=market_change,
                deposit_beta_up=assumptions.deposit_beta_up,
                deposit_beta_down=assumptions.deposit_beta_down,
            )
            principal_runoff = balance * monthly_decay
            interest = balance * (deposit_rate / 12)
            ending_balance = balance - principal_runoff
            records.append(
                {
                    "account_id": account.account_id,
                    "month": month,
                    "beginning_balance": balance,
                    "ending_balance": ending_balance,
                    "principal": principal_runoff,
                    "interest": interest,
                    "total_cash_flow": principal_runoff + interest,
                    "deposit_rate": deposit_rate,
                    "scenario_rate": scenario_rate,
                    "base_rate": base_rate,
                    "market_rate_change": market_change,
                    "deposit_rate_change": deposit_rate - previous_deposit_rate,
                    "monthly_decay_rate": monthly_decay,
                    "segment": assumptions.segment_key,
                }
            )
            previous_scenario_rate = scenario_rate
            balance = ending_balance
            if balance <= 0 or balance < 1e-6:
                break
            reached_horizon = month >= projection_months
            if reached_horizon and ending_balance <= materiality_threshold:
                break
        return records

    @staticmethod
    def _extend_rate_path(
        rates: Sequence[float], months: int
    ) -> np.ndarray:
        """Extend or truncate a rate path to the target number of months."""
        if len(rates) == 0:
            raise ValueError("Base market rates cannot be empty")
        if len(rates) >= months:
            return np.array(rates[:months], dtype=float)
        last_rate = rates[-1]
        extended = list(rates) + [last_rate] * (months - len(rates))
        return np.array(extended, dtype=float)

    @staticmethod
    def _adjust_deposit_rate(
        *,
        previous_rate: float,
        market_rate_change: float,
        deposit_beta_up: float,
        deposit_beta_down: float,
    ) -> float:
        """Apply deposit beta to the market rate change to derive deposit rate."""
        beta = deposit_beta_up if market_rate_change >= 0 else deposit_beta_down
        adjusted = previous_rate + market_rate_change * beta
        return max(0.0, adjusted)

    @staticmethod
    def _segmentation_key(method: str) -> str:
        """Translate segmentation tag to account attribute key."""
        mapping = {
            "all": None,
            "by_account_type": "account_type",
            "by_customer_segment": "customer_segment",
            "cross": "account_type_customer_segment",
        }
        if method not in mapping:
            raise ValueError(f"Unsupported segmentation method: {method}")
        return mapping[method]
