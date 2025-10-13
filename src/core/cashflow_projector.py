"""Monthly cash flow projection engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd

from ..models.account import AccountRecord
from ..models.assumptions import AssumptionSet, SegmentAssumptions
from ..models.scenario import ScenarioDefinition


@dataclass
class ProjectionSettings:
    """Settings required to run a projection."""

    projection_months: int = 120
    segmentation_method: str = "all"
    base_market_rates: Sequence[float] = (0.0,)


class CashflowProjector:
    """Generate detailed account-level cashflows for a scenario."""

    def __init__(self, assumption_set: AssumptionSet) -> None:
        self.assumption_set = assumption_set

    def project(
        self,
        accounts: Iterable[AccountRecord],
        scenario: ScenarioDefinition,
        settings: ProjectionSettings,
    ) -> pd.DataFrame:
        """Compute monthly cash flows for the supplied accounts."""
        rows: List[dict] = []
        base_rates = self._extend_rate_path(
            settings.base_market_rates, settings.projection_months
        )
        for account in accounts:
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
                    base_rates=base_rates,
                )
            )
        return pd.DataFrame(rows)

    def _project_account(
        self,
        account: AccountRecord,
        assumptions: SegmentAssumptions,
        scenario: ScenarioDefinition,
        projection_months: int,
        base_rates: np.ndarray,
    ) -> List[dict]:
        """Project a single account over the requested horizon."""
        monthly_decay = assumptions.monthly_decay_rate()
        balance = account.balance
        base_deposit_rate = account.interest_rate

        records: List[dict] = []
        for month in range(1, projection_months + 1):
            base_rate = base_rates[month - 1]
            shock = scenario.rate_adjustment(month)
            scenario_rate = max(0.0, base_rate + shock)
            adjusted_rate = self._adjust_deposit_rate(
                base_deposit_rate=base_deposit_rate,
                base_market_rate=base_rate,
                scenario_market_rate=scenario_rate,
                beta_up=assumptions.beta_up,
                beta_down=assumptions.beta_down,
            )
            principal_runoff = balance * monthly_decay
            interest = balance * (adjusted_rate / 12)
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
                    "account_rate": adjusted_rate,
                    "scenario_rate": scenario_rate,
                    "base_rate": base_rate,
                    "segment": assumptions.segment_key,
                }
            )
            balance = ending_balance
            if balance <= 0:
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
        base_deposit_rate: float,
        base_market_rate: float,
        scenario_market_rate: float,
        beta_up: float,
        beta_down: float,
    ) -> float:
        """Apply deposit beta to the market rate change to derive deposit rate."""
        market_delta = scenario_market_rate - base_market_rate
        beta = beta_up if market_delta >= 0 else beta_down
        adjusted = base_deposit_rate + market_delta * beta
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
