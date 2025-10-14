"""Monthly cash flow projection engine."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable, Iterable, List, Optional, Sequence

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
        account_progress: Optional[
            "Callable[[int, int, str, str], None]"
        ] = None,
    ) -> pd.DataFrame:
        """Compute monthly cash flows for the supplied accounts."""
        rows: List[dict] = []
        base_rates = self._extend_rate_path(
            settings.base_market_rates, settings.projection_months
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
        base_rates: np.ndarray,
    ) -> List[dict]:
        """Project a single account over the requested horizon."""
        monthly_decay = assumptions.monthly_decay_rate()
        balance = account.balance
        base_deposit_rate = account.interest_rate
        wal_months = max(1, int(ceil(assumptions.wal_years * 12)))
        months_to_project = min(projection_months, wal_months)

        records: List[dict] = []
        for month in range(1, months_to_project + 1):
            base_rate = base_rates[month - 1]
            shock = scenario.rate_adjustment(month)
            repricing_beta = (
                assumptions.repricing_beta_up if shock >= 0 else assumptions.repricing_beta_down
            )
            market_delta = shock * repricing_beta
            scenario_rate = max(0.0, base_rate + market_delta)
            adjusted_rate = self._adjust_deposit_rate(
                base_deposit_rate=base_deposit_rate,
                market_delta=market_delta,
                deposit_beta_up=assumptions.deposit_beta_up,
                deposit_beta_down=assumptions.deposit_beta_down,
            )
            principal_runoff = balance * monthly_decay
            if month == wal_months or principal_runoff >= balance:
                principal_runoff = balance
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
        market_delta: float,
        deposit_beta_up: float,
        deposit_beta_down: float,
    ) -> float:
        """Apply deposit beta to the market rate change to derive deposit rate."""
        beta = deposit_beta_up if market_delta >= 0 else deposit_beta_down
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
