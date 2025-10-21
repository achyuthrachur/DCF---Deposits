"""Monthly cash flow projection engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

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

    @dataclass
    class _AggregatedProjection:
        months: np.ndarray
        beginning_balance: np.ndarray
        ending_balance: np.ndarray
        principal: np.ndarray
        interest: np.ndarray
        total_cash_flow: np.ndarray
        deposit_rate_weight: np.ndarray
        balance_weight: np.ndarray
        terminal_balance: np.ndarray
        last_active_index: int = -1

        def accumulate(
            self,
            *,
            month: int,
            beginning_balance: float,
            ending_balance: float,
            principal: float,
            interest: float,
            deposit_rate: float,
            is_terminal: bool,
        ) -> None:
            idx = int(month) - 1
            total = principal + interest

            self.beginning_balance[idx] += beginning_balance
            self.principal[idx] += principal
            self.interest[idx] += interest
            self.total_cash_flow[idx] += total
            self.ending_balance[idx] += ending_balance
            self.balance_weight[idx] += beginning_balance
            self.deposit_rate_weight[idx] += deposit_rate * beginning_balance
            if is_terminal:
                self.terminal_balance[idx] += ending_balance

            if (
                beginning_balance
                or principal
                or interest
                or total
                or ending_balance
                or is_terminal
            ):
                self.last_active_index = max(self.last_active_index, idx)

        def active_length(self) -> int:
            if self.last_active_index < 0:
                return 0
            return self.last_active_index + 1

        def to_dataframe(self, *, drop_trailing: bool = True) -> pd.DataFrame:
            length = self.months.size if not drop_trailing else max(self.active_length(), 0)
            if drop_trailing:
                if length == 0:
                    length = 0
            else:
                length = self.months.size
            months = self.months[:length] if length else np.array([], dtype=int)

            begin = self.beginning_balance[:length]
            principal = self.principal[:length]
            interest = self.interest[:length]
            total = self.total_cash_flow[:length]
            end_bal = self.ending_balance[:length]

            deposit_rate = np.zeros(length, dtype=float)
            decay_rate = np.zeros(length, dtype=float)
            if length:
                balance_weight = self.balance_weight[:length]
                mask = balance_weight > 0
                deposit_rate[mask] = (self.deposit_rate_weight[:length][mask] / balance_weight[mask]).astype(float)
                decay_rate_mask = begin > 0
                decay_rate[decay_rate_mask] = principal[decay_rate_mask] / begin[decay_rate_mask]

            return pd.DataFrame(
                {
                    "month": months,
                    "beginning_balance": begin,
                    "ending_balance": end_bal,
                    "principal": principal,
                    "interest": interest,
                    "total_cash_flow": total,
                    "deposit_rate": deposit_rate,
                    "monthly_decay_rate": decay_rate,
                }
            )

        def portfolio_pv(self, discount_curve) -> float:
            if self.months.size == 0:
                return 0.0
            length = max(self.active_length(), 0)
            if length == 0:
                return 0.0
            months = self.months[:length]
            discount_factors = np.array(
                [discount_curve.discount_factor(int(month)) for month in months],
                dtype=float,
            )
            cashflow_pv = float(np.dot(self.total_cash_flow[:length], discount_factors))
            terminal_pv = float(np.dot(self.terminal_balance[:length], discount_factors))
            return cashflow_pv + terminal_pv

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
            for (
                month,
                beginning_balance,
                ending_balance,
                principal_runoff,
                interest,
                deposit_rate_value,
                scenario_rate,
                base_rate,
                market_change,
                deposit_rate_change,
                monthly_decay_rate,
                _terminal,
            ) in self._account_month_values(
                account=account,
                assumptions=assumptions,
                scenario=scenario,
                projection_months=settings.projection_months,
                max_projection_months=settings.max_projection_months,
                materiality_threshold=settings.materiality_threshold,
                base_rates=base_rates,
            ):
                rows.append(
                    {
                        "account_id": account.account_id,
                        "month": month,
                        "beginning_balance": beginning_balance,
                        "ending_balance": ending_balance,
                        "principal": principal_runoff,
                        "interest": interest,
                        "total_cash_flow": principal_runoff + interest,
                        "deposit_rate": deposit_rate_value,
                        "scenario_rate": scenario_rate,
                        "base_rate": base_rate,
                        "market_rate_change": market_change,
                        "deposit_rate_change": deposit_rate_change,
                        "monthly_decay_rate": monthly_decay_rate,
                        "segment": assumptions.segment_key,
                    }
                )
            if account_progress:
                account_progress(
                    idx,
                    total_accounts,
                    account.account_id,
                    scenario.scenario_id,
                )
        return pd.DataFrame(rows)

    def project_aggregate(
        self,
        accounts: Iterable[AccountRecord],
        scenario: ScenarioDefinition,
        settings: ProjectionSettings,
        account_progress: Optional[
            "Callable[[int, int, str, str], None]"
        ] = None,
    ) -> "_AggregatedProjection":
        horizon_months = max(settings.projection_months, settings.max_projection_months)
        months = np.arange(1, horizon_months + 1, dtype=int)
        aggregation = self._AggregatedProjection(
            months=months,
            beginning_balance=np.zeros(horizon_months, dtype=float),
            ending_balance=np.zeros(horizon_months, dtype=float),
            principal=np.zeros(horizon_months, dtype=float),
            interest=np.zeros(horizon_months, dtype=float),
            total_cash_flow=np.zeros(horizon_months, dtype=float),
            deposit_rate_weight=np.zeros(horizon_months, dtype=float),
            balance_weight=np.zeros(horizon_months, dtype=float),
            terminal_balance=np.zeros(horizon_months, dtype=float),
        )
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
            for (
                month,
                beginning_balance,
                ending_balance,
                principal_runoff,
                interest,
                deposit_rate_value,
                _scenario_rate,
                _base_rate,
                _market_change,
                _deposit_rate_change,
                _monthly_decay_rate,
                terminal,
            ) in self._account_month_values(
                account=account,
                assumptions=assumptions,
                scenario=scenario,
                projection_months=settings.projection_months,
                max_projection_months=settings.max_projection_months,
                materiality_threshold=settings.materiality_threshold,
                base_rates=base_rates,
            ):
                aggregation.accumulate(
                    month=month,
                    beginning_balance=beginning_balance,
                    ending_balance=ending_balance,
                    principal=principal_runoff,
                    interest=interest,
                    deposit_rate=deposit_rate_value,
                    is_terminal=terminal,
                )
            if account_progress:
                account_progress(
                    idx,
                    total_accounts,
                    account.account_id,
                    scenario.scenario_id,
                )
        return aggregation

    def _account_month_values(
        self,
        account: AccountRecord,
        assumptions: SegmentAssumptions,
        scenario: ScenarioDefinition,
        projection_months: int,
        max_projection_months: int,
        materiality_threshold: float,
        base_rates: np.ndarray,
    ) -> Iterator[
        Tuple[
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            bool,
        ]
    ]:
        """Yield per-month projection values along with a terminal flag."""
        monthly_decay = assumptions.monthly_decay_rate()
        balance = account.balance
        deposit_rate = account.interest_rate
        initial_market_rate = base_rates[0]
        previous_scenario_rate = initial_market_rate
        max_months = max(projection_months, max_projection_months)
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
            deposit_rate_change = deposit_rate - previous_deposit_rate
            next_balance = ending_balance
            reached_horizon = month >= projection_months
            terminal = (
                next_balance <= 0
                or next_balance < 1e-6
                or (reached_horizon and ending_balance <= materiality_threshold)
                or month >= max_projection_months
            )
            yield (
                month,
                balance,
                ending_balance,
                principal_runoff,
                interest,
                deposit_rate,
                scenario_rate,
                base_rate,
                market_change,
                deposit_rate_change,
                monthly_decay,
                terminal,
            )
            previous_scenario_rate = scenario_rate
            balance = next_balance
            if terminal:
                break

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
