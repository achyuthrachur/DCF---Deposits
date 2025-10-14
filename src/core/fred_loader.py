"""Utilities for loading Treasury yield curves from the FRED API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import logging

import pandas as pd

try:
    from fredapi import Fred
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError(
        "fredapi is required to fetch yield curves from FRED. Install with 'pip install fredapi'."
    ) from exc

from .yield_curve import YieldCurve

LOGGER = logging.getLogger(__name__)


class FREDYieldCurveLoader:
    """Fetch Treasury yield curve data from the FRED API."""

    SERIES_MAP: Dict[int, str] = {
        3: "DGS3MO",
        6: "DGS6MO",
        12: "DGS1",
        24: "DGS2",
        36: "DGS3",
        60: "DGS5",
        84: "DGS7",
        120: "DGS10",
    }

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("FRED API key must be provided.")
        self.fred = Fred(api_key=api_key)

    def get_current_yield_curve(
        self,
        *,
        interpolation_method: str = "linear",
        min_points: int = 4,
        series_override: Optional[Dict[int, str]] = None,
        lookback_days: int = 90,
        target_date: Optional[datetime | str] = None,
    ) -> YieldCurve:
        """Fetch the latest (or specified date) yield curve data."""
        if target_date is not None:
            return self.get_historical_curve(
                target_date,
                interpolation_method=interpolation_method,
            )

        tenors: list[int] = []
        rates: list[float] = []
        fetch_date: Optional[pd.Timestamp] = None
        mapping = series_override or self.SERIES_MAP
        observation_start = datetime.utcnow() - timedelta(days=lookback_days)

        for tenor_months, series_id in sorted(mapping.items()):
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=observation_start,
                )
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.warning("Failed to fetch series %s: %s", series_id, exc)
                continue

            if data is None or len(data.dropna()) == 0:
                LOGGER.warning("No data returned for series %s", series_id)
                continue

            valid_data = data.dropna()
            rate_value = float(valid_data.iloc[-1]) / 100.0
            fetch_date = valid_data.index[-1]

            tenors.append(int(tenor_months))
            rates.append(rate_value)

        if len(tenors) < min_points:
            raise ValueError("Insufficient data points retrieved from FRED.")

        metadata = {
            "source": "fred",
            "as_of": fetch_date.strftime("%Y-%m-%d") if fetch_date else None,
            "series_ids": [mapping[t] for t in tenors],
        }

        LOGGER.info(
            "Fetched %d tenor points from FRED (as of %s).",
            len(tenors),
            fetch_date.strftime("%Y-%m-%d") if fetch_date else "unknown",
        )

        return YieldCurve(tenors, rates, interpolation_method=interpolation_method, metadata=metadata)

    def get_historical_curve(
        self,
        date: datetime | str,
        *,
        interpolation_method: str = "linear",
        tolerance_days: int = 5,
    ) -> YieldCurve:
        """Fetch the yield curve for a specific historical date."""
        if isinstance(date, str):
            target_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            target_date = date

        tenors: list[int] = []
        rates: list[float] = []

        for tenor_months, series_id in sorted(self.SERIES_MAP.items()):
            start = target_date - timedelta(days=tolerance_days)
            end = target_date + timedelta(days=tolerance_days)
            try:
                data = self.fred.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to fetch series %s: %s", series_id, exc)
                continue

            if data is None or data.dropna().empty:
                LOGGER.warning(
                    "No data for series %s around %s", series_id, target_date.date()
                )
                continue

            valid_data = data.dropna()
            idx = (valid_data.index - target_date).map(abs).argmin()
            rate_value = float(valid_data.iloc[idx]) / 100.0

            tenors.append(int(tenor_months))
            rates.append(rate_value)

        if not tenors:
            raise ValueError(f"No data available for {target_date:%Y-%m-%d}.")

        metadata = {
            "source": "fred",
            "as_of": target_date.strftime("%Y-%m-%d"),
        }

        LOGGER.info(
            "Fetched historical curve (%d points) for %s.",
            len(tenors),
            target_date.strftime("%Y-%m-%d"),
        )
        return YieldCurve(tenors, rates, interpolation_method=interpolation_method, metadata=metadata)

    def get_curve_history(
        self,
        start_date: datetime | str,
        end_date: datetime | str,
        *,
        series_override: Optional[Dict[int, str]] = None,
    ) -> pd.DataFrame:
        """Return a dataframe with historical yields for each tenor."""
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

        mapping = series_override or self.SERIES_MAP
        data_frames: Dict[str, pd.Series] = {}

        for tenor_months, series_id in sorted(mapping.items()):
            try:
                series = self.fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=end_date,
                )
            except Exception as exc:  # pragma: no cover
                LOGGER.warning("Failed to fetch series %s: %s", series_id, exc)
                continue

            if series is None:
                continue
            data_frames[f"{tenor_months}M"] = (series / 100.0).rename(f"{tenor_months}M")

        if not data_frames:
            raise ValueError("No data retrieved for requested period.")

        return pd.DataFrame(data_frames).sort_index()


__all__ = ["FREDYieldCurveLoader"]
