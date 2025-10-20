"""Utilities for loading Treasury yield curves from the FRED API."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

import logging

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
    _curve_cache: Dict[Tuple[object, ...], Tuple[datetime, YieldCurve]] = {}
    _cache_ttl = timedelta(minutes=30)
    _OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("FRED API key must be provided.")
        self.fred = Fred(api_key=api_key)
        self.api_key = api_key
        self._session = requests.Session()
        self._session.params = {"api_key": api_key, "file_type": "json"}
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        adapter = HTTPAdapter(max_retries=retries)
        self._session.mount("https://", adapter)
        self._session.mount("http://", adapter)
        self._request_timeout: Tuple[float, float] = (3.0, 15.0)

    @classmethod
    def _make_cache_key(
        cls,
        *,
        api_key: str,
        interpolation_method: str,
        min_points: int,
        series_mapping: Dict[int, str],
        lookback_days: int,
        target_date: Optional[str],
    ) -> Tuple[object, ...]:
        return (
            api_key,
            interpolation_method.lower(),
            int(min_points),
            int(lookback_days),
            tuple(sorted(series_mapping.items())),
            target_date,
        )

    @classmethod
    def _get_cached_curve(cls, key: Tuple[object, ...]) -> Optional[YieldCurve]:
        cached = cls._curve_cache.get(key)
        if not cached:
            return None
        cached_at, curve = cached
        if datetime.utcnow() - cached_at > cls._cache_ttl:
            cls._curve_cache.pop(key, None)
            return None
        return curve

    @classmethod
    def _store_cached_curve(cls, key: Tuple[object, ...], curve: YieldCurve) -> None:
        cls._curve_cache[key] = (datetime.utcnow(), curve)

    def _fetch_latest_observation(
        self, series_id: str, observation_start: datetime
    ) -> Tuple[float, pd.Timestamp]:
        params = {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": 1,
            "observation_start": observation_start.strftime("%Y-%m-%d"),
        }
        response = self._session.get(
            self._OBSERVATIONS_URL, params=params, timeout=self._request_timeout
        )
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations", [])
        for observation in observations:
            value = observation.get("value")
            if value in (None, "", "."):
                continue
            rate = float(value) / 100.0
            obs_date = pd.to_datetime(observation.get("date"))
            return rate, obs_date
        raise ValueError(f"No valid observation returned for series {series_id}.")

    def _fetch_latest_with_fred(
        self, series_id: str, observation_start: datetime
    ) -> Tuple[float, pd.Timestamp]:
        data = self.fred.get_series(series_id, observation_start=observation_start)
        if data is None or data.dropna().empty:
            raise ValueError(f"No data returned for series {series_id}")
        valid_data = data.dropna()
        return float(valid_data.iloc[-1]) / 100.0, pd.to_datetime(valid_data.index[-1])

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
            mapping = series_override or self.SERIES_MAP
            cache_key = self._make_cache_key(
                api_key=self.api_key,
                interpolation_method=interpolation_method,
                min_points=min_points,
                series_mapping=mapping,
                lookback_days=lookback_days,
                target_date=str(target_date),
            )
            cached_curve = self._get_cached_curve(cache_key)
            if cached_curve is not None:
                return cached_curve
            curve = self.get_historical_curve(
                target_date,
                interpolation_method=interpolation_method,
            )
            self._store_cached_curve(cache_key, curve)
            return curve

        tenors: list[int] = []
        rates: list[float] = []
        fetch_date: Optional[pd.Timestamp] = None
        mapping = series_override or self.SERIES_MAP
        observation_start = datetime.utcnow() - timedelta(days=lookback_days)

        cache_key = self._make_cache_key(
            api_key=self.api_key,
            interpolation_method=interpolation_method,
            min_points=min_points,
            series_mapping=mapping,
            lookback_days=lookback_days,
            target_date=None,
        )
        cached_curve = self._get_cached_curve(cache_key)
        if cached_curve is not None:
            return cached_curve

        for tenor_months, series_id in sorted(mapping.items()):
            try:
                rate_value, obs_date = self._fetch_latest_observation(
                    series_id, observation_start
                )
            except Exception as exc:  # pragma: no cover - network failure
                LOGGER.warning(
                    "Fast fetch failed for series %s (%s). Falling back to fredapi.",
                    series_id,
                    exc,
                )
                try:
                    rate_value, obs_date = self._fetch_latest_with_fred(
                        series_id, observation_start
                    )
                except Exception as fallback_exc:
                    LOGGER.warning(
                        "Unable to fetch series %s: %s", series_id, fallback_exc
                    )
                    continue

            fetch_date = obs_date
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

        curve = YieldCurve(
            tenors, rates, interpolation_method=interpolation_method, metadata=metadata
        )

        self._store_cached_curve(cache_key, curve)

        return curve

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
