"""Utilities for retrieving Treasury yield curves from the FRED API."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
except ImportError as exc:  # pragma: no cover - import guard
    Fred = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

from ..core.yield_curve import YieldCurve


DEFAULT_SERIES_MAP: Dict[int, str] = {
    3: "DGS3MO",
    6: "DGS6MO",
    12: "DGS1",
    24: "DGS2",
    36: "DGS3",
    60: "DGS5",
    84: "DGS7",
    120: "DGS10",
}


@dataclass
class FREDYieldCurveLoader:
    """
    Fetch Treasury yield curve data from the Federal Reserve (FRED) API.
    """

    api_key: str
    series_map: Dict[int, str] = field(default_factory=lambda: dict(DEFAULT_SERIES_MAP))

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("FRED API key is required.")
        if Fred is None:
            raise ImportError(
                "fredapi is not installed. Install it with `pip install fredapi`."
            ) from _IMPORT_ERROR
        object.__setattr__(self, "_fred_client", Fred(api_key=self.api_key))
        object.__setattr__(self, "_last_fetch_date", None)

    # ----------------------------------------------------------------- Fetching
    def _fetch_latest_rate(self, series_id: str) -> Tuple[Optional[datetime], Optional[float]]:
        """Retrieve the latest available rate for a FRED series."""
        data = self._fred_client.get_series(series_id, limit=10)
        if data.empty:
            return None, None
        valid = data.dropna()
        if valid.empty:
            return None, None
        rate = float(valid.iloc[-1]) / 100.0
        date = valid.index[-1].to_pydatetime()
        return date, rate

    def get_current_yield_curve(self, interpolation_method: str = "linear") -> YieldCurve:
        """Fetch the most recent complete yield curve available from FRED."""
        tenors: List[int] = []
        rates: List[float] = []
        fetch_dates: List[datetime] = []

        print("Fetching current Treasury yield curve from FRED...")

        for tenor, series_id in sorted(self.series_map.items()):
            try:
                date, rate = self._fetch_latest_rate(series_id)
            except Exception as exc:  # pragma: no cover - network guard
                print(f"  Warning: Could not fetch {series_id}: {exc}")
                continue
            if rate is None or date is None:
                continue
            tenors.append(tenor)
            rates.append(rate)
            fetch_dates.append(date)
            print(f"  {tenor:3d} month ({series_id}): {rate * 100:.3f}%")

        if len(tenors) < 2:
            raise ValueError("Could not fetch enough yield curve data from FRED.")

        latest_date = max(fetch_dates) if fetch_dates else datetime.utcnow()
        object.__setattr__(self, "_last_fetch_date", latest_date)

        print(f"\nYield curve fetched successfully (as of {latest_date.strftime('%Y-%m-%d')})")
        return YieldCurve(tenors, np.array(rates), interpolation_method)

    def get_historical_curve(
        self,
        date: datetime | str,
        interpolation_method: str = "linear",
    ) -> YieldCurve:
        """Fetch a historical curve centred around the provided date."""
        if isinstance(date, str):
            target_date = datetime.strptime(date, "%Y-%m-%d")
        else:
            target_date = date

        tenors: List[int] = []
        rates: List[float] = []

        for tenor, series_id in sorted(self.series_map.items()):
            try:
                start = target_date - timedelta(days=5)
                end = target_date + timedelta(days=5)
                data = self._fred_client.get_series(
                    series_id,
                    observation_start=start,
                    observation_end=end,
                )
            except Exception as exc:  # pragma: no cover - network guard
                print(f"  Warning: Could not fetch {series_id}: {exc}")
                continue
            if data.empty:
                continue
            valid = data.dropna()
            if valid.empty:
                continue
            closest_idx = (valid.index - target_date).abs().argmin()
            rate = float(valid.iloc[closest_idx]) / 100.0
            tenors.append(tenor)
            rates.append(rate)
            print(f"  {tenor:3d} month ({series_id}): {rate * 100:.3f}%")

        if len(tenors) < 2:
            raise ValueError(f"Insufficient data to build curve for {target_date:%Y-%m-%d}")

        object.__setattr__(self, "_last_fetch_date", target_date)
        return YieldCurve(tenors, np.array(rates), interpolation_method)

    def get_curve_history(
        self,
        start_date: datetime | str,
        end_date: datetime | str,
    ) -> pd.DataFrame:
        """Retrieve a dataframe of tenor-labelled yield history between two dates."""
        if isinstance(start_date, str):
            start = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start = start_date
        if isinstance(end_date, str):
            end = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end = end_date

        data: Dict[str, pd.Series] = {}
        for tenor, series_id in sorted(self.series_map.items()):
            series = self._fred_client.get_series(
                series_id,
                observation_start=start,
                observation_end=end,
            )
            data[f"{tenor}M"] = series / 100.0
        return pd.DataFrame(data)

    # ---------------------------------------------------------------- Properties
    @property
    def last_fetch_date(self) -> Optional[datetime]:
        """Return the timestamp associated with the most recent curve request."""
        return getattr(self, "_last_fetch_date", None)


# ---------------------------------------------------------------------- Testing
def test_fred_loader(api_key: Optional[str]) -> bool:
    """Quick validation harness for the FRED loader."""
    if not api_key or api_key.strip().lower() == "your_fred_api_key_here":
        print("Skipping FRED test - no API key provided.")
        print("Get a free API key at: https://fred.stlouisfed.org/")
        return False
    loader = FREDYieldCurveLoader(api_key=api_key)
    curve = loader.get_current_yield_curve()
    assert len(curve.tenors) >= 6, "Should fetch at least 6 tenor points"
    assert np.all(curve.rates > 0), "All rates should be positive"
    assert np.all(curve.rates < 0.20), "All rates should be < 20%"
    print("\nFRED loader test passed.")
    print(curve)
    return True
