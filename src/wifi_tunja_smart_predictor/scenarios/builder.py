"""Location/date/time to model-input feature construction using historical analogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from wifi_tunja_smart_predictor.config import MODEL_INPUT_COLUMNS
from wifi_tunja_smart_predictor.exceptions import ScenarioBuildError
from wifi_tunja_smart_predictor.geospatial.locations import LocationResolver, ResolvedLocation


@dataclass(frozen=True)
class ScenarioRequest:
    """User-level scenario inputs; exactly one or more location selectors are needed."""

    datetime: datetime
    zone: str | None = None
    zone_id: str | None = None
    access_point_id: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    context_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioContext:
    """Resolved scenario, constructed model features, and historical analog metadata."""

    scenario_mode: str
    scenario_time: datetime
    location: ResolvedLocation
    features: dict[str, Any]
    analog_count: int
    analog_strategy: str
    analog_period_start: str
    analog_period_end: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize scenario context without exposing target columns."""
        return {
            "scenario_mode": self.scenario_mode,
            "scenario_time": self.scenario_time.isoformat(),
            "location": self.location.to_dict(),
            "analog_count": self.analog_count,
            "analog_strategy": self.analog_strategy,
            "analog_period": {"start": self.analog_period_start, "end": self.analog_period_end},
        }


class HistoricalAnalogEngine:
    """Select repeatable prior observations using ordered calendar/location fallbacks."""

    MIN_ANALOGS = 3

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()
        if "timestamp" in self._frame:
            self._frame["timestamp"] = pd.to_datetime(self._frame["timestamp"], errors="raise")

    def mode_for(self, when: datetime) -> str:
        """Classify requested time as a historical replay or synthetic scenario."""
        timestamp = pd.Timestamp(when)
        return (
            "HISTORICAL_REPLAY"
            if self._frame["timestamp"].min() <= timestamp <= self._frame["timestamp"].max()
            else "SYNTHETIC_SCENARIO"
        )

    def select(self, location: ResolvedLocation, when: datetime) -> tuple[pd.DataFrame, str]:
        """Return a non-empty analog group; historical replay never sees later rows."""
        historical = self._frame.loc[self._frame["timestamp"] < pd.Timestamp(when)]
        mode = self.mode_for(when)
        pool = historical if mode == "HISTORICAL_REPLAY" else self._frame
        if pool.empty:
            raise ScenarioBuildError(
                "No earlier synthetic observations exist for this historical replay."
            )

        hour = when.hour
        weekday = when.weekday()
        month = when.month
        tests = [
            ("access_point_hour_weekday_month", ("wifi_id", "hour", "day_of_week", "month")),
            ("access_point_hour_weekday", ("wifi_id", "hour", "day_of_week")),
            ("zone_type_hour_weekday_month", ("zone_type", "hour", "day_of_week", "month")),
            ("zone_type_hour_weekday", ("zone_type", "hour", "day_of_week")),
            ("zone_type_hour", ("zone_type", "hour")),
            ("hour_weekday", ("hour", "day_of_week")),
            ("hour", ("hour",)),
        ]
        expected = {
            "wifi_id": location.access_point_id,
            "zone_type": location.zone_type,
            "hour": hour,
            "day_of_week": weekday,
            "month": month,
        }
        for name, columns in tests:
            mask = pd.Series(True, index=pool.index)
            for column in columns:
                mask &= pool[column].eq(expected[column])
            candidates = pool.loc[mask]
            if len(candidates) >= self.MIN_ANALOGS:
                return candidates, name
        # Sparse AP/month combinations still resolve reproducibly to all matching hours.
        candidates = pool.loc[pool["hour"].eq(hour)]
        if candidates.empty:
            raise ScenarioBuildError("No historical synthetic rows can support this scenario.")
        return candidates, "hour_fallback"


class ScenarioBuilder:
    """Build a complete preprocessed-input row from location, time, and analog context."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.copy()
        self.resolver = LocationResolver(frame)
        self.analogs = HistoricalAnalogEngine(frame)

    def build(self, request: ScenarioRequest) -> ScenarioContext:
        """Resolve user inputs and aggregate a deterministic analog feature record."""
        when = request.datetime
        if when.tzinfo is not None:
            when = when.astimezone(ZoneInfo("America/Bogota")).replace(tzinfo=None)
        location = self.resolver.resolve(
            zone=request.zone,
            zone_id=request.zone_id,
            access_point_id=request.access_point_id,
            latitude=request.latitude,
            longitude=request.longitude,
        )
        candidates, strategy = self.analogs.select(location, when)
        analog_features = candidates[MODEL_INPUT_COLUMNS]
        features: dict[str, Any] = {}
        categorical = {
            column
            for column, dtype in analog_features.dtypes.items()
            if pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or str(dtype).startswith("string")
        }
        for column in MODEL_INPUT_COLUMNS:
            values = analog_features[column].dropna()
            if column in categorical:
                modes = values.astype(str).mode()
                features[column] = str(modes.sort_values().iloc[0]) if len(modes) else ""
            else:
                features[column] = float(pd.to_numeric(values, errors="coerce").median())

        is_weekend = int(when.weekday() >= 5)
        features.update(
            {
                "zone_type": location.zone_type,
                "month": when.month,
                "day_of_week": when.weekday(),
                "hour": when.hour,
                "is_weekend": is_weekend,
                "is_working_day": int(not is_weekend),
                "time_period": _time_period(when.hour),
                "access_point_capacity": location.access_point_capacity,
            }
        )
        invalid_overrides = set(request.context_overrides) - set(MODEL_INPUT_COLUMNS)
        if invalid_overrides:
            raise ScenarioBuildError(
                f"Unsupported scenario context fields: {sorted(invalid_overrides)}"
            )
        features.update(request.context_overrides)
        # Calendar flags and time-of-day are deterministic; simulated holiday/context values
        # come only from matched historical analogs, never from either target column.
        return ScenarioContext(
            scenario_mode=self.analogs.mode_for(when),
            scenario_time=when,
            location=location,
            features=features,
            analog_count=len(candidates),
            analog_strategy=strategy,
            analog_period_start=str(candidates["timestamp"].min()),
            analog_period_end=str(candidates["timestamp"].max()),
        )


def _time_period(hour: int) -> str:
    """Map hour to the dataset's four time-of-day buckets."""
    if hour < 6 or hour >= 20:
        return "NIGHT"
    if hour < 12:
        return "MORNING"
    if hour < 18:
        return "AFTERNOON"
    return "EVENING"
