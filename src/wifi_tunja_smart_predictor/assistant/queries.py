"""Allowlisted, structured analytics over the synthetic observations."""

from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from wifi_tunja_smart_predictor.config import REQUIRED_COLUMNS, TARGET_COLUMN
from wifi_tunja_smart_predictor.exceptions import AssistantQueryError

Metric = Literal["count", "mean", "median", "min", "max", "nunique", "high_share"]
GROUPABLE_COLUMNS = {
    "zone_id",
    "zone_name",
    "zone_type",
    "wifi_id",
    "hour",
    "day_of_week",
    "is_weekend",
    "weather_condition",
    "traffic_level",
    "demand_level",
    "month",
    "year",
    "day_name",
    "time_period",
}
FILTERABLE_COLUMNS = GROUPABLE_COLUMNS | {"date", "timestamp", "date_start", "date_end"}
QUERYABLE_COLUMNS = set(REQUIRED_COLUMNS) - {"timestamp"}


class DatasetQuery(BaseModel):
    """Constrained query spec; no user-provided expressions or Python execution."""

    model_config = ConfigDict(extra="forbid")

    metric: Metric = "count"
    column: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict, max_length=8)
    group_by: list[str] = Field(default_factory=list, max_length=2)

    @field_validator("filters")
    @classmethod
    def validate_filter_values(cls, filters: dict[str, Any]) -> dict[str, Any]:
        allowed_values = {
            "time_period": {"NIGHT", "MORNING", "AFTERNOON", "EVENING"},
            "demand_level": {"LOW", "HIGH"},
            "weather_condition": {"CLEAR", "CLOUDY", "RAIN", "HEAVY_RAIN", "FOG"},
            "traffic_level": {"LOW", "MEDIUM", "HIGH"},
            "zone_type": {
                "DOWNTOWN",
                "UNIVERSITY",
                "COMMERCIAL",
                "PARK",
                "TRANSPORT",
                "RESIDENTIAL",
                "PUBLIC_SERVICE",
                "HEALTHCARE",
                "EDUCATION",
                "TOURISM",
            },
        }
        integer_ranges = {
            "hour": (0, 23),
            "day_of_week": (0, 6),
            "is_weekend": (0, 1),
            "month": (1, 12),
            "year": (2024, 2025),
            "day": (1, 31),
            "week_of_year": (1, 53),
            "is_holiday": (0, 1),
            "is_working_day": (0, 1),
            "is_school_day": (0, 1),
        }
        for column, value in filters.items():
            if isinstance(value, (dict, list, tuple, set)) or value is None:
                raise ValueError(f"Filter '{column}' must be a single scalar value")
            if column in integer_ranges:
                lower, upper = integer_ranges[column]
                if type(value) is not int or not lower <= value <= upper:
                    raise ValueError(
                        f"Filter '{column}' must be an integer from {lower} to {upper}"
                    )
            elif column in allowed_values:
                if not isinstance(value, str) or value not in allowed_values[column]:
                    raise ValueError(f"Filter '{column}' has an unsupported value")
            elif column in {"date", "date_start", "date_end"}:
                if not isinstance(value, str):
                    raise ValueError(f"Filter '{column}' must be an ISO date")
                date.fromisoformat(value)
            elif column == "timestamp":
                if not isinstance(value, str):
                    raise ValueError("Filter 'timestamp' must be an ISO datetime")
                datetime.fromisoformat(value.replace("Z", "+00:00"))
            elif column in {"zone_id", "wifi_id"}:
                if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", value):
                    raise ValueError(f"Filter '{column}' has an invalid identifier")
            elif column in {"zone_name", "day_name"}:
                if not isinstance(value, str) or not value.strip() or len(value) > 100:
                    raise ValueError(f"Filter '{column}' must be a bounded string")
            elif column in {"latitude", "longitude", "altitude_m"}:
                if type(value) not in {int, float} or not math.isfinite(value):
                    raise ValueError(f"Filter '{column}' must be a finite number")
        return filters

    @model_validator(mode="after")
    def validate_allowlists(self) -> DatasetQuery:
        if self.column and self.column not in QUERYABLE_COLUMNS:
            raise ValueError(f"Unsupported dataset column: {self.column}")
        if any(column not in GROUPABLE_COLUMNS for column in self.group_by):
            raise ValueError("Unsupported grouping column")
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("Grouping columns must be unique")
        if any(column not in FILTERABLE_COLUMNS for column in self.filters):
            raise ValueError("Unsupported filter column")
        if {"date_start", "date_end"}.issubset(self.filters):
            if date.fromisoformat(self.filters["date_start"]) > date.fromisoformat(
                self.filters["date_end"]
            ):
                raise ValueError("date_start must not be later than date_end")
        if self.metric in {"mean", "median", "min", "max"} and not self.column:
            raise ValueError(f"metric '{self.metric}' requires a column")
        if self.metric == "high_share" and self.column not in {None, TARGET_COLUMN}:
            raise ValueError("high_share is defined only for demand_level")
        return self


class DatasetQueryEngine:
    """Execute safe aggregate queries against a provided frame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.copy()
        if "timestamp" in self.frame:
            self.frame["timestamp"] = pd.to_datetime(self.frame["timestamp"], errors="coerce")
        if "date" in self.frame:
            self.frame["date"] = pd.to_datetime(self.frame["date"], errors="coerce")

    def execute(self, query: DatasetQuery) -> dict[str, Any] | list[dict[str, Any]]:
        """Apply allowlisted filters and aggregates, returning JSON-friendly data."""
        frame = self._filter(self.frame, query.filters)
        if query.group_by:
            groups: list[dict[str, Any]] = []
            for key, subset in frame.groupby(
                query.group_by, dropna=False, observed=True, sort=True
            ):
                keys = key if isinstance(key, tuple) else (key,)
                result = {column: _json_value(value) for column, value in zip(query.group_by, keys)}
                result.update(self._aggregate(subset, query))
                result["rows"] = int(len(subset))
                groups.append(result)
            return groups[:100]
        return {"rows_after_filters": int(len(frame)), **self._aggregate(frame, query)}

    def _filter(self, frame: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
        selected = frame
        for column, value in filters.items():
            if column == "date_start":
                selected = selected.loc[
                    selected["timestamp"].dt.date >= date.fromisoformat(str(value))
                ]
            elif column == "date_end":
                selected = selected.loc[
                    selected["timestamp"].dt.date <= date.fromisoformat(str(value))
                ]
            elif column == "is_weekend":
                selected = selected.loc[selected[column].eq(int(value))]
            elif column == "date":
                selected = selected.loc[selected[column].dt.date.eq(date.fromisoformat(value))]
            elif column == "timestamp":
                selected = selected.loc[
                    selected[column].eq(pd.Timestamp(value.replace("Z", "+00:00")))
                ]
            elif column in selected:
                selected = selected.loc[selected[column].eq(value)]
            else:
                raise AssistantQueryError(f"Unsupported filter: {column}")
        return selected

    @staticmethod
    def _aggregate(frame: pd.DataFrame, query: DatasetQuery) -> dict[str, Any]:
        if query.metric == "count":
            value = len(frame) if not query.column else int(frame[query.column].count())
        elif query.metric == "nunique":
            if not query.column:
                raise AssistantQueryError("nunique requires a column")
            value = int(frame[query.column].nunique(dropna=True))
        elif query.metric == "high_share":
            value = float(frame[TARGET_COLUMN].eq("HIGH").mean()) if len(frame) else None
        else:
            if query.column is None:
                raise AssistantQueryError(f"metric '{query.metric}' requires a column")
            series = pd.to_numeric(frame[query.column], errors="coerce")
            aggregation = getattr(series, query.metric)
            result = aggregation()
            value = float(result) if pd.notna(result) else None
        return {"metric": query.metric, "column": query.column, "value": _json_value(value)}


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    return value
