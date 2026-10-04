"""Allowlisted, structured analytics over the synthetic observations."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

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

    metric: Metric = "count"
    column: str | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    group_by: list[str] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def validate_allowlists(self) -> DatasetQuery:
        if self.column and self.column not in QUERYABLE_COLUMNS:
            raise ValueError(f"Unsupported dataset column: {self.column}")
        if any(column not in GROUPABLE_COLUMNS for column in self.group_by):
            raise ValueError("Unsupported grouping column")
        if any(column not in FILTERABLE_COLUMNS for column in self.filters):
            raise ValueError("Unsupported filter column")
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
