"""Validated assistant request, context, and response contracts."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime as DateTime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DASHBOARD_SECTIONS = {
    "overview",
    "live scenario",
    "scenario prediction",
    "scenario",
    "demand explorer",
    "demand analysis",
    "geographic analysis",
    "geographic",
    "geography",
    "network analysis",
    "network",
    "model performance",
    "performance",
    "ai assistant",
    "assistant",
    "advanced prediction",
    "advanced",
    "about",
}
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ,()./%:+_-]{0,99}$")
_SCENARIO_KEYS = {
    "scenario",
    "classification",
    "regression",
    "capacity",
    "explanation",
    "disclaimer",
}


class AssistantContext(BaseModel):
    """Optional location/time context supplied by a dashboard client."""

    model_config = ConfigDict(extra="forbid")

    zone: str | None = Field(default=None, min_length=1, max_length=100)
    zone_id: str | None = Field(default=None, min_length=1, max_length=40)
    access_point_id: str | None = Field(default=None, min_length=1, max_length=40)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    datetime: DateTime | None = None
    dashboard_section: str | None = Field(default=None, max_length=80)
    scenario_result: dict[str, Any] | None = None

    @field_validator("zone_id", "access_point_id")
    @classmethod
    def validate_identifiers(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_ID.fullmatch(value):
            raise ValueError("identifier contains unsupported characters")
        return value

    @field_validator("dashboard_section")
    @classmethod
    def validate_dashboard_section(cls, value: str | None) -> str | None:
        if value is not None and " ".join(value.casefold().split()) not in _DASHBOARD_SECTIONS:
            raise ValueError("dashboard_section must name a supported dashboard section")
        return value

    @field_validator("scenario_result")
    @classmethod
    def validate_scenario_result(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        if not set(value).issubset(_SCENARIO_KEYS) or not {
            "scenario",
            "classification",
            "regression",
        }.issubset(value):
            raise ValueError("scenario_result does not match the supported result structure")
        if any(
            not isinstance(value.get(key), dict)
            for key in ("scenario", "classification", "regression")
        ):
            raise ValueError("scenario_result sections must be objects")
        if any(
            key in value and not isinstance(value[key], dict) for key in ("capacity", "explanation")
        ) or ("disclaimer" in value and not isinstance(value["disclaimer"], str)):
            raise ValueError("scenario_result optional sections have invalid types")
        count = [0]

        def check_tree(item: Any, depth: int = 0) -> None:
            count[0] += 1
            if count[0] > 512 or depth > 6:
                raise ValueError("scenario_result exceeds supported size or nesting")
            if isinstance(item, dict):
                for key, child in item.items():
                    if not isinstance(key, str) or len(key) > 80:
                        raise ValueError("scenario_result contains an invalid field name")
                    check_tree(child, depth + 1)
            elif isinstance(item, list):
                if len(item) > 50:
                    raise ValueError("scenario_result list is too large")
                for child in item:
                    check_tree(child, depth + 1)
            elif isinstance(item, str):
                if len(item) > 256:
                    raise ValueError("scenario_result text field is too long")
            elif isinstance(item, float):
                if not math.isfinite(item) or abs(item) > 1_000_000_000:
                    raise ValueError("scenario_result number is outside supported bounds")
            elif type(item) is int:
                if abs(item) > 1_000_000_000:
                    raise ValueError("scenario_result number is outside supported bounds")
            elif item is not None and type(item) not in {int, bool}:
                raise ValueError("scenario_result contains an unsupported value type")

        check_tree(value)
        if len(json.dumps(value, ensure_ascii=False)) > 32_768:
            raise ValueError("scenario_result is too large")
        classification = value.get("classification", {})
        demand = classification.get("predicted_demand_level")
        if demand is not None and (not isinstance(demand, str) or demand not in {"LOW", "HIGH"}):
            raise ValueError("scenario_result contains an invalid demand label")
        numeric_bounds = {
            ("classification", "probability_high"): (0, 1),
            ("regression", "predicted_connections_next_hour"): (0, 100_000),
            ("regression", "prediction_interval_lower"): (0, 100_000),
            ("regression", "prediction_interval_upper"): (0, 100_000),
            ("regression", "interval_confidence"): (0, 1),
            ("capacity", "predicted_capacity_utilization_pct"): (0, 10_000),
        }
        for (section, name), (minimum, maximum) in numeric_bounds.items():
            number = value.get(section, {}).get(name)
            if number is not None and (
                type(number) not in {int, float} or not minimum <= number <= maximum
            ):
                raise ValueError(f"scenario_result field '{name}' is outside supported bounds")
        lower = value.get("regression", {}).get("prediction_interval_lower")
        upper = value.get("regression", {}).get("prediction_interval_upper")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("scenario_result prediction interval bounds are inconsistent")
        factors = value.get("explanation", {}).get("top_factors", [])
        if not isinstance(factors, list) or len(factors) > 5:
            raise ValueError("scenario_result has an unsupported factor list")
        if any(
            not isinstance(item, dict)
            or not isinstance(item.get("label"), str)
            or not _SAFE_LABEL.fullmatch(item["label"])
            for item in factors
        ):
            raise ValueError("scenario_result contains an unsafe factor label")
        return value

    @model_validator(mode="after")
    def coordinates_are_paired(self) -> AssistantContext:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        return self


class ChatRequest(BaseModel):
    """User message, optional session key, and explicit scenario context."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)
    context: AssistantContext | None = None

    @field_validator("message")
    @classmethod
    def reject_whitespace_only(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("message must contain non-whitespace characters")
        return value

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str | None) -> str | None:
        if value is not None and not _SAFE_ID.fullmatch(value):
            raise ValueError("session_id contains unsupported characters")
        return value


class ChatResponse(BaseModel):
    """Tool-grounded natural-language response and structured provenance."""

    answer: str
    intent: str
    sources: list[str]
    scenario: dict[str, Any] | None = None
    structured_result: dict[str, Any] | list[Any] | None = None
    session_id: str
