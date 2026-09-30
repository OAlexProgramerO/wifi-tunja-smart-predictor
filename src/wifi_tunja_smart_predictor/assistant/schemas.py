"""Validated assistant request, context, and response contracts."""

from __future__ import annotations

from datetime import datetime as DateTime
from typing import Any

from pydantic import BaseModel, Field, model_validator


class AssistantContext(BaseModel):
    """Optional location/time context supplied by a dashboard client."""

    zone: str | None = None
    zone_id: str | None = None
    access_point_id: str | None = None
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    datetime: DateTime | None = None

    @model_validator(mode="after")
    def coordinates_are_paired(self) -> AssistantContext:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be supplied together")
        return self


class ChatRequest(BaseModel):
    """User message, optional session key, and explicit scenario context."""

    message: str = Field(min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, max_length=128)
    context: AssistantContext | None = None


class ChatResponse(BaseModel):
    """Tool-grounded natural-language response and structured provenance."""

    answer: str
    intent: str
    sources: list[str]
    scenario: dict[str, Any] | None = None
    structured_result: dict[str, Any] | list[Any] | None = None
    session_id: str
