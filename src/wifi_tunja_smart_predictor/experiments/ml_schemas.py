"""Validated request and response contracts for synthetic ML runs."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class MLExperimentCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task: Literal["classification", "regression"]
    model_candidate: str = Field(min_length=1, max_length=80)
    evaluation_split: Literal["validation", "test"] = "validation"
    hyperparameters: dict[str, Any] = Field(default_factory=dict, max_length=12)


class MLExperimentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    created_at_utc: datetime
    updated_at_utc: datetime
    completed_at_utc: datetime | None = None
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED"]
    configuration: dict[str, Any]
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None
    replayed: bool = False


class MLExperimentPage(BaseModel):
    items: list[MLExperimentRecord]
    limit: int
    offset: int
    total_count: int
