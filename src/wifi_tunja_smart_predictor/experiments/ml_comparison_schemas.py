"""Typed contracts for read-only ML experiment comparisons."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MLExperimentComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_ids: list[str] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def require_unique_ids(self) -> MLExperimentComparisonRequest:
        if len(set(self.experiment_ids)) != len(self.experiment_ids):
            raise ValueError("experiment_ids must be unique.")
        if any(not value or len(value) > 80 for value in self.experiment_ids):
            raise ValueError("experiment_ids contain an invalid identifier.")
        return self


class MLExperimentComparisonMetric(BaseModel):
    metric: str
    direction: Literal["higher_is_better", "lower_is_better"]
    left_value: float
    right_value: float
    delta: float
    assessment: Literal["IMPROVEMENT", "REGRESSION", "TIE"]


class MLExperimentComparisonResponse(BaseModel):
    report_schema_version: str
    generated_at_utc: str
    project_version: str
    data_boundary: str
    overall_status: Literal["COMPARABLE", "PARTIAL", "INCOMPATIBLE"]
    experiments: list[dict]
    comparisons: list[dict]
    interpretation: str
    limitations: list[str]
