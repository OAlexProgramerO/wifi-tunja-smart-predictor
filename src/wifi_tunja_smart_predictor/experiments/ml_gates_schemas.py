"""Contracts for read-only synthetic experiment acceptance gates."""

from __future__ import annotations

from math import isfinite
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MLGateCriterion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = Field(min_length=1, max_length=32)
    operator: Literal["gte", "lte"]
    threshold: float

    @model_validator(mode="after")
    def validate_finite_threshold(self) -> MLGateCriterion:
        if not isfinite(self.threshold):
            raise ValueError("threshold must be finite.")
        return self


class MLExperimentGateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_ids: list[str] = Field(min_length=2, max_length=10)
    task: Literal["classification", "regression"]
    criteria: list[MLGateCriterion] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def validate_ids_and_criteria(self) -> MLExperimentGateRequest:
        if len(set(self.experiment_ids)) != len(self.experiment_ids):
            raise ValueError("experiment_ids must be unique.")
        if any(not value or len(value) > 80 for value in self.experiment_ids):
            raise ValueError("experiment_ids contain an invalid identifier.")
        allowed = (
            {"accuracy", "precision", "recall", "f1", "roc_auc"}
            if self.task == "classification"
            else {"mae", "rmse", "r2"}
        )
        expected_operator = {
            **{name: "gte" for name in ("accuracy", "precision", "recall", "f1", "roc_auc", "r2")},
            "mae": "lte",
            "rmse": "lte",
        }
        names = [criterion.metric for criterion in self.criteria]
        if len(set(names)) != len(names) or any(name not in allowed for name in names):
            raise ValueError("criteria must contain unique metrics for the selected task.")
        if any(item.operator != expected_operator[item.metric] for item in self.criteria):
            raise ValueError("criterion operators must follow the metric direction.")
        return self


class MLExperimentGateResponse(BaseModel):
    report_schema_version: str
    generated_at_utc: str
    project_version: str
    data_boundary: str
    task: Literal["classification", "regression"]
    criteria: list[dict]
    results: list[dict]
    compatibility: dict
    limitations: list[str]
