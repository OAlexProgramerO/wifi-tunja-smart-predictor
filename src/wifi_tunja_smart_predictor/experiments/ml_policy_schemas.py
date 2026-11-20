"""Contracts for immutable reusable ML evaluation policy profiles."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wifi_tunja_smart_predictor.experiments.ml_gates_schemas import (
    MLGateCriterion,
    validate_gate_criteria,
)


class MLEvaluationPolicyCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_id: str | None = Field(default=None, min_length=1, max_length=80)
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    tags: list[str] = Field(default_factory=list, max_length=10)
    classification_criteria: list[MLGateCriterion] | None = Field(default=None, max_length=8)
    regression_criteria: list[MLGateCriterion] | None = Field(default=None, max_length=8)

    @model_validator(mode="after")
    def validate_profile(self) -> MLEvaluationPolicyCreate:
        self.name = self.name.strip()
        if not self.name:
            raise ValueError("name cannot be blank.")
        if len(set(self.tags)) != len(self.tags) or any(
            not tag.strip() or len(tag) > 32 for tag in self.tags
        ):
            raise ValueError("tags must be unique non-empty strings of at most 32 characters.")
        profiles = {
            "classification": self.classification_criteria,
            "regression": self.regression_criteria,
        }
        if not any(criteria for criteria in profiles.values()):
            raise ValueError("at least one task must have acceptance criteria.")
        for task, criteria in profiles.items():
            if criteria:
                validate_gate_criteria(task, criteria)
        return self


class MLEvaluationPolicyRecord(BaseModel):
    policy_id: str
    version: int
    name: str
    description: str
    tags: list[str]
    created_at_utc: str
    criteria: dict[str, list[dict[str, Any]]]


class MLEvaluationPolicyPage(BaseModel):
    items: list[MLEvaluationPolicyRecord]
    limit: int
    offset: int
    total_count: int


class MLEvaluationPolicyVersionPage(BaseModel):
    items: list[MLEvaluationPolicyRecord]
    limit: int
    offset: int
    total_count: int


class MLEvaluationPolicyRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str = Field(min_length=1, max_length=80)


class MLEvaluationPolicyRunResponse(BaseModel):
    report_schema_version: str
    run_id: str
    created_at_utc: str
    generated_at_utc: str
    project_version: str
    data_boundary: str
    policy: MLEvaluationPolicyRecord
    experiment: dict[str, Any]
    status: Literal["PASSED", "FAILED", "INCONCLUSIVE"]
    explanation: str
    reasons: list[str]
    metrics: dict[str, float | None]
    criteria: list[dict[str, Any]]
    compatibility: dict[str, Any]
    limitations: list[str]


class MLEvaluationPolicyRunHistoryItem(BaseModel):
    run_id: str
    created_at_utc: str
    experiment_id: str
    policy_id: str
    policy_version: int
    status: Literal["PASSED", "FAILED", "INCONCLUSIVE"]
    criteria: list[dict[str, Any]]
    provenance: dict[str, Any]


class MLEvaluationPolicyRunHistoryPage(BaseModel):
    items: list[MLEvaluationPolicyRunHistoryItem]
    limit: int
    offset: int
    total_count: int


class MLEvaluationPolicyRunDetail(MLEvaluationPolicyRunHistoryItem):
    report: MLEvaluationPolicyRunResponse


class MLEvaluationPolicyRunComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    left_run_id: str = Field(min_length=1, max_length=80)
    right_run_id: str = Field(min_length=1, max_length=80)

    @model_validator(mode="after")
    def require_distinct_run_ids(self) -> MLEvaluationPolicyRunComparisonRequest:
        if self.left_run_id == self.right_run_id:
            raise ValueError("left_run_id and right_run_id must be different.")
        return self


class MLEvaluationPolicyRunComparisonResponse(BaseModel):
    report_schema_version: str
    generated_at_utc: str
    status: Literal["COMPATIBLE", "PARTIAL", "INCOMPATIBLE"]
    run_ids: list[str | None]
    context: dict[str, Any]
    reasons: list[str]
    reason_details: list[dict[str, str]]
    criterion_comparisons: list[dict[str, Any]]
    interpretation: str
    limitations: list[str]
