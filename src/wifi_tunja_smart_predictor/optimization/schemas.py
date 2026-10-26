"""Typed input and result contracts for bounded integer allocation optimization."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OptimizationObjective = Literal["MINIMIZE_TOTAL_DEFICIT", "MINIMIZE_WEIGHTED_DEFICIT"]


class OptimizationZoneInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_id: str = Field(min_length=1, max_length=40)
    capacity_connections: float = Field(ge=0, le=100_000, allow_inf_nan=False)
    baseline_allocation_connections: int = Field(default=0, strict=True, ge=0, le=100_000)
    minimum_allocation_connections: int = Field(default=0, strict=True, ge=0, le=100_000)
    maximum_allocation_connections: int = Field(default=100_000, strict=True, ge=0, le=100_000)
    priority_weight: float | None = Field(default=None, gt=0, le=10_000, allow_inf_nan=False)

    @field_validator("zone_id")
    @classmethod
    def normalize_zone_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("zone_id must contain a non-space character")
        return normalized

    @model_validator(mode="after")
    def validate_allocation_bounds(self) -> OptimizationZoneInput:
        if self.minimum_allocation_connections > self.maximum_allocation_connections:
            raise ValueError("minimum allocation cannot exceed maximum allocation")
        return self


class AllocationOptimizationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    datetime: datetime
    objective: OptimizationObjective = "MINIMIZE_TOTAL_DEFICIT"
    total_budget_connections: int = Field(strict=True, ge=0, le=100_000)
    max_allocation_steps: int = Field(default=100_000, strict=True, ge=1, le=100_000)
    zones: list[OptimizationZoneInput] = Field(min_length=1, max_length=12)

    @model_validator(mode="after")
    def validate_optimization(self) -> AllocationOptimizationRequest:
        if not self.name.strip():
            raise ValueError("optimization name must contain a non-space character")
        zone_ids = [zone.zone_id.casefold() for zone in self.zones]
        if len(zone_ids) != len(set(zone_ids)):
            raise ValueError("zone_id values must be unique")
        if self.objective == "MINIMIZE_WEIGHTED_DEFICIT":
            if any(zone.priority_weight is None for zone in self.zones):
                raise ValueError("weighted objective requires a priority weight for every zone")
        elif any(zone.priority_weight is not None for zone in self.zones):
            raise ValueError("priority weights are only valid for the weighted objective")
        return self


class OptimizationZoneResult(BaseModel):
    zone_id: str
    predicted_demand_connections: float
    capacity_connections: float
    allocation_connections: int
    effective_capacity_connections: float
    deficit_connections: float
    priority_weight: float
    minimum_allocation_connections: int
    maximum_allocation_connections: int
    allocation_within_bounds: bool


class OptimizationPlanResult(BaseModel):
    feasible: bool
    feasibility_reason_codes: list[str]
    allocated_connections: int
    total_budget_connections: int
    unallocated_budget_connections: int
    total_deficit_connections: float
    weighted_deficit_score: float
    zones_with_deficit: int
    zones: list[OptimizationZoneResult]


class AllocationOptimizationResult(BaseModel):
    schema_version: str = "1.0"
    algorithm_id: str = "marginal_benefit_greedy"
    algorithm_version: str = "1.0"
    optimization_status: Literal["optimal", "infeasible", "search_limit_reached"]
    assessment_status: Literal["assessable", "not_assessable"]
    objective: OptimizationObjective
    objective_definition: dict[str, str]
    search_strategy: str
    search_exhaustive: bool
    optimality_proven: bool
    max_allocation_steps: int
    allocation_steps_evaluated: int
    candidates_evaluated: int
    feasible_candidates: int
    infeasible_candidates: int
    baseline: OptimizationPlanResult
    selected_candidate: OptimizationPlanResult | None
    objective_improvement: float | None
    selection_explanation: str
    selection_explanation_es: str
    constraints_that_limited_allocation: list[str]
    missing_metrics: list[str]
    reason_codes: list[str]
    provenance: dict[str, Any] = Field(default_factory=dict)
    synthetic_data_disclaimer: str
    limitations: list[str]
