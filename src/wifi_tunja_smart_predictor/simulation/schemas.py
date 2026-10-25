"""Typed, bounded request and result contracts for what-if simulations."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ScenarioZoneAssumption(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_id: str = Field(min_length=1, max_length=40)
    capacity_connections: float = Field(ge=0, le=100_000, allow_inf_nan=False)

    @field_validator("zone_id")
    @classmethod
    def normalize_zone_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("zone_id must contain a non-space character")
        return value


class ScenarioDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    zones: list[ScenarioZoneAssumption] = Field(min_length=1, max_length=12)
    resource_budget_connections: float = Field(default=0, ge=0, le=100_000, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_scenario(self) -> ScenarioDefinition:
        if not self.name.strip():
            raise ValueError("scenario name must contain a non-space character")
        zone_ids = [zone.zone_id for zone in self.zones]
        if len(zone_ids) != len({zone_id.casefold() for zone_id in zone_ids}):
            raise ValueError("scenario zone_id values must be unique")
        return self


class WhatIfSimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    datetime: datetime
    baseline: ScenarioDefinition
    alternatives: list[ScenarioDefinition] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_comparison(self) -> WhatIfSimulationRequest:
        if not self.name.strip():
            raise ValueError("experiment name must contain a non-space character")
        labels = [self.baseline.name, *(scenario.name for scenario in self.alternatives)]
        if len({label.strip().casefold() for label in labels}) != len(labels):
            raise ValueError("baseline and alternative scenario names must be unique")
        expected_zones = {zone.zone_id.casefold() for zone in self.baseline.zones}
        for scenario in self.alternatives:
            if {zone.zone_id.casefold() for zone in scenario.zones} != expected_zones:
                raise ValueError("baseline and alternatives must define the same zone IDs")
        return self


class ScenarioZoneResult(BaseModel):
    zone_id: str
    predicted_demand_connections: float
    predicted_demand_level: str
    assumed_capacity_connections: float
    allocated_budget_connections: float
    effective_capacity_connections: float
    capacity_surplus_connections: float
    capacity_deficit_connections: float
    utilization_proxy_pct: float | None
    constraint_violation: bool


class ScenarioAggregateResult(BaseModel):
    predicted_demand_connections: float
    effective_capacity_connections: float
    overload_connections: float
    utilization_proxy_pct: float | None
    allocated_budget_connections: float
    unallocated_budget_connections: float
    budget_constraint_satisfied: bool
    constraint_violations: list[dict[str, str]]


class ScenarioResult(BaseModel):
    name: str
    description: str
    datetime: datetime
    zones: list[ScenarioZoneResult]
    aggregate: ScenarioAggregateResult


class ZoneScenarioDifference(BaseModel):
    zone_id: str
    predicted_demand_difference_connections: float
    predicted_demand_difference_pct: float | None
    configured_capacity_difference_connections: float
    effective_capacity_difference_connections: float
    overload_difference_connections: float
    utilization_proxy_difference_pct_points: float | None
    assessment_status: str
    reason_codes: list[str]


class AlternativeScenarioComparison(BaseModel):
    alternative_name: str
    comparison_status: str
    assessment_status: str
    reason_codes: list[str]
    affected_zone_count: int
    baseline: ScenarioResult
    alternative: ScenarioResult
    aggregate_differences: dict[str, float | None]
    zones: list[ZoneScenarioDifference]


class WhatIfSimulationResult(BaseModel):
    schema_version: str = "1.0"
    metric_registry_version: str
    assessment_status: str
    reason_codes: list[str]
    baseline: ScenarioResult
    comparisons: list[AlternativeScenarioComparison]
    metric_definitions: dict[str, Any]
    provenance: dict[str, Any]
    disclaimer: str
    limitations: list[str]
