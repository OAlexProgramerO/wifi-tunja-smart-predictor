"""Bounded, deterministic sensitivity analysis over saved optimization runs."""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from wifi_tunja_smart_predictor.optimization.schemas import (
    AllocationOptimizationRequest,
    AllocationOptimizationResult,
)
from wifi_tunja_smart_predictor.optimization.service import AllocationOptimizationService


class RobustnessAnalysisRequest(BaseModel):
    """One-factor demand stresses and an inclusive, bounded budget sweep."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    parent_experiment_id: str = Field(min_length=1, max_length=64)
    demand_multiplier_min: float = Field(default=0.8, ge=0.25, le=2, allow_inf_nan=False)
    demand_multiplier_max: float = Field(default=1.2, ge=0.25, le=2, allow_inf_nan=False)
    demand_multiplier_step: float = Field(default=0.2, gt=0, le=1, allow_inf_nan=False)
    budget_min_connections: int = Field(default=0, strict=True, ge=0, le=100_000)
    budget_max_connections: int = Field(default=1_000, strict=True, ge=0, le=100_000)
    budget_step_connections: int = Field(default=100, strict=True, ge=1, le=100_000)
    max_evaluations: int = Field(default=100, strict=True, ge=1, le=100)
    max_allocation_steps_per_evaluation: int = Field(default=10_000, strict=True, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_ranges(self) -> RobustnessAnalysisRequest:
        if not self.name.strip():
            raise ValueError("analysis name must contain a non-space character")
        if self.demand_multiplier_min > self.demand_multiplier_max:
            raise ValueError("demand multiplier minimum cannot exceed maximum")
        if self.budget_min_connections > self.budget_max_connections:
            raise ValueError("budget minimum cannot exceed maximum")
        multiplier_count = (
            int(
                (
                    Decimal(str(self.demand_multiplier_max))
                    - Decimal(str(self.demand_multiplier_min))
                )
                // Decimal(str(self.demand_multiplier_step))
            )
            + 1
        )
        if multiplier_count > 6:
            raise ValueError("demand multiplier range may contain at most 6 values")
        budget_count = (
            (self.budget_max_connections - self.budget_min_connections)
            // self.budget_step_connections
        ) + 1
        if budget_count > 20:
            raise ValueError("budget sweep may contain at most 20 values")
        return self


class RobustnessScenarioResult(BaseModel):
    scenario_id: str
    scenario_type: Literal["baseline", "demand_stress", "budget_sweep"]
    varied_parameter: str
    varied_zone_id: str | None
    varied_value: float | int
    demand_multiplier: float
    budget_connections: int
    feasibility_status: Literal["feasible", "infeasible", "not_assessable"]
    assessment_status: Literal["assessable", "not_assessable"]
    optimizer_status: str
    optimality_proven: bool
    parent_objective_value: float
    original_allocation_feasible: bool
    original_allocation_objective_value: float
    reoptimized_objective_value: float | None
    zones_with_deficit: int | None
    unallocated_budget_connections: int | None
    objective_degradation: float | None
    objective_degradation_pct: float | None
    reference_allocations: dict[str, int]
    reoptimized_allocations: dict[str, int]
    allocation_differences: dict[str, int]
    total_absolute_allocation_change: int | None
    changed_zone_count: int | None
    constraint_violation_count: int
    failed_constraints: list[str]
    minimum_additional_budget_connections: int
    reason_codes: list[str]


class BudgetFrontierPoint(BaseModel):
    budget_connections: int
    feasibility_status: Literal["feasible", "infeasible", "not_assessable"]
    objective_value: float | None
    total_allocated_connections: int | None
    unallocated_budget_connections: int | None
    zones_with_deficit: int | None
    allocation_change_from_previous_budget: dict[str, int] | None
    minimum_additional_budget_connections: int
    optimizer_status: str
    optimality_proven: bool
    reason_codes: list[str]


class ZoneStabilityResult(BaseModel):
    zone_id: str
    assessable_scenario_count: int
    unchanged_allocation_count: int
    allocation_stability_rate: float | None
    mean_absolute_allocation_change: float | None
    maximum_absolute_allocation_change: int | None
    demand_stress_case_count: int
    own_demand_stress_allocation_change_count: int


class RobustnessAnalysisResult(BaseModel):
    schema_version: str = "1.0"
    analysis_method: str = "one_factor_at_a_time_and_bounded_budget_sweep"
    analysis_status: Literal["completed", "partial"]
    assessment_status: Literal["assessable", "not_assessable"]
    parent_experiment_id: str
    parent_integrity_status: str
    objective: str
    objective_direction: Literal["minimize"] = "minimize"
    reference_allocation_source: Literal["selected_candidate", "parent_baseline"]
    parent_objective_value: float
    scenarios_defined: int
    scenarios_evaluated: int
    feasible_scenario_count: int
    infeasible_scenario_count: int
    not_assessable_scenario_count: int
    optimizer_incomplete_scenario_count: int
    feasibility_rate_under_tested_scenarios: float | None
    feasibility_rate_denominator_count: int
    max_evaluations: int
    evaluations_performed: int
    max_allocation_steps_per_evaluation: int
    demand_rounding_method: str
    scenario_results: list[RobustnessScenarioResult]
    budget_frontier: list[BudgetFrontierPoint]
    zone_stability: list[ZoneStabilityResult]
    most_sensitive_zones: list[str]
    reason_codes: list[str]
    provenance: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str]
    synthetic_data_disclaimer: str


class RobustnessAnalysisService:
    """Evaluate bounded one-factor stresses using the V6.1 optimizer unchanged."""

    _DISCLAIMER = (
        "All predictions, demand adjustments, budgets, and allocations are synthetic or hypothetical. "
        "Scenario counts are a hand-defined test set, not probabilities or evidence of real-world reliability in Tunja."
    )

    def __init__(self, optimizer: AllocationOptimizationService | None = None) -> None:
        self.optimizer = optimizer or AllocationOptimizationService()

    def analyze(
        self, parent_record: dict[str, Any], request: RobustnessAnalysisRequest
    ) -> RobustnessAnalysisResult:
        if parent_record.get("id") != request.parent_experiment_id:
            raise ValueError("parent experiment ID does not match the saved record")
        configuration = parent_record.get("configuration")
        result = parent_record.get("result")
        if not isinstance(configuration, dict) or not isinstance(result, dict):
            raise ValueError("parent experiment record is malformed")
        if configuration.get("scenario_kind") != "allocation_optimization":
            raise ValueError("parent experiment is not an allocation optimization run")
        if self._integrity_status(parent_record) == "invalid":
            raise ValueError("parent optimization integrity is invalid")
        original_request = AllocationOptimizationRequest.model_validate(
            configuration.get("optimization")
        )
        parent_optimization = AllocationOptimizationResult.model_validate(
            result.get("optimization")
        )
        reference_plan = parent_optimization.selected_candidate
        source: Literal["selected_candidate", "parent_baseline"] = "selected_candidate"
        if reference_plan is None:
            reference_plan = parent_optimization.baseline
            source = "parent_baseline"
        reference_rows = {row.zone_id: row for row in reference_plan.zones}
        zone_ids = {zone.zone_id for zone in original_request.zones}
        if set(reference_rows) != zone_ids:
            raise ValueError("parent optimization does not include the requested zone results")
        base_demand = {
            zone_id: reference_rows[zone_id].predicted_demand_connections for zone_id in zone_ids
        }
        reference_allocations = {
            zone_id: reference_rows[zone_id].allocation_connections for zone_id in zone_ids
        }
        parent_objective_value = reference_plan.weighted_deficit_score
        multipliers = self._range(
            request.demand_multiplier_min,
            request.demand_multiplier_max,
            request.demand_multiplier_step,
        )
        if not any(value == Decimal("1") for value in multipliers):
            multipliers.append(Decimal("1"))
        multipliers = sorted(set(multipliers))
        if len(multipliers) > 6:
            raise ValueError("demand multiplier range plus reference may contain at most 6 values")
        budgets = self._range(
            request.budget_min_connections,
            request.budget_max_connections,
            request.budget_step_connections,
        )
        if Decimal(original_request.total_budget_connections) not in budgets:
            budgets.append(Decimal(original_request.total_budget_connections))
        budgets = sorted(set(budgets))
        if len(budgets) > 20:
            raise ValueError("budget sweep plus parent budget may contain at most 20 values")

        definitions: list[tuple[str, str, str | None, Decimal, int]] = [
            ("baseline", "baseline", None, Decimal("1"), original_request.total_budget_connections)
        ]
        for zone_id in sorted(zone_ids):
            for multiplier in multipliers:
                if multiplier == 1:
                    continue
                definitions.append(
                    (
                        "demand_stress",
                        f"demand_multiplier:{zone_id}",
                        zone_id,
                        multiplier,
                        original_request.total_budget_connections,
                    )
                )
        for budget in budgets:
            budget_value = int(budget)
            if budget_value == original_request.total_budget_connections:
                continue
            definitions.append(
                ("budget_sweep", "total_budget_connections", None, Decimal("1"), budget_value)
            )

        scenarios_defined = len(definitions)
        selected_definitions = definitions[: request.max_evaluations]
        partial = len(selected_definitions) < scenarios_defined
        scenarios: list[RobustnessScenarioResult] = []
        for index, (scenario_type, parameter, zone_id, multiplier, budget) in enumerate(
            selected_definitions, start=1
        ):
            multiplier_for_budget = multiplier
            varied_value: float | int = (
                budget if scenario_type == "budget_sweep" else float(multiplier)
            )
            if scenario_type == "baseline":
                varied_value = original_request.total_budget_connections
            demand = dict(base_demand)
            if zone_id is not None:
                demand[zone_id] = float(Decimal(str(base_demand[zone_id])) * multiplier)
            changed_zones = [
                zone.model_copy(
                    update={"baseline_allocation_connections": reference_allocations[zone.zone_id]}
                )
                for zone in original_request.zones
            ]
            scenario_request = original_request.model_copy(
                update={
                    "total_budget_connections": budget,
                    "max_allocation_steps": min(
                        original_request.max_allocation_steps,
                        request.max_allocation_steps_per_evaluation,
                    ),
                    "zones": changed_zones,
                }
            )
            optimized = self.optimizer.optimize(scenario_request, demand)
            original_plan = optimized.baseline
            candidate = optimized.selected_candidate
            assessment = self._integrity_status(parent_record) == "verified"
            feasible = candidate is not None and candidate.feasible
            if not feasible:
                feasibility_status: Literal["feasible", "infeasible", "not_assessable"] = (
                    "infeasible"
                )
            else:
                feasibility_status = "feasible"
            candidate_allocations = (
                {row.zone_id: row.allocation_connections for row in candidate.zones}
                if candidate is not None
                else {}
            )
            differences = {
                zone: candidate_allocations[zone] - reference_allocations[zone]
                for zone in candidate_allocations
            }
            new_objective = candidate.weighted_deficit_score if candidate is not None else None
            degradation = (
                new_objective - original_plan.weighted_deficit_score
                if new_objective is not None
                else None
            )
            degradation_pct = (
                degradation / abs(original_plan.weighted_deficit_score) * 100
                if degradation is not None and original_plan.weighted_deficit_score != 0
                else None
            )
            min_required = sum(
                zone.minimum_allocation_connections for zone in original_request.zones
            )
            reasons = list(optimized.reason_codes)
            if not assessment:
                reasons.append("PARENT_PROVENANCE_INCOMPLETE")
            if partial:
                reasons.append("ANALYSIS_LIMIT_REACHED")
            scenarios.append(
                RobustnessScenarioResult(
                    scenario_id=f"scenario-{index:03d}",
                    scenario_type=scenario_type,
                    varied_parameter=parameter,
                    varied_zone_id=zone_id,
                    varied_value=varied_value,
                    demand_multiplier=float(multiplier_for_budget),
                    budget_connections=budget,
                    feasibility_status=feasibility_status,
                    assessment_status="assessable" if assessment else "not_assessable",
                    optimizer_status=optimized.optimization_status,
                    optimality_proven=optimized.optimality_proven,
                    parent_objective_value=parent_objective_value,
                    original_allocation_feasible=original_plan.feasible,
                    original_allocation_objective_value=original_plan.weighted_deficit_score,
                    reoptimized_objective_value=new_objective,
                    zones_with_deficit=(
                        candidate.zones_with_deficit if candidate is not None else None
                    ),
                    unallocated_budget_connections=(
                        candidate.unallocated_budget_connections if candidate is not None else None
                    ),
                    objective_degradation=degradation,
                    objective_degradation_pct=degradation_pct,
                    reference_allocations=reference_allocations,
                    reoptimized_allocations=candidate_allocations,
                    allocation_differences=differences,
                    total_absolute_allocation_change=(
                        sum(abs(value) for value in differences.values())
                        if candidate is not None
                        else None
                    ),
                    changed_zone_count=(
                        sum(value != 0 for value in differences.values())
                        if candidate is not None
                        else None
                    ),
                    constraint_violation_count=(
                        len(original_plan.feasibility_reason_codes)
                        + (
                            len(optimized.constraints_that_limited_allocation)
                            if candidate is None
                            else 0
                        )
                    ),
                    failed_constraints=sorted(
                        set(
                            original_plan.feasibility_reason_codes
                            + (
                                optimized.constraints_that_limited_allocation
                                if candidate is None
                                else []
                            )
                        )
                    ),
                    minimum_additional_budget_connections=max(0, min_required - budget),
                    reason_codes=sorted(set(reasons)),
                )
            )

        scenario_by_budget = {
            row.budget_connections: row
            for row in scenarios
            if row.scenario_type in {"baseline", "budget_sweep"}
        }
        frontier: list[BudgetFrontierPoint] = []
        previous_allocations: dict[str, int] | None = None
        for budget in sorted(scenario_by_budget):
            row = scenario_by_budget[budget]
            differences = (
                {
                    zone: row.reoptimized_allocations[zone] - previous_allocations[zone]
                    for zone in row.reoptimized_allocations
                }
                if previous_allocations is not None
                and set(previous_allocations) == set(row.reoptimized_allocations)
                else None
            )
            frontier.append(
                BudgetFrontierPoint(
                    budget_connections=budget,
                    feasibility_status=row.feasibility_status,
                    objective_value=row.reoptimized_objective_value,
                    total_allocated_connections=(
                        sum(row.reoptimized_allocations.values())
                        if row.reoptimized_allocations
                        else None
                    ),
                    unallocated_budget_connections=(row.unallocated_budget_connections),
                    zones_with_deficit=row.zones_with_deficit,
                    allocation_change_from_previous_budget=differences,
                    minimum_additional_budget_connections=row.minimum_additional_budget_connections,
                    optimizer_status=row.optimizer_status,
                    optimality_proven=row.optimality_proven,
                    reason_codes=row.reason_codes,
                )
            )
            if row.reoptimized_allocations:
                previous_allocations = row.reoptimized_allocations

        feasible_count = sum(row.feasibility_status == "feasible" for row in scenarios)
        infeasible_count = sum(row.feasibility_status == "infeasible" for row in scenarios)
        not_assessable_count = sum(row.assessment_status == "not_assessable" for row in scenarios)
        optimizer_incomplete_count = sum(
            row.optimizer_status == "search_limit_reached" for row in scenarios
        )
        stability = self._stability(scenarios, zone_ids, reference_allocations)
        sensitivity_order = sorted(
            stability,
            key=lambda row: (
                -(row.maximum_absolute_allocation_change or 0),
                -(row.mean_absolute_allocation_change or 0),
                row.zone_id,
            ),
        )
        reasons = []
        if partial:
            reasons.append("ANALYSIS_LIMIT_REACHED")
        if optimizer_incomplete_count:
            reasons.append("OPTIMIZER_SEARCH_LIMIT_REACHED")
        analysis_partial = partial or optimizer_incomplete_count > 0
        if self._integrity_status(parent_record) != "verified":
            reasons.append("PARENT_PROVENANCE_INCOMPLETE")
        return RobustnessAnalysisResult(
            analysis_status="partial" if analysis_partial else "completed",
            assessment_status=(
                "assessable"
                if self._integrity_status(parent_record) == "verified"
                else "not_assessable"
            ),
            parent_experiment_id=request.parent_experiment_id,
            parent_integrity_status=self._integrity_status(parent_record),
            objective=original_request.objective,
            reference_allocation_source=source,
            parent_objective_value=parent_objective_value,
            scenarios_defined=scenarios_defined,
            scenarios_evaluated=len(scenarios),
            feasible_scenario_count=feasible_count,
            infeasible_scenario_count=infeasible_count,
            not_assessable_scenario_count=not_assessable_count,
            optimizer_incomplete_scenario_count=optimizer_incomplete_count,
            feasibility_rate_under_tested_scenarios=(
                feasible_count / len(scenarios) if scenarios else None
            ),
            feasibility_rate_denominator_count=len(scenarios),
            max_evaluations=request.max_evaluations,
            evaluations_performed=len(scenarios),
            max_allocation_steps_per_evaluation=request.max_allocation_steps_per_evaluation,
            demand_rounding_method=(
                "Stressed demand retains the stored decimal model estimate multiplied by the selected factor; "
                "no integer rounding is applied. Allocation and budget remain integer connection counts."
            ),
            scenario_results=scenarios,
            budget_frontier=frontier,
            zone_stability=stability,
            most_sensitive_zones=[row.zone_id for row in sensitivity_order],
            reason_codes=reasons,
            provenance=result.get("provenance") or configuration.get("parent_provenance", {}),
            limitations=[
                "Demand multipliers vary one synthetic zone at a time; the budget sweep varies budget at the original stored demand. Joint demand-budget interactions are not exhaustively combined.",
                "The feasibility rate divides mathematically feasible evaluated cases by all evaluated cases. Parent provenance assessability is reported separately; this descriptive rate is not a probability.",
                "Allocation stability rates use assessable scenarios with a returned feasible candidate as their denominator; the denominator is reported per zone.",
                "A budget shortfall is calculated exactly as max(0, sum(per-zone minimum allocations) - tested budget) for the optimizer's declared hard constraints.",
                "Objective degradation is the reoptimized objective minus the original reference allocation's objective under the same varied assumptions; percentage is null when the reference value is zero.",
                "All demand stresses, outputs, and allocations are hypothetical and synthetic, not calibrated worst cases or real operational reliability estimates.",
            ],
            synthetic_data_disclaimer=self._DISCLAIMER,
        )

    @staticmethod
    def _range(start: float | int, stop: float | int, step: float | int) -> list[Decimal]:
        lower, upper, increment = Decimal(str(start)), Decimal(str(stop)), Decimal(str(step))
        values = []
        current = lower
        while current <= upper:
            values.append(current)
            current += increment
        return values

    @staticmethod
    def _integrity_status(parent_record: dict[str, Any]) -> str:
        integrity = parent_record.get("integrity")
        return (
            integrity.get("status", "incomplete") if isinstance(integrity, dict) else "incomplete"
        )

    @staticmethod
    def _stability(
        scenarios: list[RobustnessScenarioResult],
        zone_ids: set[str],
        reference_allocations: dict[str, int],
    ) -> list[ZoneStabilityResult]:
        rows = []
        successful = [
            scenario
            for scenario in scenarios
            if scenario.assessment_status == "assessable" and scenario.reoptimized_allocations
        ]
        for zone_id in sorted(zone_ids):
            changes = [
                abs(scenario.allocation_differences[zone_id])
                for scenario in successful
                if zone_id in scenario.allocation_differences
            ]
            demand_scenarios = [
                scenario
                for scenario in successful
                if scenario.scenario_type == "demand_stress" and scenario.varied_zone_id == zone_id
            ]
            rows.append(
                ZoneStabilityResult(
                    zone_id=zone_id,
                    assessable_scenario_count=len(changes),
                    unchanged_allocation_count=sum(value == 0 for value in changes),
                    allocation_stability_rate=(
                        sum(value == 0 for value in changes) / len(changes) if changes else None
                    ),
                    mean_absolute_allocation_change=(
                        sum(changes) / len(changes) if changes else None
                    ),
                    maximum_absolute_allocation_change=max(changes) if changes else None,
                    demand_stress_case_count=len(demand_scenarios),
                    own_demand_stress_allocation_change_count=sum(
                        scenario.allocation_differences.get(zone_id, 0) != 0
                        for scenario in demand_scenarios
                    ),
                )
            )
        return rows
