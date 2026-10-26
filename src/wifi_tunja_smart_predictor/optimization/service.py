"""Bounded integer allocation optimizer over fixed synthetic demand estimates."""

from __future__ import annotations

import heapq
from decimal import Decimal
from math import isfinite
from typing import Mapping

from wifi_tunja_smart_predictor.optimization.schemas import (
    AllocationOptimizationRequest,
    AllocationOptimizationResult,
    OptimizationPlanResult,
    OptimizationZoneResult,
)

_OBJECTIVES = {
    "MINIMIZE_TOTAL_DEFICIT": {
        "label_en": "Minimize total predicted capacity deficit",
        "label_es": "Minimizar el déficit total de capacidad predicho",
        "formula": "minimize sum(max(predicted demand_i - capacity_i - allocation_i, 0))",
    },
    "MINIMIZE_WEIGHTED_DEFICIT": {
        "label_en": "Minimize priority-weighted predicted capacity deficit",
        "label_es": "Minimizar el déficit predicho ponderado por prioridad",
        "formula": "minimize sum(priority_weight_i * max(predicted demand_i - capacity_i - allocation_i, 0))",
    },
}

_DISCLAIMER = (
    "All data and predictions are synthetic. Allocations are hypothetical connection counts, "
    "not verified public infrastructure requirements in Tunja."
)
_UNASSESSED_METRICS = [
    "latency",
    "throughput",
    "energy",
    "financial_cost",
    "geographic_coverage",
    "real_world_service_quality",
]


class AllocationOptimizationService:
    """Find the best feasible integer allocation for the supported separable objectives.

    For each zone, deficit reduction has non-increasing marginal benefit as integer
    allocation grows. A max-heap chooses the largest remaining marginal benefit, with
    zone ID as a stable tie-break. This is globally optimal for the two additive
    objectives under the independent per-zone bounds and one total-budget constraint.
    """

    def optimize(
        self,
        request: AllocationOptimizationRequest,
        predicted_demand: Mapping[str, float],
    ) -> AllocationOptimizationResult:
        expected_ids = {zone.zone_id for zone in request.zones}
        if set(predicted_demand) != expected_ids:
            raise ValueError("predicted demand must contain exactly the requested zone IDs")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            for value in predicted_demand.values()
        ):
            raise ValueError("predicted demand must be a finite non-negative number per zone")
        # Decimal conversion preserves the user/model decimal representation for stable
        # marginal ordering and avoids accumulated binary-float tie-breaking drift.
        demand = {key: Decimal(str(value)) for key, value in predicted_demand.items()}
        weights = {
            zone.zone_id: (
                Decimal(str(zone.priority_weight or 1.0))
                if request.objective == "MINIMIZE_WEIGHTED_DEFICIT"
                else Decimal(1)
            )
            for zone in request.zones
        }
        baseline_allocations = {
            zone.zone_id: zone.baseline_allocation_connections for zone in request.zones
        }
        minimums = {zone.zone_id: zone.minimum_allocation_connections for zone in request.zones}
        maximums = {zone.zone_id: zone.maximum_allocation_connections for zone in request.zones}
        zones_by_id = {zone.zone_id: zone for zone in request.zones}
        baseline = self._plan(
            request,
            demand,
            weights,
            baseline_allocations,
            is_candidate=False,
        )
        minimum_total = sum(minimums.values())
        if minimum_total > request.total_budget_connections:
            return AllocationOptimizationResult(
                optimization_status="infeasible",
                assessment_status="assessable",
                objective=request.objective,
                objective_definition=_OBJECTIVES[request.objective],
                search_strategy="bounded integer marginal-benefit greedy",
                search_exhaustive=False,
                optimality_proven=False,
                max_allocation_steps=request.max_allocation_steps,
                allocation_steps_evaluated=0,
                candidates_evaluated=0,
                feasible_candidates=0,
                infeasible_candidates=1,
                baseline=baseline,
                selected_candidate=None,
                objective_improvement=None,
                selection_explanation=(
                    "No allocation satisfies the hard constraints because the sum of per-zone "
                    "minimums exceeds the total budget."
                ),
                selection_explanation_es=(
                    "Ninguna asignación cumple las restricciones porque la suma de los mínimos "
                    "por zona supera el presupuesto total."
                ),
                constraints_that_limited_allocation=["MINIMUM_ALLOCATIONS_EXCEED_TOTAL_BUDGET"],
                missing_metrics=_UNASSESSED_METRICS,
                reason_codes=[
                    "INFEASIBLE_CONSTRAINTS",
                    "NO_FEASIBLE_CANDIDATE",
                    "METRIC_NOT_ASSESSABLE",
                ],
                synthetic_data_disclaimer=_DISCLAIMER,
                limitations=[
                    "The baseline is shown for context and may itself violate the requested candidate bounds.",
                    "No optimization candidate is recommended for an infeasible request.",
                ],
            )

        allocations = dict(minimums)
        heap: list[tuple[Decimal, str]] = []
        for zone in sorted(request.zones, key=lambda item: item.zone_id):
            marginal = self._marginal_benefit(
                zone.zone_id,
                allocations,
                maximums,
                demand,
                zone.capacity_connections,
                weights,
            )
            if marginal > 0:
                heapq.heappush(heap, (-marginal, zone.zone_id))

        budget_remaining = request.total_budget_connections - minimum_total
        steps = 0
        while budget_remaining > 0 and heap and steps < request.max_allocation_steps:
            _negative_benefit, zone_id = heapq.heappop(heap)
            allocations[zone_id] += 1
            budget_remaining -= 1
            steps += 1
            zone = zones_by_id[zone_id]
            marginal = self._marginal_benefit(
                zone_id, allocations, maximums, demand, zone.capacity_connections, weights
            )
            if marginal > 0:
                heapq.heappush(heap, (-marginal, zone_id))

        search_limited = (
            budget_remaining > 0 and bool(heap) and steps >= request.max_allocation_steps
        )
        selected = self._plan(request, demand, weights, allocations, is_candidate=True)
        if search_limited:
            status = "search_limit_reached"
            reason_codes = [
                "CANDIDATE_LIMIT_REACHED",
                "SEARCH_BUDGET_EXHAUSTED",
                "OPTIMALITY_NOT_PROVEN",
            ]
            explanation = (
                "This is the best feasible allocation reached before the configured allocation-step "
                "limit. The search stopped with beneficial budget remaining, so optimality is not proven."
            )
            explanation_es = (
                "Esta es la mejor asignación factible alcanzada antes del límite de pasos configurado. "
                "Quedó presupuesto con beneficio posible, por lo que no se demuestra optimalidad."
            )
        else:
            status = "optimal"
            reason_codes = []
            explanation = (
                "The candidate minimizes the selected additive deficit objective under the declared "
                "integer bounds and total budget. Non-increasing marginal benefits make the greedy "
                "allocation globally optimal for these constraints."
            )
            explanation_es = (
                "El candidato minimiza el objetivo aditivo de déficit bajo los límites enteros y el "
                "presupuesto declarados. Las mejoras marginales no crecientes hacen que la asignación "
                "greedy sea globalmente óptima para estas restricciones."
            )
        constraints = []
        if selected.unallocated_budget_connections > 0:
            constraints.append(
                "NO_REMAINING_DEFICIT_BENEFIT"
                if all(row.deficit_connections == 0 for row in selected.zones)
                else "ZONE_MAXIMUMS_LIMITED_ALLOCATION"
            )
        if search_limited:
            constraints.append("ALLOCATION_STEP_LIMIT_REACHED")
        improvement = baseline.weighted_deficit_score - selected.weighted_deficit_score
        return AllocationOptimizationResult(
            optimization_status=status,
            assessment_status="assessable",
            objective=request.objective,
            objective_definition=_OBJECTIVES[request.objective],
            search_strategy="bounded integer marginal-benefit greedy with stable zone_id tie-break",
            search_exhaustive=False,
            optimality_proven=not search_limited,
            max_allocation_steps=request.max_allocation_steps,
            allocation_steps_evaluated=steps,
            candidates_evaluated=steps + 1,
            feasible_candidates=steps + 1,
            infeasible_candidates=0,
            baseline=baseline,
            selected_candidate=selected,
            objective_improvement=float(improvement),
            selection_explanation=explanation,
            selection_explanation_es=explanation_es,
            constraints_that_limited_allocation=constraints,
            missing_metrics=_UNASSESSED_METRICS,
            reason_codes=sorted(set([*reason_codes, "METRIC_NOT_ASSESSABLE"])),
            synthetic_data_disclaimer=_DISCLAIMER,
            limitations=[
                "Predicted demand is fixed; allocations do not change the model output.",
                "The objective covers predicted connection-count deficits only, not cost, latency, throughput, coverage, or service quality.",
                "The proof applies only to the separable deficit objectives, integer allocation units, independent per-zone bounds, and one total-budget constraint.",
                "Synthetic predictions and mathematically optimal allocations do not establish real operational or economic requirements in Tunja.",
            ],
        )

    @staticmethod
    def _marginal_benefit(
        zone_id: str,
        allocations: dict[str, int],
        maximums: dict[str, int],
        demand: dict[str, Decimal],
        capacity: float,
        weights: dict[str, Decimal],
    ) -> Decimal:
        if allocations[zone_id] >= maximums[zone_id]:
            return Decimal(0)
        deficit = max(Decimal(0), demand[zone_id] - Decimal(str(capacity)) - allocations[zone_id])
        return weights[zone_id] * min(Decimal(1), deficit)

    @staticmethod
    def _plan(
        request: AllocationOptimizationRequest,
        demand: dict[str, Decimal],
        weights: dict[str, Decimal],
        allocations: dict[str, int],
        *,
        is_candidate: bool,
    ) -> OptimizationPlanResult:
        zones = []
        total_deficit = Decimal(0)
        weighted_deficit = Decimal(0)
        feasibility_reasons = []
        total_allocated = sum(allocations.values())
        if total_allocated > request.total_budget_connections:
            feasibility_reasons.append("TOTAL_BUDGET_EXCEEDED")
        deficit_zone_count = 0
        for zone in sorted(request.zones, key=lambda item: item.zone_id):
            allocated = allocations[zone.zone_id]
            effective_capacity = Decimal(str(zone.capacity_connections)) + allocated
            deficit = max(Decimal(0), demand[zone.zone_id] - effective_capacity)
            total_deficit += deficit
            weighted_deficit += weights[zone.zone_id] * deficit
            deficit_zone_count += int(deficit > 0)
            within_bounds = (
                zone.minimum_allocation_connections
                <= allocated
                <= zone.maximum_allocation_connections
            )
            if not within_bounds:
                feasibility_reasons.append(f"ALLOCATION_BOUNDS_VIOLATED:{zone.zone_id}")
            zones.append(
                OptimizationZoneResult(
                    zone_id=zone.zone_id,
                    predicted_demand_connections=float(demand[zone.zone_id]),
                    capacity_connections=zone.capacity_connections,
                    allocation_connections=allocated,
                    effective_capacity_connections=float(effective_capacity),
                    deficit_connections=float(deficit),
                    priority_weight=float(weights[zone.zone_id]),
                    minimum_allocation_connections=zone.minimum_allocation_connections,
                    maximum_allocation_connections=zone.maximum_allocation_connections,
                    allocation_within_bounds=within_bounds,
                )
            )
        if (
            is_candidate
            and sum(zone.minimum_allocation_connections for zone in request.zones)
            > request.total_budget_connections
        ):
            feasibility_reasons.append("MINIMUM_ALLOCATIONS_EXCEED_TOTAL_BUDGET")
        return OptimizationPlanResult(
            feasible=not feasibility_reasons,
            feasibility_reason_codes=feasibility_reasons,
            allocated_connections=total_allocated,
            total_budget_connections=request.total_budget_connections,
            unallocated_budget_connections=max(
                0, request.total_budget_connections - total_allocated
            ),
            total_deficit_connections=float(total_deficit),
            weighted_deficit_score=float(weighted_deficit),
            zones_with_deficit=deficit_zone_count,
            zones=zones,
        )
