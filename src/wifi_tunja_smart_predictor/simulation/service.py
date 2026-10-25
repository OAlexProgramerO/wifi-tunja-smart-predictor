"""Reusable deterministic what-if simulation over existing model predictions."""

from __future__ import annotations

import re
from typing import Any

from wifi_tunja_smart_predictor.capacity import plan_capacity
from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.simulation.metrics import (
    METRIC_REGISTRY_VERSION,
    metric_definitions,
)
from wifi_tunja_smart_predictor.simulation.schemas import (
    AlternativeScenarioComparison,
    ScenarioAggregateResult,
    ScenarioResult,
    ScenarioZoneResult,
    WhatIfSimulationRequest,
    WhatIfSimulationResult,
    ZoneScenarioDifference,
)

SYNTHETIC_DISCLAIMER = (
    "All locations, observations, predictions, capacities, and outcomes are synthetic. "
    "These scenarios do not represent actual public WiFi infrastructure in Tunja."
)


class WhatIfSimulationService:
    """Run fixed model predictions against deterministic capacity assumptions."""

    def __init__(self, prediction_service: Any) -> None:
        self.prediction_service = prediction_service

    def run(self, request: WhatIfSimulationRequest) -> WhatIfSimulationResult:
        predictions = {
            zone.zone_id: self.prediction_service.predict(
                ScenarioRequest(datetime=request.datetime, zone_id=zone.zone_id)
            ).to_dict()
            for zone in sorted(request.baseline.zones, key=lambda item: item.zone_id)
        }
        provenance = self._provenance(predictions)
        baseline = self._evaluate(request.baseline, request.datetime, predictions)
        comparisons = [
            self._compare(baseline, self._evaluate(item, request.datetime, predictions))
            for item in request.alternatives
        ]
        reasons = self._provenance_reasons(provenance)
        status = "assessable" if not reasons else "not_assessable"
        comparisons = [
            item.model_copy(
                update={
                    "comparison_status": status,
                    "assessment_status": status,
                    "reason_codes": reasons,
                }
            )
            for item in comparisons
        ]
        return WhatIfSimulationResult(
            metric_registry_version=METRIC_REGISTRY_VERSION,
            assessment_status=status,
            reason_codes=reasons,
            baseline=baseline,
            comparisons=comparisons,
            metric_definitions=metric_definitions(),
            provenance=provenance,
            disclaimer=SYNTHETIC_DISCLAIMER,
            limitations=[
                "Alternative capacity and budget assumptions do not alter predicted demand.",
                "Utilization is a simulated connection-count proxy, not measured network utilization.",
                "A demand deficit marks a simulated constraint; it does not predict causal service improvement.",
                "Allocation is greedy by largest predicted demand deficit, with zone ID as the stable tie-break.",
            ],
        )

    @staticmethod
    def _provenance(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        first = next(iter(predictions.values()), {})
        scenario = first.get("scenario", {})
        return {
            "data": "synthetic historical analogs and model predictions",
            "dataset_version": scenario.get("dataset_version"),
            "dataset_sha256": scenario.get("dataset_sha256"),
            "dataset_fingerprint_algorithm": scenario.get("dataset_fingerprint_algorithm"),
            "application_version": (first.get("traceability") or {}).get(
                "application_version", PROJECT_VERSION
            ),
            "models": [
                {
                    **(prediction.get("traceability") or {}),
                    "application_version": (prediction.get("traceability") or {}).get(
                        "application_version", PROJECT_VERSION
                    ),
                    "zone_id": zone_id,
                }
                for zone_id, prediction in sorted(predictions.items())
            ],
        }

    @staticmethod
    def _provenance_reasons(provenance: dict[str, Any]) -> list[str]:
        reasons = []
        fingerprint = provenance.get("dataset_sha256")
        if (
            not isinstance(fingerprint, str)
            or re.fullmatch(r"[a-fA-F0-9]{64}", fingerprint) is None
        ):
            reasons.append("DATASET_FINGERPRINT_UNAVAILABLE")
        if not provenance.get("dataset_version"):
            reasons.append("DATASET_VERSION_UNAVAILABLE")
        if not provenance.get("dataset_fingerprint_algorithm"):
            reasons.append("DATASET_FINGERPRINT_ALGORITHM_UNAVAILABLE")
        models = provenance.get("models")
        if not isinstance(models, list) or not models:
            reasons.append("MODEL_PROVENANCE_UNAVAILABLE")
            models = []
        for trace in models:
            if not isinstance(trace, dict):
                reasons.append("MODEL_PROVENANCE_INVALID")
                continue
            for role in ("classifier", "regressor"):
                artifact = trace.get(role)
                digest = artifact.get("artifact_sha256") if isinstance(artifact, dict) else None
                if not isinstance(digest, str) or re.fullmatch(r"[a-fA-F0-9]{64}", digest) is None:
                    reasons.append(f"{role.upper()}_FINGERPRINT_UNAVAILABLE")
                if not isinstance(artifact, dict) or not artifact.get("model_version"):
                    reasons.append(f"{role.upper()}_VERSION_UNAVAILABLE")
            if not trace.get("application_version"):
                reasons.append("APPLICATION_VERSION_UNAVAILABLE")
        return sorted(set(reasons))

    @staticmethod
    def _evaluate(
        definition: Any, scenario_time: Any, predictions: dict[str, dict[str, Any]]
    ) -> ScenarioResult:
        demands = {
            key: {
                "demand": float(value["regression"]["predicted_connections_next_hour"]),
                "classification": str(value["classification"]["predicted_demand_level"]),
            }
            for key, value in predictions.items()
        }
        plan = plan_capacity(
            [
                {
                    "zone_id": zone.zone_id,
                    "demand": demands[zone.zone_id]["demand"],
                    "capacity": float(zone.capacity_connections),
                }
                for zone in definition.zones
            ],
            float(definition.resource_budget_connections),
        )
        capacities = {zone.zone_id: zone.capacity_connections for zone in definition.zones}
        zones = [
            ScenarioZoneResult(
                zone_id=row["zone_id"],
                predicted_demand_connections=row["demand"],
                predicted_demand_level=demands[row["zone_id"]]["classification"],
                assumed_capacity_connections=float(capacities[row["zone_id"]]),
                allocated_budget_connections=row["allocated_capacity"],
                effective_capacity_connections=row["effective_capacity"],
                capacity_surplus_connections=max(0.0, row["effective_capacity"] - row["demand"]),
                capacity_deficit_connections=row["overload_connections"],
                utilization_proxy_pct=row["utilization_pct"],
                constraint_violation=row["overload_connections"] > 0,
            )
            for row in plan["zones"]
        ]
        aggregate = plan["aggregate"]
        return ScenarioResult(
            name=definition.name,
            description=definition.description,
            datetime=scenario_time,
            zones=zones,
            aggregate=ScenarioAggregateResult(
                predicted_demand_connections=aggregate["estimated_demand_connections"],
                effective_capacity_connections=aggregate["effective_capacity_connections"],
                overload_connections=aggregate["overload_connections"],
                utilization_proxy_pct=aggregate["utilization_pct"],
                allocated_budget_connections=plan["allocated_connections"],
                unallocated_budget_connections=plan["remaining_budget_connections"],
                budget_constraint_satisfied=plan["allocated_connections"]
                <= plan["resource_budget_connections"] + 1e-9,
                constraint_violations=[
                    {
                        "reason_code": "PREDICTED_DEMAND_EXCEEDS_EFFECTIVE_CAPACITY",
                        "zone_id": zone.zone_id,
                    }
                    for zone in zones
                    if zone.constraint_violation
                ],
            ),
        )

    @staticmethod
    def _relative_difference(alternative: float, baseline: float) -> float | None:
        return None if baseline == 0 else (alternative - baseline) / abs(baseline) * 100

    @classmethod
    def _compare(
        cls, baseline: ScenarioResult, alternative: ScenarioResult
    ) -> AlternativeScenarioComparison:
        left_zones = {zone.zone_id: zone for zone in baseline.zones}
        right_zones = {zone.zone_id: zone for zone in alternative.zones}
        differences = []
        affected = 0
        for zone_id in sorted(left_zones):
            left, right = left_zones[zone_id], right_zones[zone_id]
            diffs = (
                right.assumed_capacity_connections - left.assumed_capacity_connections,
                right.effective_capacity_connections - left.effective_capacity_connections,
                right.capacity_deficit_connections - left.capacity_deficit_connections,
            )
            affected += int(any(abs(value) > 1e-9 for value in diffs))
            differences.append(
                ZoneScenarioDifference(
                    zone_id=zone_id,
                    predicted_demand_difference_connections=right.predicted_demand_connections
                    - left.predicted_demand_connections,
                    predicted_demand_difference_pct=cls._relative_difference(
                        right.predicted_demand_connections, left.predicted_demand_connections
                    ),
                    configured_capacity_difference_connections=diffs[0],
                    effective_capacity_difference_connections=diffs[1],
                    overload_difference_connections=diffs[2],
                    utilization_proxy_difference_pct_points=(
                        None
                        if left.utilization_proxy_pct is None or right.utilization_proxy_pct is None
                        else right.utilization_proxy_pct - left.utilization_proxy_pct
                    ),
                    assessment_status="assessable",
                    reason_codes=(
                        ["BASELINE_DEMAND_IS_ZERO"]
                        if left.predicted_demand_connections == 0
                        else []
                    ),
                )
            )
        a, b = baseline.aggregate, alternative.aggregate
        deltas = {
            "predicted_demand_difference_connections": b.predicted_demand_connections
            - a.predicted_demand_connections,
            "predicted_demand_difference_pct": cls._relative_difference(
                b.predicted_demand_connections, a.predicted_demand_connections
            ),
            "effective_capacity_difference_connections": b.effective_capacity_connections
            - a.effective_capacity_connections,
            "overload_difference_connections": b.overload_connections - a.overload_connections,
            "utilization_proxy_difference_pct_points": (
                None
                if a.utilization_proxy_pct is None or b.utilization_proxy_pct is None
                else b.utilization_proxy_pct - a.utilization_proxy_pct
            ),
            "allocated_budget_difference_connections": b.allocated_budget_connections
            - a.allocated_budget_connections,
            "unallocated_budget_difference_connections": b.unallocated_budget_connections
            - a.unallocated_budget_connections,
        }
        return AlternativeScenarioComparison(
            alternative_name=alternative.name,
            comparison_status="assessable",
            assessment_status="assessable",
            reason_codes=[],
            affected_zone_count=affected,
            baseline=baseline,
            alternative=alternative,
            aggregate_differences=deltas,
            zones=differences,
        )
