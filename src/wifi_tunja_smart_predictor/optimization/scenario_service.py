"""Model-backed orchestration for the pure allocation optimizer."""

from __future__ import annotations

from typing import Any

from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.optimization.schemas import (
    AllocationOptimizationRequest,
    AllocationOptimizationResult,
)
from wifi_tunja_smart_predictor.optimization.service import AllocationOptimizationService
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.simulation.service import WhatIfSimulationService


class ScenarioOptimizationService:
    """Fetch model predictions once per zone, then call the UI/API-independent optimizer."""

    def __init__(self, prediction_service: Any) -> None:
        self.prediction_service = prediction_service
        self.optimizer = AllocationOptimizationService()

    def run(self, request: AllocationOptimizationRequest) -> AllocationOptimizationResult:
        predictions = {
            zone.zone_id: self.prediction_service.predict(
                ScenarioRequest(datetime=request.datetime, zone_id=zone.zone_id)
            ).to_dict()
            for zone in sorted(request.zones, key=lambda item: item.zone_id)
        }
        demand = {
            zone_id: prediction["regression"]["predicted_connections_next_hour"]
            for zone_id, prediction in predictions.items()
        }
        result = self.optimizer.optimize(request, demand)
        provenance = WhatIfSimulationService._provenance(predictions)
        provenance_reasons = WhatIfSimulationService._provenance_reasons(provenance)
        return result.model_copy(
            update={
                "assessment_status": "assessable" if not provenance_reasons else "not_assessable",
                "reason_codes": sorted(set(result.reason_codes + provenance_reasons)),
                "provenance": provenance,
                "limitations": [
                    *result.limitations,
                    "Model predictions are conditional estimates from the recorded synthetic dataset and exact fitted artifacts.",
                    f"The application package version recorded for this run is {PROJECT_VERSION}.",
                ],
            }
        )
