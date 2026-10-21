"""Reusable orchestration for model-backed synthetic capacity experiments."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from wifi_tunja_smart_predictor.capacity import plan_capacity
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest


class ScenarioExperimentService:
    """Connect scenario inference, capacity rules, provenance, and persistence."""

    def __init__(self, prediction_service: Any, store: Any) -> None:
        self.prediction_service = prediction_service
        self.store = store

    def run(
        self,
        *,
        name: str,
        description: str,
        when: datetime,
        zones: list[dict[str, Any]],
        resource_budget_connections: float,
    ) -> dict[str, Any]:
        estimates = []
        for zone in zones:
            prediction = self.prediction_service.predict(
                ScenarioRequest(datetime=when, zone_id=zone["zone_id"])
            ).to_dict()
            estimates.append(
                {
                    "zone_id": zone["zone_id"],
                    "demand": float(prediction["regression"]["predicted_connections_next_hour"]),
                    "capacity": float(zone["capacity_connections"]),
                    "model_version": prediction["scenario"].get("model_version"),
                    "dataset_version": prediction["scenario"].get("dataset_version"),
                    "dataset_sha256": prediction["scenario"].get("dataset_sha256"),
                    "dataset_fingerprint_algorithm": prediction["scenario"].get(
                        "dataset_fingerprint_algorithm"
                    ),
                    "traceability": prediction.get("traceability"),
                }
            )
        plan = plan_capacity(estimates, resource_budget_connections)
        configuration = {
            "name": name,
            "description": description,
            "datetime": when.isoformat(),
            "zones": zones,
            "resource_budget_connections": resource_budget_connections,
            "configuration_version": "1.0",
            "random_seed": None,
        }
        result = {
            "time_window": when.isoformat(),
            "zones": plan["zones"],
            "capacity_plan": plan,
            "provenance": {
                "data": "synthetic historical analogs and model predictions",
                "dataset_version": estimates[0].get("dataset_version") if estimates else None,
                "dataset_sha256": estimates[0].get("dataset_sha256") if estimates else None,
                "dataset_fingerprint_algorithm": (
                    estimates[0].get("dataset_fingerprint_algorithm") if estimates else None
                ),
                "models": [zone["traceability"] for zone in estimates],
            },
            "limitations": [
                "Simulation only; not an operational recommendation for a real network."
            ],
        }
        return self.store.save(name, configuration, result)
