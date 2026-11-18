"""Read persisted experiment records and issue versioned evaluation gate reports."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.experiments.ml_gates import evaluate_experiment_gates
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


class MLExperimentGateNotFoundError(LookupError):
    """One or more requested experiment records are absent."""


class MLExperimentGateService:
    """Generate read-only threshold reports from saved synthetic runs."""

    def __init__(self, store: MLExperimentStore) -> None:
        self.store = store

    def evaluate(
        self, experiment_ids: list[str], task: str, criteria: list[dict[str, Any]]
    ) -> dict[str, Any]:
        records = [self.store.get(experiment_id) for experiment_id in experiment_ids]
        if any(record is None for record in records):
            raise MLExperimentGateNotFoundError("One or more requested experiments were not found.")
        report = evaluate_experiment_gates(
            [record for record in records if record is not None], task, criteria
        )
        return {
            "report_schema_version": "1.0",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "project_version": PROJECT_VERSION,
            "data_boundary": "Synthetic ML experiments only; official MinTIC records are excluded.",
            "task": task,
            "criteria": criteria,
            **report,
            "limitations": [
                "Gate results evaluate recorded synthetic metrics only.",
                "Incompatible datasets, tasks, or evaluation windows are inconclusive.",
                "A passed gate is not a universal model ranking or real-world validation.",
                "The report is read-only and does not promote models or modify artifacts.",
            ],
        }
