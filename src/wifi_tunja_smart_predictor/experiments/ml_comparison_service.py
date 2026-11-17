"""Read-only report generation for persisted synthetic ML experiment comparisons."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.experiments.ml_comparison import compare_experiment_records
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


class MLExperimentNotFoundError(LookupError):
    """One or more requested saved experiments are unavailable."""


class MLExperimentComparisonService:
    """Read persisted runs and generate a versioned pairwise compatibility report."""

    def __init__(self, store: MLExperimentStore) -> None:
        self.store = store

    def compare(self, experiment_ids: list[str]) -> dict[str, Any]:
        records = [self.store.get(experiment_id) for experiment_id in experiment_ids]
        if any(record is None for record in records):
            raise MLExperimentNotFoundError("One or more requested experiments were not found.")
        report = compare_experiment_records([record for record in records if record is not None])
        return {
            "report_schema_version": "1.0",
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "project_version": PROJECT_VERSION,
            "data_boundary": "Synthetic ML experiments only; official MinTIC records are excluded.",
            **report,
            "limitations": [
                "Comparisons describe recorded synthetic evaluation runs only.",
                "Comparisons do not select a universal best model.",
                "Artifact fingerprints identify candidate outputs only when persisted; they are not compatibility keys.",
                "Incompatible fingerprints, evaluation windows, or task contracts suppress metric deltas.",
            ],
        }
