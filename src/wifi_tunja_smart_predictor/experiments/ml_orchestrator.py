"""Experiment lifecycle orchestration kept independent from HTTP and UI code."""

from __future__ import annotations

from typing import Any, Callable

from wifi_tunja_smart_predictor.experiments.ml_config import normalize_configuration
from wifi_tunja_smart_predictor.experiments.ml_schemas import MLExperimentCreate
from wifi_tunja_smart_predictor.experiments.ml_service import (
    execute_experiment,
)
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


class MLExperimentOrchestrator:
    """Validate, record, execute, and report one deterministic synthetic ML run."""

    def __init__(
        self,
        store: MLExperimentStore,
        runner: Callable[[dict[str, Any]], dict[str, Any]] = execute_experiment,
    ) -> None:
        self.store = store
        self.runner = runner

    def create(self, payload: MLExperimentCreate, idempotency_key: str) -> dict[str, Any]:
        configuration = normalize_configuration(payload)
        record, should_execute = self.store.create_pending(configuration, idempotency_key)
        if not should_execute:
            return record
        self.store.transition(record["experiment_id"], "RUNNING")
        try:
            result = self.runner(configuration)
        except Exception as exc:
            # Persist only an exception class and generic summary; never exception text.
            return self.store.transition(
                record["experiment_id"],
                "FAILED",
                error={
                    "code": "EXECUTION_FAILED",
                    "error_type": type(exc).__name__[:80],
                    "message": "Experiment execution failed; sensitive details were omitted.",
                },
            )
        return self.store.transition(record["experiment_id"], "COMPLETED", result=result)

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        return self.store.get(experiment_id)

    def list_page(self, *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        return self.store.list_page(limit=limit, offset=offset)

    def report(self, experiment_id: str) -> dict[str, Any] | None:
        record = self.get(experiment_id)
        if record is None:
            return None
        return {
            "report_schema_version": "1.0",
            "experiment": record,
            "data_boundary": "Synthetic ML only; official MinTIC data is excluded.",
        }
