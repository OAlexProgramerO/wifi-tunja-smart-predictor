"""Compatibility, metric-direction, provenance, and API tests for ML comparisons."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.experiments.ml_comparison import compare_experiment_records
from wifi_tunja_smart_predictor.experiments.ml_comparison_schemas import (
    MLExperimentComparisonRequest,
)
from wifi_tunja_smart_predictor.experiments.ml_comparison_service import (
    MLExperimentComparisonService,
    MLExperimentNotFoundError,
)
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


def _record(
    experiment_id: str,
    *,
    task: str = "classification",
    split: str = "validation",
    fingerprint: str = "a" * 64,
    metrics: dict | None = None,
) -> dict:
    target = "demand_level" if task == "classification" else "connections_next_hour"
    default_metrics = (
        {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "roc_auc": 0.75}
        if task == "classification"
        else {"mae": 10.0, "rmse": 12.0, "r2": 0.5}
    )
    return {
        "experiment_id": experiment_id,
        "created_at_utc": "2026-11-17T10:00:00+00:00",
        "updated_at_utc": "2026-11-17T10:00:05+00:00",
        "completed_at_utc": "2026-11-17T10:00:05+00:00",
        "status": "COMPLETED",
        "configuration": {
            "task": task,
            "model_candidate": "candidate_" + experiment_id,
            "evaluation_split": split,
            "hyperparameters": {"max_depth": 4},
            "feature_columns": ["hour", "weather_condition"],
            "target_column": target,
            "random_seed": 42,
        },
        "result": {
            "project_version": PROJECT_VERSION,
            "environment": {"python": "3.12.1", "scikit_learn": "1.7.0"},
            "dataset": {
                "kind": "synthetic",
                "fingerprint_algorithm": "sha256(test)",
                "fingerprint": fingerprint,
                "rows": 1000,
            },
            "temporal_split": {
                "strategy": "chronological",
                "train": {
                    "start": "2024-01-01T00:00:00",
                    "end": "2025-06-30T23:00:00",
                    "rows": 500,
                    "cutoff": "2025-06-30 23:59:59",
                },
                "validation": {
                    "start": "2025-07-01T00:00:00",
                    "end": "2025-09-30T23:00:00",
                    "rows": 100,
                    "cutoff": "2025-09-30 23:59:59",
                },
                "test": {
                    "start": "2025-10-01T00:00:00",
                    "end": "2025-12-31T23:00:00",
                    "rows": 100,
                    "cutoff": None,
                },
                "evaluation_partition": split,
            },
            "model": {
                "task": task,
                "candidate": "candidate_" + experiment_id,
                "hyperparameters": {"max_depth": 4},
                "random_seed": 42,
                "artifact_persisted": False,
                "artifact_fingerprint": None,
            },
            "evaluation": {
                "rows": 100,
                "metrics": metrics if metrics is not None else default_metrics,
            },
        },
    }


def test_compatible_classification_reports_improvement_regression_and_tie() -> None:
    base = _record("a")
    candidate = _record(
        "b",
        metrics={"accuracy": 0.9, "precision": 0.6, "recall": 0.6, "f1": 0.65, "roc_auc": 0.8},
    )
    pair = compare_experiment_records([base, candidate])["comparisons"][0]
    assert pair["status"] == "COMPARABLE"
    assessments = {row["metric"]: row["assessment"] for row in pair["metric_results"]}
    assert assessments == {
        "accuracy": "IMPROVEMENT",
        "f1": "TIE",
        "precision": "REGRESSION",
        "recall": "TIE",
        "roc_auc": "IMPROVEMENT",
    }


def test_regression_metric_directions_are_respected() -> None:
    base = _record("a", task="regression")
    candidate = _record("b", task="regression", metrics={"mae": 8.0, "rmse": 15.0, "r2": 0.6})
    pair = compare_experiment_records([base, candidate])["comparisons"][0]
    assessments = {row["metric"]: row["assessment"] for row in pair["metric_results"]}
    assert assessments == {"mae": "IMPROVEMENT", "r2": "IMPROVEMENT", "rmse": "REGRESSION"}


@pytest.mark.parametrize(
    ("mutator", "expected_reason"),
    [
        (
            lambda record: record["result"]["dataset"].update(fingerprint="b" * 64),
            "INCOMPATIBLE_DATASET_FINGERPRINTS",
        ),
        (
            lambda record: record["configuration"].update(evaluation_split="test"),
            "INCOMPATIBLE_EVALUATION_SPLITS",
        ),
        (lambda record: record["configuration"].update(random_seed=9), "INCOMPATIBLE_RANDOM_SEED"),
        (
            lambda record: record["configuration"].update(
                feature_columns=["connections_next_hour"]
            ),
            "TARGET_OR_FUTURE_FEATURE_LEAKAGE",
        ),
        (
            lambda record: record["result"]["temporal_split"]["train"].update(
                end="2025-07-02T00:00:00"
            ),
            "INVALID_CHRONOLOGICAL_WINDOWS",
        ),
    ],
)
def test_incompatible_records_suppress_metric_deltas(mutator, expected_reason: str) -> None:
    left, right = _record("left"), _record("right")
    mutator(right)
    pair = compare_experiment_records([left, right])["comparisons"][0]
    assert pair["status"] == "INCOMPATIBLE"
    assert pair["metric_results"] == []
    assert expected_reason in pair["reasons"]


def test_task_types_and_runtime_provenance_must_match() -> None:
    mixed = compare_experiment_records(
        [_record("classifier"), _record("regressor", task="regression")]
    )["comparisons"][0]
    assert mixed["status"] == "INCOMPATIBLE"
    assert "INCOMPATIBLE_TASK_TYPES" in mixed["reasons"]
    changed = _record("changed")
    changed["result"]["environment"]["scikit_learn"] = "different"
    pair = compare_experiment_records([_record("base"), changed])["comparisons"][0]
    assert "INCOMPATIBLE_ENVIRONMENT" in pair["reasons"]


def test_candidate_configuration_is_reported_but_may_differ() -> None:
    left, right = _record("one"), _record("two")
    right["configuration"]["model_candidate"] = "another_candidate"
    right["configuration"]["hyperparameters"] = {"max_depth": 7}
    right["result"]["model"]["candidate"] = "another_candidate"
    right["result"]["model"]["hyperparameters"] = {"max_depth": 7}
    pair = compare_experiment_records([left, right])["comparisons"][0]
    assert pair["status"] == "COMPARABLE"
    assert pair["right_model"] == {
        "candidate": "another_candidate",
        "hyperparameters": {"max_depth": 7},
    }


def test_missing_fingerprint_or_inconsistent_model_metadata_is_incompatible() -> None:
    left, right = _record("left"), _record("right")
    left["result"]["dataset"]["fingerprint"] = None
    pair = compare_experiment_records([left, right])["comparisons"][0]
    assert "SYNTHETIC_DATASET_PROVENANCE_INCOMPLETE" in pair["reasons"]
    broken = _record("broken")
    broken["result"]["model"]["random_seed"] = 9
    pair = compare_experiment_records([_record("base"), broken])["comparisons"][0]
    assert "MODEL_CONFIGURATION_INCONSISTENT" in pair["reasons"]
    wrong_target = _record("wrong-target")
    wrong_target["configuration"]["target_column"] = "connections_next_hour"
    wrong_target["result"]["model"]["task"] = "classification"
    pair = compare_experiment_records([_record("base2"), wrong_target])["comparisons"][0]
    assert "TASK_TARGET_MISMATCH" in pair["reasons"]


def test_malformed_task_and_split_metadata_returns_incompatible_report() -> None:
    malformed = _record("malformed")
    malformed["configuration"]["task"] = ["classification"]
    malformed["configuration"]["evaluation_split"] = ["validation"]
    pair = compare_experiment_records([_record("valid"), malformed])["comparisons"][0]
    assert pair["status"] == "INCOMPATIBLE"
    assert "TASK_METADATA_UNAVAILABLE" in pair["reasons"]
    assert "EVALUATION_SPLIT_UNAVAILABLE" in pair["reasons"]


def test_missing_metrics_produce_partial_results_and_provenance_is_retained() -> None:
    report = compare_experiment_records(
        [
            _record("base", metrics={"accuracy": 0.8, "f1": None}),
            _record("other", metrics={"accuracy": 0.9}),
        ]
    )
    pair = report["comparisons"][0]
    assert pair["status"] == "PARTIAL"
    assert pair["metric_results"][0]["metric"] == "accuracy"
    assert "roc_auc" in {item["metric"] for item in pair["unavailable_metrics"]}
    assert report["experiments"][0]["dataset"]["fingerprint"] == "a" * 64
    assert report["experiments"][0]["completed_at_utc"].endswith("+00:00")
    assert "universal best" in report["interpretation"]
    empty_metrics = compare_experiment_records(
        [_record("empty-a", metrics={}), _record("empty-b", metrics={})]
    )["comparisons"][0]
    assert empty_metrics["status"] == "PARTIAL"
    assert "NO_SHARED_VALID_METRICS" in empty_metrics["reasons"]


def _persist(store: MLExperimentStore, record: dict) -> str:
    created, _ = store.create_pending(record["configuration"], "key-" + record["experiment_id"])
    store.transition(created["experiment_id"], "RUNNING")
    store.transition(
        created["experiment_id"],
        "COMPLETED",
        result=record["result"],
    )
    return created["experiment_id"]


def test_service_generates_versioned_provenance_report_without_writes(tmp_path: Path) -> None:
    store = MLExperimentStore(tmp_path / "experiments.sqlite3")
    base = _record("run-a")
    other = _record("run-b")
    base_id = _persist(store, base)
    other_id = _persist(store, other)
    service = MLExperimentComparisonService(store)
    before = store.list_page(limit=100, offset=0)[1]
    report = service.compare([base_id, other_id])
    after = store.list_page(limit=100, offset=0)[1]
    assert before == after == 2
    assert report["report_schema_version"] == "1.0"
    assert report["project_version"] == PROJECT_VERSION
    assert report["overall_status"] == "COMPARABLE"
    assert len(report["comparisons"]) == 1
    assert report["data_boundary"].startswith("Synthetic")


def test_service_missing_ids_and_request_validation(tmp_path: Path) -> None:
    store = MLExperimentStore(tmp_path / "experiments.sqlite3")
    with pytest.raises(MLExperimentNotFoundError):
        MLExperimentComparisonService(store).compare(["missing-a", "missing-b"])
    with pytest.raises(ValueError, match="unique"):
        MLExperimentComparisonRequest(experiment_ids=["same", "same"])


def test_comparison_api_returns_report_and_safe_errors(tmp_path: Path, monkeypatch) -> None:
    import api.routes.ml_experiments as routes
    from wifi_tunja_smart_predictor.experiments.ml_orchestrator import MLExperimentOrchestrator

    store = MLExperimentStore(tmp_path / "experiments.sqlite3")
    left, right = _record("api-left"), _record("api-right")
    left_id = _persist(store, left)
    right_id = _persist(store, right)
    monkeypatch.setattr(
        routes,
        "_orchestrator",
        lambda: MLExperimentOrchestrator(store, runner=lambda _: {}),
    )
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    response = client.post(
        "/ml-experiments/compare",
        json={"experiment_ids": [left_id, right_id]},
    )
    assert response.status_code == 200
    assert response.json()["overall_status"] == "COMPARABLE"
    assert (
        client.post(
            "/ml-experiments/compare", json={"experiment_ids": ["missing-a", "missing-b"]}
        ).status_code
        == 404
    )
    assert (
        client.post(
            "/ml-experiments/compare", json={"experiment_ids": ["same", "same"]}
        ).status_code
        == 422
    )
