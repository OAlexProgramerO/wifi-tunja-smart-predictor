"""Threshold, compatibility, provenance, and API tests for experiment gates."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import ml_experiments
from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.experiments.ml_gates import evaluate_experiment_gates
from wifi_tunja_smart_predictor.experiments.ml_gates_schemas import MLExperimentGateRequest
from wifi_tunja_smart_predictor.experiments.ml_gates_service import MLExperimentGateService
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


def _record(experiment_id: str, *, fingerprint: str = "a" * 64, task="classification") -> dict:
    target = "demand_level" if task == "classification" else "connections_next_hour"
    metrics = (
        {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.7, "roc_auc": 0.75}
        if task == "classification"
        else {"mae": 10.0, "rmse": 12.0, "r2": 0.5}
    )
    return {
        "experiment_id": experiment_id,
        "created_at_utc": "2026-11-17T10:00:00+00:00",
        "completed_at_utc": "2026-11-17T10:00:05+00:00",
        "status": "COMPLETED",
        "configuration": {
            "task": task,
            "model_candidate": "candidate",
            "evaluation_split": "validation",
            "hyperparameters": {},
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
                },
                "validation": {
                    "start": "2025-07-01T00:00:00",
                    "end": "2025-09-30T23:00:00",
                    "rows": 100,
                },
                "test": {"start": "2025-10-01T00:00:00", "end": "2025-12-31T23:00:00", "rows": 100},
                "evaluation_partition": "validation",
            },
            "model": {
                "task": task,
                "candidate": "candidate",
                "hyperparameters": {},
                "random_seed": 42,
            },
            "evaluation": {"rows": 100, "metrics": metrics},
        },
    }


def _criterion(metric="f1", operator="gte", threshold=0.7):
    return {"metric": metric, "operator": operator, "threshold": threshold}


def test_gate_threshold_boundary_passes_and_report_has_provenance():
    report = evaluate_experiment_gates(
        [_record("one"), _record("two")], "classification", [_criterion()]
    )
    assert [row["status"] for row in report["results"]] == ["PASSED", "PASSED"]
    assert report["results"][0]["criteria"][0]["value"] == 0.7
    assert report["results"][0]["provenance"]["dataset_fingerprint"] == "a" * 64
    assert report["compatibility"]["status"] == "COMPATIBLE"


def test_gate_failure_and_missing_metric_are_distinguished():
    first = _record("one")
    first["result"]["evaluation"]["metrics"]["f1"] = 0.69
    second = _record("two")
    second["result"]["evaluation"]["metrics"].pop("f1")
    results = evaluate_experiment_gates([first, second], "classification", [_criterion()])[
        "results"
    ]
    assert [item["status"] for item in results] == ["FAILED", "INCONCLUSIVE"]
    assert results[1]["criteria"][0]["value"] is None


def test_nonfinite_metric_is_inconclusive_and_sanitized():
    first = _record("one")
    first["result"]["evaluation"]["metrics"]["f1"] = float("nan")
    report = evaluate_experiment_gates([first, _record("two")], "classification", [_criterion()])
    assert report["results"][0]["status"] == "INCONCLUSIVE"
    assert report["results"][0]["metrics"]["f1"] is None


@pytest.mark.parametrize(
    "records,task",
    [
        ([_record("a"), _record("b", fingerprint="b" * 64)], "classification"),
        ([_record("a"), _record("b", task="regression")], "classification"),
    ],
)
def test_incompatible_dataset_or_task_is_inconclusive(records, task):
    report = evaluate_experiment_gates(records, task, [_criterion()])
    assert all(item["status"] == "INCONCLUSIVE" for item in report["results"])
    assert report["compatibility"]["status"] == "INCOMPATIBLE"


def test_regression_lte_boundary_passes():
    report = evaluate_experiment_gates(
        [_record("a", task="regression"), _record("b", task="regression")],
        "regression",
        [_criterion("mae", "lte", 10.0)],
    )
    assert all(item["status"] == "PASSED" for item in report["results"])


def test_gate_request_rejects_wrong_direction_and_nonfinite_threshold():
    with pytest.raises(ValueError):
        MLExperimentGateRequest(
            experiment_ids=["a", "b"], task="regression", criteria=[_criterion("mae", "gte")]
        )
    with pytest.raises(ValueError):
        MLExperimentGateRequest(
            experiment_ids=["a", "b"],
            task="regression",
            criteria=[_criterion("mae", "lte", float("inf"))],
        )


def _persist(store: MLExperimentStore, record: dict) -> str:
    pending, _ = store.create_pending(record["configuration"], f"key-{record['experiment_id']}")
    store.transition(pending["experiment_id"], "RUNNING")
    store.transition(pending["experiment_id"], "COMPLETED", result=record["result"])
    return pending["experiment_id"]


def test_gate_service_is_read_only_and_report_is_versioned(tmp_path):
    store = MLExperimentStore(tmp_path / "experiments.sqlite")
    identifiers = [_persist(store, _record(name)) for name in ("one", "two")]
    before = store.list_page(limit=10, offset=0)[1]
    report = MLExperimentGateService(store).evaluate(identifiers, "classification", [_criterion()])
    assert report["report_schema_version"] == "1.0"
    assert report["data_boundary"].startswith("Synthetic ML")
    assert store.list_page(limit=10, offset=0)[1] == before


def test_gates_api_returns_versioned_report(monkeypatch, tmp_path):
    store = MLExperimentStore(tmp_path / "experiments.sqlite")
    identifiers = [_persist(store, _record(name)) for name in ("one", "two")]
    monkeypatch.setattr(ml_experiments, "_orchestrator", lambda: SimpleNamespace(store=store))
    app = FastAPI()
    app.include_router(ml_experiments.router)
    response = TestClient(app).post(
        "/ml-experiments/gates",
        json={
            "experiment_ids": identifiers,
            "task": "classification",
            "criteria": [_criterion()],
        },
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "PASSED"
