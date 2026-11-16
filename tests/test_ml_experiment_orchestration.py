"""Lifecycle, reproducibility, API, and leakage guards for ML experiments."""

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.config import MODEL_INPUT_COLUMNS, TARGET_COLUMNS
from wifi_tunja_smart_predictor.experiments.ml_config import normalize_configuration
from wifi_tunja_smart_predictor.experiments.ml_orchestrator import MLExperimentOrchestrator
from wifi_tunja_smart_predictor.experiments.ml_schemas import MLExperimentCreate
from wifi_tunja_smart_predictor.experiments.ml_service import (
    execute_experiment,
)
from wifi_tunja_smart_predictor.experiments.ml_store import (
    IdempotencyConflict,
    MLExperimentStore,
)


def _payload(**overrides: object) -> MLExperimentCreate:
    values = {
        "task": "classification",
        "model_candidate": "decision_tree",
        "evaluation_split": "validation",
        "hyperparameters": {"max_depth": 4},
    }
    values.update(overrides)
    return MLExperimentCreate(**values)


def test_configuration_validates_task_candidate_and_bounded_parameters() -> None:
    config = normalize_configuration(_payload())
    assert config["task"] == "classification"
    assert config["hyperparameters"]["max_depth"] == 4
    assert not set(config["feature_columns"]) & set(TARGET_COLUMNS)
    with pytest.raises(ValueError, match="not available"):
        normalize_configuration(_payload(task="regression"))
    with pytest.raises(ValueError, match="allowed range"):
        normalize_configuration(_payload(hyperparameters={"max_depth": 1000}))
    with pytest.raises(ValueError, match="not allowed"):
        normalize_configuration(_payload(hyperparameters={"random_state": 99}))


def test_regression_configuration_uses_separate_candidate_contract() -> None:
    config = normalize_configuration(
        _payload(task="regression", model_candidate="random_forest_regressor")
    )
    assert config["target_column"] == "connections_next_hour"
    assert "n_estimators" in config["hyperparameters"]
    with pytest.raises(ValueError, match="not available"):
        normalize_configuration(_payload(task="regression", model_candidate="decision_tree"))


def test_lifecycle_persists_states_results_and_utc_times(tmp_path: Path) -> None:
    store = MLExperimentStore(tmp_path / "experiments.sqlite3")

    def runner(_: dict) -> dict:
        active = store.list_page(limit=1, offset=0)[0][0]
        assert active["status"] == "RUNNING"
        return {"metrics": {"f1": 0.5}}

    orchestrator = MLExperimentOrchestrator(store, runner=runner)
    record = orchestrator.create(_payload(), "request-key-001")
    assert record["status"] == "COMPLETED"
    assert record["result"]["metrics"]["f1"] == 0.5
    assert pd.Timestamp(record["created_at_utc"]).tzinfo is not None
    assert pd.Timestamp(record["updated_at_utc"]).tzinfo == UTC
    assert pd.Timestamp(record["completed_at_utc"]).tzinfo == UTC
    assert orchestrator.get(record["experiment_id"]) == record
    report = orchestrator.report(record["experiment_id"])
    assert report["experiment"]["configuration"]["task"] == "classification"
    assert "Synthetic ML only" in report["data_boundary"]


def test_failures_are_persisted_with_sanitized_summary(tmp_path: Path) -> None:
    def fail(_: dict) -> dict:
        raise RuntimeError("secret path and raw data must never be saved")

    orchestrator = MLExperimentOrchestrator(
        MLExperimentStore(tmp_path / "experiments.sqlite3"), runner=fail
    )
    record = orchestrator.create(_payload(), "request-key-failure")
    assert record["status"] == "FAILED"
    assert record["error"]["error_type"] == "RuntimeError"
    assert "secret" not in str(record)


def test_idempotency_replay_conflict_and_pagination(tmp_path: Path) -> None:
    store = MLExperimentStore(tmp_path / "experiments.sqlite3")
    orchestrator = MLExperimentOrchestrator(store, runner=lambda _: {"ok": True})
    first = orchestrator.create(_payload(), "request-key-001")
    replay = orchestrator.create(_payload(), "request-key-001")
    assert replay["experiment_id"] == first["experiment_id"]
    assert replay["replayed"] is True
    with pytest.raises(IdempotencyConflict):
        orchestrator.create(_payload(evaluation_split="test"), "request-key-001")
    orchestrator.create(_payload(model_candidate="knn", hyperparameters={}), "request-key-002")
    page, total = orchestrator.list_page(limit=1, offset=0)
    assert total == 2
    assert len(page) == 1
    assert orchestrator.list_page(limit=1, offset=2)[0] == []


def test_runner_uses_only_configured_features_and_records_dataset_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wifi_tunja_smart_predictor.experiments.ml_service as service

    frame = make_mini_frame(82)
    timestamps = list(pd.date_range("2025-06-29", periods=72, freq="h")) + list(
        pd.date_range("2025-10-01", periods=10, freq="h")
    )
    frame["timestamp"] = timestamps
    monkeypatch.setattr(service, "_load_synthetic_dataset", lambda: frame)
    captured: dict[str, list[str]] = {}

    class Pipeline:
        classes_ = ["LOW", "HIGH"]

        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            captured["features"] = list(features.columns)
            captured["target"] = [target.name]

        def predict(self, features: pd.DataFrame) -> list[str]:
            return ["HIGH"] * len(features)

        def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
            return [[0.1, 0.9]] * len(features)

    monkeypatch.setattr(service, "build_classifier_pipeline", lambda *args, **kwargs: Pipeline())
    config = normalize_configuration(_payload(evaluation_split="test"))
    result = execute_experiment(config)
    assert captured["features"] == MODEL_INPUT_COLUMNS
    assert not set(captured["features"]) & set(TARGET_COLUMNS)
    assert captured["target"] == ["demand_level"]
    assert result["dataset"]["kind"] == "synthetic"
    assert len(result["dataset"]["fingerprint"]) == 64
    assert result["temporal_split"]["evaluation_partition"] == "test"
    assert result["model"]["artifact_fingerprint"] is None


def test_regression_runner_keeps_target_out_of_features(monkeypatch: pytest.MonkeyPatch) -> None:
    import wifi_tunja_smart_predictor.experiments.ml_service as service

    frame = make_mini_frame(82)
    frame["timestamp"] = list(pd.date_range("2025-06-29", periods=72, freq="h")) + list(
        pd.date_range("2025-10-01", periods=10, freq="h")
    )
    monkeypatch.setattr(service, "_load_synthetic_dataset", lambda: frame)
    captured: dict[str, list[str]] = {}

    class Regressor:
        def fit(self, features: pd.DataFrame, target: pd.Series) -> None:
            captured["features"] = list(features.columns)
            captured["target"] = [target.name]

        def predict(self, features: pd.DataFrame) -> list[float]:
            return [100.0] * len(features)

    monkeypatch.setattr(service, "build_classifier_pipeline", lambda *args, **kwargs: Regressor())
    config = normalize_configuration(
        _payload(
            task="regression",
            model_candidate="random_forest_regressor",
            evaluation_split="test",
        )
    )
    result = execute_experiment(config)
    assert captured["features"] == MODEL_INPUT_COLUMNS
    assert captured["target"] == ["connections_next_hour"]
    assert set(result["evaluation"]["metrics"]) == {"mae", "rmse", "r2"}


@pytest.mark.parametrize(
    ("task", "candidate", "hyperparameters"),
    [
        ("classification", "decision_tree", {"max_depth": 3}),
        ("regression", "random_forest_regressor", {"n_estimators": 10, "max_depth": 4}),
    ],
)
def test_real_candidate_fit_completes_with_synthetic_chronological_data(
    task: str,
    candidate: str,
    hyperparameters: dict[str, int],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import wifi_tunja_smart_predictor.experiments.ml_service as service

    frame = make_mini_frame(82)
    frame["timestamp"] = list(pd.date_range("2025-06-29", periods=72, freq="h")) + list(
        pd.date_range("2025-10-01", periods=10, freq="h")
    )
    monkeypatch.setattr(service, "_load_synthetic_dataset", lambda: frame)
    payload = _payload(
        task=task,
        model_candidate=candidate,
        evaluation_split="test",
        hyperparameters=hyperparameters,
    )
    orchestrator = MLExperimentOrchestrator(
        MLExperimentStore(tmp_path / f"{task}.sqlite3")
    )
    record = orchestrator.create(payload, f"trained-run-{task}")
    assert record["status"] == "COMPLETED"
    assert record["result"]["dataset"]["kind"] == "synthetic"
    assert record["result"]["evaluation"]["rows"] == 10
    assert record["result"]["model"]["artifact_persisted"] is False


def test_api_create_retrieve_report_and_pagination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.ml_experiments as routes

    orchestrator = MLExperimentOrchestrator(
        MLExperimentStore(tmp_path / "experiments.sqlite3"), runner=lambda _: {"score": 1.0}
    )
    monkeypatch.setattr(routes, "_orchestrator", lambda: orchestrator)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    response = client.post(
        "/ml-experiments",
        json={
            "task": "classification",
            "model_candidate": "decision_tree",
            "evaluation_split": "validation",
            "hyperparameters": {"max_depth": 3},
        },
        headers={"X-Idempotency-Key": "api-request-0001"},
    )
    assert response.status_code == 201
    record = response.json()
    experiment_id = record["experiment_id"]
    assert client.get(f"/ml-experiments/{experiment_id}").json()["status"] == "COMPLETED"
    assert (
        client.get(f"/ml-experiments/{experiment_id}/report").json()["experiment"]["experiment_id"]
        == experiment_id
    )
    assert client.get("/ml-experiments?limit=1&offset=0").json()["total_count"] == 1
    assert client.get("/ml-experiments?limit=101").status_code == 422
    assert client.get("/ml-experiments/missing").status_code == 404


def test_api_rejects_invalid_task_contract_and_idempotency_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.ml_experiments as routes

    orchestrator = MLExperimentOrchestrator(
        MLExperimentStore(tmp_path / "experiments.sqlite3"), runner=lambda _: {"ok": True}
    )
    monkeypatch.setattr(routes, "_orchestrator", lambda: orchestrator)
    app = FastAPI()
    app.include_router(routes.router)
    client = TestClient(app)
    headers = {"X-Idempotency-Key": "api-request-0002"}
    payload = {
        "task": "classification",
        "model_candidate": "decision_tree",
        "hyperparameters": {},
    }
    created = client.post("/ml-experiments", json=payload, headers=headers)
    assert created.status_code == 201
    replay = client.post("/ml-experiments", json=payload, headers=headers)
    assert replay.status_code == 200
    assert replay.json()["experiment_id"] == created.json()["experiment_id"]
    changed = {**payload, "evaluation_split": "test"}
    assert client.post("/ml-experiments", json=changed, headers=headers).status_code == 409
    invalid = {**payload, "task": "regression"}
    assert (
        client.post(
            "/ml-experiments", json=invalid, headers={"X-Idempotency-Key": "api-request-0003"}
        ).status_code
        == 422
    )


def test_dashboard_orchestration_page_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    from app import dashboard

    def fake_request(
        path: str, payload: dict | None = None, *, headers: dict | None = None
    ) -> dict:
        if path == "/ml-experiments?limit=20&offset=0":
            return {"items": [], "limit": 20, "offset": 0, "total_count": 0}
        return {}

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    app = AppTest.from_string(
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment Orchestration', 'MLExperiments')\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any(
        "All experiment data and metrics are synthetic" in item.value for item in app.warning
    )
    assert any(item.value == "No ML experiment runs have been recorded yet." for item in app.info)
    assert any(item.label == "Run synthetic experiment" for item in app.button)
