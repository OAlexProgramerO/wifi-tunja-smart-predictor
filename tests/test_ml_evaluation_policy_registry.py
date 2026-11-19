"""Policy validation, version persistence, retrieval, and evaluation tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import ml_experiments
from tests.test_ml_experiment_gates import _record
from wifi_tunja_smart_predictor.experiments.ml_policy_schemas import MLEvaluationPolicyCreate
from wifi_tunja_smart_predictor.experiments.ml_policy_service import (
    MLEvaluationPolicyNotFoundError,
    MLEvaluationPolicyService,
)
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


def _criteria(threshold: float = 0.7) -> dict[str, list[dict]]:
    return {
        "classification": [{"metric": "f1", "operator": "gte", "threshold": threshold}],
        "regression": [],
    }


def _save_experiment(store: MLExperimentStore, record: dict) -> str:
    pending, _ = store.create_pending(
        record["configuration"], f"idempotency-{record['experiment_id']}"
    )
    store.transition(pending["experiment_id"], "RUNNING")
    store.transition(pending["experiment_id"], "COMPLETED", result=record["result"])
    return pending["experiment_id"]


def _create_policy(service: MLEvaluationPolicyService, policy_id=None, threshold=0.7):
    return service.create_version(
        policy_id=policy_id,
        name="Synthetic classifier baseline",
        description="Minimum F1 criterion for validation runs.",
        tags=["synthetic", "classification"],
        criteria=_criteria(threshold),
    )


def test_policy_schema_accepts_classification_and_regression_criteria():
    policy = MLEvaluationPolicyCreate(
        name="Both tasks",
        classification_criteria=[{"metric": "f1", "operator": "gte", "threshold": 0.8}],
        regression_criteria=[{"metric": "mae", "operator": "lte", "threshold": 5}],
    )
    assert policy.classification_criteria[0].metric == "f1"
    assert policy.regression_criteria[0].metric == "mae"


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "empty"},
        {
            "name": "bad metric",
            "classification_criteria": [{"metric": "mae", "operator": "lte", "threshold": 1}],
        },
        {
            "name": "bad direction",
            "regression_criteria": [{"metric": "mae", "operator": "gte", "threshold": 1}],
        },
        {
            "name": "duplicate",
            "classification_criteria": [
                {"metric": "f1", "operator": "gte", "threshold": 0.7},
                {"metric": "f1", "operator": "gte", "threshold": 0.8},
            ],
        },
    ],
)
def test_policy_schema_rejects_missing_or_invalid_criteria(payload):
    with pytest.raises(ValueError):
        MLEvaluationPolicyCreate(**payload)


def test_policy_versions_are_immutable_and_latest_policy_is_listed(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    first = _create_policy(service)
    second = _create_policy(service, first["policy_id"], threshold=0.8)

    assert first["version"] == 1
    assert second["version"] == 2
    assert store.get_policy(first["policy_id"], 1)["criteria"] == _criteria(0.7)
    assert store.get_policy(first["policy_id"])["version"] == 2
    versions, total_versions = store.list_policy_versions(first["policy_id"])
    latest, total_policies = store.list_policies(limit=10, offset=0)
    assert [item["version"] for item in versions] == [2, 1]
    assert total_versions == 2
    assert total_policies == 1
    assert latest[0]["version"] == 2


def test_unknown_policy_cannot_be_updated_in_place(tmp_path):
    service = MLEvaluationPolicyService(MLExperimentStore(tmp_path / "ml.sqlite"))
    with pytest.raises(MLEvaluationPolicyNotFoundError):
        _create_policy(service, "unknown-policy")


def test_policy_evaluation_reuses_gates_and_reports_exact_policy_provenance(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    policy = _create_policy(service)
    experiment_id = _save_experiment(store, _record("run-1"))

    report = service.evaluate(policy["policy_id"], policy["version"], experiment_id)

    assert report["status"] == "PASSED"
    assert report["policy"]["policy_id"] == policy["policy_id"]
    assert report["policy"]["version"] == 1
    assert report["criteria"][0]["threshold"] == 0.7
    assert report["experiment"]["dataset"]["fingerprint"] == "a" * 64
    assert report["experiment"]["evaluation_window"]["start"]
    assert "synthetic" in report["data_boundary"].lower()


def test_policy_task_mismatch_and_missing_metric_are_inconclusive(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    policy = _create_policy(service)
    regression_id = _save_experiment(store, _record("regression", task="regression"))
    incompatible = service.evaluate(policy["policy_id"], 1, regression_id)
    assert incompatible["status"] == "INCONCLUSIVE"
    assert "POLICY_TASK_UNSUPPORTED" in incompatible["reasons"]

    missing_record = _record("missing")
    missing_record["result"]["evaluation"]["metrics"].pop("f1")
    missing_id = _save_experiment(store, missing_record)
    missing = service.evaluate(policy["policy_id"], 1, missing_id)
    assert missing["status"] == "INCONCLUSIVE"
    assert "MISSING_REQUIRED_METRIC_F1" in missing["reasons"]


def test_policy_api_creates_lists_retrieves_versions_and_evaluates(monkeypatch, tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    experiment_id = _save_experiment(store, _record("api-run"))
    app = FastAPI()
    app.include_router(ml_experiments.router)
    monkeypatch.setattr(ml_experiments, "_orchestrator", lambda: SimpleNamespace(store=store))
    client = TestClient(app)

    create = client.post(
        "/ml-experiments/policies",
        json={
            "name": "API policy",
            "description": "Created by API test",
            "tags": ["test"],
            "classification_criteria": _criteria()["classification"],
        },
    )
    assert create.status_code == 201
    policy = create.json()
    policy_id = policy["policy_id"]

    updated = client.post(
        "/ml-experiments/policies",
        json={
            "policy_id": policy_id,
            "name": "API policy v2",
            "classification_criteria": [{"metric": "f1", "operator": "gte", "threshold": 0.8}],
        },
    )
    assert updated.status_code == 201
    assert updated.json()["version"] == 2
    assert client.get(f"/ml-experiments/policies/{policy_id}?version=1").json()["version"] == 1
    assert client.get(f"/ml-experiments/policies/{policy_id}/versions").json()["total_count"] == 2
    assert client.get("/ml-experiments/policies").json()["items"][0]["version"] == 2

    evaluation = client.post(
        f"/ml-experiments/policies/{policy_id}/versions/1/evaluate",
        json={"experiment_id": experiment_id},
    )
    assert evaluation.status_code == 200
    assert evaluation.json()["policy"]["version"] == 1
    assert evaluation.json()["status"] == "PASSED"


def test_dashboard_browses_policy_versions_and_runs_selected_version(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import dashboard

    requests = []
    records = [
        {
            "experiment_id": f"run-{index}",
            "created_at_utc": "2026-11-18T10:00:00+00:00",
            "status": "COMPLETED",
            "configuration": {
                "task": "classification",
                "model_candidate": f"candidate-{index}",
                "evaluation_split": "validation",
            },
        }
        for index in (1, 2)
    ]
    policy = {
        "policy_id": "policy-1",
        "version": 2,
        "name": "F1 baseline",
        "description": "Minimum F1",
        "tags": ["test"],
        "criteria": {"classification": [_criteria()["classification"][0]], "regression": []},
    }

    def fake_request(path, payload=None, **kwargs):
        requests.append((path, payload))
        if path == "/ml-experiments?limit=20&offset=0":
            return {"items": records, "limit": 20, "offset": 0, "total_count": 2}
        if path == "/ml-experiments/policies?limit=100&offset=0":
            return {"items": [policy], "total_count": 1}
        if path == "/ml-experiments/policies/policy-1/versions?limit=100&offset=0":
            return {"items": [policy, {**policy, "version": 1}], "total_count": 2}
        if "/versions/" in path and path.endswith("/evaluate"):
            return {"status": "PASSED", "policy": policy, "results": []}
        if path.endswith("/report"):
            return {"experiment": {"experiment_id": path.split("/")[2]}}
        return {"experiment_id": path.split("/")[-1], "status": "COMPLETED"}

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    app = AppTest.from_string(
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment Orchestration', 'MLExperiments')\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any(item.label == "Policy version to apply" for item in app.selectbox)
    assert (
        any(item.label == "Download policy evaluation JSON" for item in app.download_button)
        is False
    )
    next(
        button for button in app.button if button.label == "Evaluate experiment with policy"
    ).click().run()
    assert any(
        path == "/ml-experiments/policies/policy-1/versions/2/evaluate" for path, _ in requests
    )
    assert any(item.label == "Download policy evaluation JSON" for item in app.download_button)
