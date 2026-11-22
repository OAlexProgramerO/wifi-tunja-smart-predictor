"""Policy validation, version persistence, retrieval, and evaluation tests."""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from urllib.parse import urlencode

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


def test_evaluation_history_is_immutable_filterable_and_comparable(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    policy_v1 = _create_policy(service)
    experiment_id = _save_experiment(store, _record("run-1"))
    first_report = service.evaluate(policy_v1["policy_id"], 1, experiment_id)
    policy_v2 = _create_policy(service, policy_v1["policy_id"], threshold=0.8)
    second_report = service.evaluate(policy_v2["policy_id"], 2, experiment_id)

    assert first_report["status"] == "PASSED"
    assert second_report["status"] == "FAILED"
    assert (
        store.get_policy_evaluation_run(first_report["run_id"])["report"]["policy"]["version"] == 1
    )
    assert (
        store.get_policy_evaluation_run(first_report["run_id"])["report"]["criteria"][0][
            "threshold"
        ]
        == 0.7
    )

    page, total = service.list_runs(
        limit=1,
        offset=0,
        policy_id=policy_v1["policy_id"],
        policy_version=2,
        experiment_id=experiment_id,
        status="FAILED",
    )
    assert total == 1
    assert page[0]["run_id"] == second_report["run_id"]
    first_page, first_total = service.list_runs(limit=1, offset=0)
    second_page, second_total = service.list_runs(limit=1, offset=1)
    assert first_total == second_total == 2
    assert first_page[0]["run_id"] != second_page[0]["run_id"]
    exact_date, exact_total = service.list_runs(
        limit=10,
        offset=0,
        created_from_utc=datetime.fromisoformat(second_report["created_at_utc"]),
        created_to_utc=datetime.fromisoformat(second_report["created_at_utc"]),
    )
    assert exact_total == 1
    assert exact_date[0]["run_id"] == second_report["run_id"]

    comparison = service.compare_runs(first_report["run_id"], second_report["run_id"])
    assert comparison["status"] == "COMPATIBLE"
    assert comparison["criterion_comparisons"][0]["threshold_changed"] is True
    assert comparison["criterion_comparisons"][0]["assessment"] == "OUTCOME_CHANGED"


def test_run_comparison_rejects_incompatible_experiments_and_handles_missing_evidence(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    policy = _create_policy(service)
    first_experiment = _save_experiment(store, _record("first"))
    second_experiment = _save_experiment(store, _record("second"))
    first = service.evaluate(policy["policy_id"], 1, first_experiment)
    second = service.evaluate(policy["policy_id"], 1, second_experiment)
    incompatible = service.compare_runs(first["run_id"], second["run_id"])
    assert incompatible["status"] == "INCOMPATIBLE"
    assert "DIFFERENT_EXPERIMENTS" in incompatible["reasons"]

    missing_record = _record("missing-evidence")
    missing_record["result"]["evaluation"]["metrics"].pop("f1")
    missing_experiment = _save_experiment(store, missing_record)
    missing_first = service.evaluate(policy["policy_id"], 1, missing_experiment)
    missing_second = service.evaluate(policy["policy_id"], 1, missing_experiment)
    partial = service.compare_runs(missing_first["run_id"], missing_second["run_id"])
    assert partial["status"] == "PARTIAL"
    assert partial["criterion_comparisons"][0]["assessment"] == "INCONCLUSIVE"


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
    first_run = evaluation.json()
    second_evaluation = client.post(
        f"/ml-experiments/policies/{policy_id}/versions/2/evaluate",
        json={"experiment_id": experiment_id},
    )
    assert second_evaluation.json()["status"] == "FAILED"
    history_page = client.get("/ml-experiments/policy-runs?limit=1&offset=1")
    assert history_page.status_code == 200
    assert history_page.json()["total_count"] == 2
    assert len(history_page.json()["items"]) == 1
    query = urlencode(
        {
            "policy_id": policy_id,
            "policy_version": 1,
            "experiment_id": experiment_id,
            "status": "PASSED",
            "created_from_utc": first_run["created_at_utc"],
            "created_to_utc": first_run["created_at_utc"],
        }
    )
    history = client.get(f"/ml-experiments/policy-runs?{query}")
    assert history.status_code == 200
    assert history.json()["total_count"] == 1
    run_id = history.json()["items"][0]["run_id"]
    detail = client.get(f"/ml-experiments/policy-runs/{run_id}")
    assert detail.status_code == 200
    assert detail.json()["report"]["policy"]["version"] == 1
    report_response = client.get(f"/ml-experiments/policy-runs/{run_id}/report")
    assert report_response.json()["run_id"] == run_id
    comparison = client.post(
        "/ml-experiments/policy-runs/compare",
        json={
            "left_run_id": first_run["run_id"],
            "right_run_id": second_evaluation.json()["run_id"],
        },
    )
    assert comparison.status_code == 200
    assert comparison.json()["status"] == "COMPATIBLE"


def test_dashboard_browses_policy_versions_and_runs_selected_version(monkeypatch):
    from streamlit.testing.v1 import AppTest

    from app import dashboard

    requests = []
    download_requests = []
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
    history_items = [
        {
            "run_id": f"history-{index}",
            "created_at_utc": "2026-11-20T12:00:00+00:00",
            "experiment_id": f"run-{index}",
            "policy_id": "policy-1",
            "policy_version": index,
            "status": "PASSED",
            "criteria": [],
            "provenance": {},
        }
        for index in (1, 2)
    ]

    def fake_request(path, payload=None, **kwargs):
        requests.append((path, payload))
        if path == "/ml-experiments?limit=20&offset=0":
            return {"items": records, "limit": 20, "offset": 0, "total_count": 2}
        if path == "/ml-experiments/policies?limit=100&offset=0":
            return {"items": [policy], "total_count": 1}
        if path == "/ml-experiments/policies/policy-1/versions?limit=100&offset=0":
            return {"items": [policy, {**policy, "version": 1}], "total_count": 2}
        if path == "/ml-experiments/policy-runs?limit=20&offset=0":
            return {"items": history_items, "total_count": 2}
        if path.startswith("/ml-experiments/policy-runs/") and path.endswith("/report"):
            run_id = path.split("/")[-2]
            return {"run_id": run_id, "policy": policy, "criteria": []}
        if path == "/ml-experiments/policy-runs/compare":
            return {"status": "COMPATIBLE", "criterion_comparisons": []}
        if path.startswith("/ml-experiments/policy-audit?"):
            return {
                "items": [
                    {
                        "audit_id": "history-1",
                        "run_id": "history-1",
                        "timestamp_utc": "2026-11-20T12:00:00+00:00",
                        "execution_source": "api",
                        "outcome": "PASSED",
                        "digest_sha256": "a" * 64,
                        "verification": {"status": "VERIFIED"},
                        "event": {"run_id": "history-1"},
                    }
                ],
                "total_count": 1,
            }
        if path.endswith("/audit/verify"):
            return {"status": "VERIFIED", "reasons": [], "explanation": "consistent"}
        if path.startswith("/ml-experiments/policy-runs/"):
            run_id = path.rsplit("/", maxsplit=1)[-1]
            item = next(row for row in history_items if row["run_id"] == run_id)
            return {**item, "report": {"run_id": run_id}}
        if "/versions/" in path and path.endswith("/evaluate"):
            return {"status": "PASSED", "policy": policy, "results": []}
        if path.endswith("/report"):
            return {"experiment": {"experiment_id": path.split("/")[2]}}
        return {"experiment_id": path.split("/")[-1], "status": "COMPLETED"}

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    monkeypatch.setattr(
        dashboard,
        "main_api_download",
        lambda path: download_requests.append(path) or path.encode("utf-8"),
    )
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

    next(
        button for button in app.button if button.label == "Load evaluation run history"
    ).click().run()
    assert any("Showing 2 of 2 matching runs." in item.value for item in app.caption)
    history_selection = next(
        item for item in app.multiselect if item.key == "ml_policy_run_comparison_selection"
    )
    history_selection.set_value(["history-1", "history-2"]).run()
    next(button for button in app.button if button.label == "Compare evaluation runs").click().run()
    assert any(path == "/ml-experiments/policy-runs/compare" for path, _ in requests)
    assert any(
        item.label == "Download evaluation run comparison JSON" for item in app.download_button
    )
    next(
        button for button in app.button if button.label == "Load evaluation audit history"
    ).click().run()
    assert any(path.startswith("/ml-experiments/policy-audit?") for path, _ in requests)
    next(
        button for button in app.button if button.label == "Prepare complete filtered audit exports"
    ).click().run()
    assert any(path.endswith("/export.json") for path in download_requests)
    assert any(path.endswith("/export.csv") for path in download_requests)
    assert all("limit=" not in path and "offset=" not in path for path in download_requests)
    assert any(item.label == "Download complete audit JSON" for item in app.download_button)
    assert any(item.label == "Download complete audit CSV" for item in app.download_button)
    app.text_input(key="ml_policy_audit_verify_run_id").set_value("history-1").run()
    next(button for button in app.button if button.label == "Verify audit record").click().run()
    assert any(path.endswith("/policy-runs/history-1/audit/verify") for path, _ in requests)
    assert any("Audit record status: VERIFIED" in item.value for item in app.success)
