"""Integrity, pagination, privacy, and API tests for policy evaluation audits."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import ml_experiments
from tests.test_ml_evaluation_policy_registry import _criteria, _save_experiment
from tests.test_ml_experiment_gates import _record
from wifi_tunja_smart_predictor.experiments.ml_policy_audit import (
    audit_digest,
    canonical_audit_json,
)
from wifi_tunja_smart_predictor.experiments.ml_policy_service import MLEvaluationPolicyService
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


def _recorded_run(store: MLExperimentStore, service: MLEvaluationPolicyService, threshold: float):
    policy = service.create_version(
        policy_id=None,
        name="Audit test policy",
        description="Synthetic test",
        tags=["test"],
        criteria=_criteria(threshold),
    )
    experiment_id = _save_experiment(store, _record(f"experiment-{threshold}"))
    return service.evaluate(policy["policy_id"], 1, experiment_id)


def test_canonical_serialization_and_digest_are_deterministic():
    first = {"b": "ñ", "a": {"z": 2, "x": 1}}
    second = {"a": {"x": 1, "z": 2}, "b": "ñ"}
    assert canonical_audit_json(first) == '{"a":{"x":1,"z":2},"b":"ñ"}'
    assert audit_digest(first) == audit_digest(second)


def test_policy_evaluation_writes_privacy_safe_audit_and_verifies(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    report = _recorded_run(store, service, 0.7)

    audit = store.get_policy_audit(report["run_id"])
    assert audit["event"]["run_id"] == report["run_id"]
    assert audit["event"]["policy_version"] == 1
    assert audit["event"]["execution_source"] == "api"
    assert audit["event"]["outcome"] == report["status"]
    assert audit["digest_sha256"] == audit_digest(audit["event"])
    assert service.verify_audit(report["run_id"])["status"] == "VERIFIED"
    serialized = canonical_audit_json(audit["event"]).lower()
    assert all(
        secret not in serialized for secret in ("chat", "credential", "request_body", "email")
    )


def test_audit_verification_detects_modified_event_and_missing_event(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    report = _recorded_run(store, service, 0.7)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE ml_policy_evaluation_audit SET event_json = ? WHERE run_id = ?",
            ('{"outcome":"FAILED"}', report["run_id"]),
        )
    altered = service.verify_audit(report["run_id"])
    assert altered["status"] == "INVALID"
    assert "DIGEST_MISMATCH" in altered["reasons"]

    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM ml_policy_evaluation_audit WHERE run_id = ?", (report["run_id"],)
        )
    missing = service.verify_audit(report["run_id"])
    assert missing["status"] == "UNVERIFIABLE"
    assert missing["reasons"] == ["AUDIT_EVENT_MISSING"]


def test_audit_verification_detects_orphan_run_reference(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    report = _recorded_run(store, service, 0.7)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "DELETE FROM ml_policy_evaluation_runs WHERE run_id = ?", (report["run_id"],)
        )
    verification = service.verify_audit(report["run_id"])
    assert verification["status"] == "INVALID"
    assert "LINKED_RUN_MISSING" in verification["reasons"]


def test_audit_verification_detects_consistent_digest_with_wrong_run_reference(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    report = _recorded_run(store, service, 0.7)
    audit = store.get_policy_audit(report["run_id"])
    event = audit["event"]
    event["experiment_id"] = "different-experiment"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE ml_policy_evaluation_audit SET event_json = ?, digest_sha256 = ? "
            "WHERE run_id = ?",
            (canonical_audit_json(event), audit_digest(event), report["run_id"]),
        )
    verification = service.verify_audit(report["run_id"])
    assert verification["status"] == "INVALID"
    assert "AUDIT_RUN_REFERENCE_MISMATCH" in verification["reasons"]


def test_audit_verification_detects_provenance_mismatch_even_with_matching_digest(tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    report = _recorded_run(store, service, 0.7)
    audit = store.get_policy_audit(report["run_id"])
    event = audit["event"]
    event["provenance"]["dataset_fingerprint"] = "0" * 64
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE ml_policy_evaluation_audit SET event_json = ?, digest_sha256 = ? "
            "WHERE run_id = ?",
            (canonical_audit_json(event), audit_digest(event), report["run_id"]),
        )
    verification = service.verify_audit(report["run_id"])
    assert verification["status"] == "INVALID"
    assert "AUDIT_PROVENANCE_REPORT_MISMATCH" in verification["reasons"]


def test_audit_history_api_paginates_filters_and_verifies(monkeypatch, tmp_path):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    first = _recorded_run(store, service, 0.7)
    second = _recorded_run(store, service, 0.8)
    app = FastAPI()
    app.include_router(ml_experiments.router)
    monkeypatch.setattr(ml_experiments, "_orchestrator", lambda: SimpleNamespace(store=store))
    client = TestClient(app)

    page = client.get("/ml-experiments/policy-audit?limit=1&offset=0")
    assert page.status_code == 200
    assert page.json()["total_count"] == 2
    assert len(page.json()["items"]) == 1
    assert (
        client.get("/ml-experiments/policy-runs/" + first["run_id"] + "/audit/verify").json()[
            "status"
        ]
        == "VERIFIED"
    )
    query = urlencode(
        {
            "policy_id": second["policy"]["policy_id"],
            "policy_version": 1,
            "experiment_id": second["experiment"]["experiment_id"],
            "outcome": second["status"],
            "execution_source": "api",
        }
    )
    filtered = client.get(f"/ml-experiments/policy-audit?{query}")
    assert filtered.status_code == 200
    assert filtered.json()["total_count"] == 1
    assert filtered.json()["items"][0]["run_id"] == second["run_id"]
    assert client.get("/ml-experiments/policy-runs/missing/audit/verify").json()["status"] == (
        "UNVERIFIABLE"
    )
