"""JSON/CSV audit exports preserve filters, provenance, integrity, and privacy."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import urlencode

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.routes import ml_experiments
from tests.test_ml_evaluation_policy_registry import _criteria, _save_experiment
from tests.test_ml_experiment_gates import _record
from wifi_tunja_smart_predictor.experiments.ml_policy_audit import audit_digest
from wifi_tunja_smart_predictor.experiments.ml_policy_export import (
    build_policy_audit_export,
    render_policy_audit_csv,
)
from wifi_tunja_smart_predictor.experiments.ml_policy_service import MLEvaluationPolicyService
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


def _setup_exports(tmp_path, monkeypatch):
    store = MLExperimentStore(tmp_path / "ml.sqlite")
    service = MLEvaluationPolicyService(store)
    policy = service.create_version(
        policy_id=None,
        name="Export baseline",
        description="Export test policy",
        tags=["test"],
        criteria=_criteria(0.7),
    )
    experiment_id = _save_experiment(store, _record("export-fixture"))
    report = service.evaluate(policy["policy_id"], 1, experiment_id)
    app = FastAPI()
    app.include_router(ml_experiments.router)
    monkeypatch.setattr(ml_experiments, "_orchestrator", lambda: SimpleNamespace(store=store))
    return store, service, policy, experiment_id, report, TestClient(app)


def test_export_builder_has_stable_schema_timestamp_order_and_safe_allowlist():
    records = [
        {
            "audit_id": "b",
            "run_id": "run-b",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "event": {
                "run_id": "run-b",
                "experiment_id": "exp-b",
                "policy_id": "policy-b",
                "policy_version": 1,
                "sensitive_token": "must-not-export",
                "provenance": {"task": "classification", "private_email": "x@example.com"},
            },
            "digest_sha256": "a" * 64,
            "verification": {"status": "INVALID", "reasons": ["Z_REASON", "A_REASON"]},
        },
        {
            "audit_id": "a",
            "run_id": "run-a",
            "timestamp_utc": "2026-01-02T00:00:00Z",
            "event": {"run_id": "run-a", "provenance": {}},
            "verification": {"status": "UNVERIFIABLE", "reasons": []},
        },
    ]
    export = build_policy_audit_export(
        records,
        {"policy_id": "policy-b"},
        generated_at_utc="2026-01-03T00:00:00+00:00",
    )
    assert export["export_schema_version"] == "1.0"
    assert export["generated_at_utc"] == "2026-01-03T00:00:00+00:00"
    assert export["record_count"] == 2
    assert export["records"][0]["run_id"] == "run-a"
    assert export["records"][1]["verification_status"] == "INVALID"
    assert export["records"][1]["verification_reasons"] == ["A_REASON", "Z_REASON"]
    encoded = json.dumps(export, ensure_ascii=False)
    assert "must-not-export" not in encoded
    assert "private_email" not in encoded

    csv_rows = list(csv.DictReader(io.StringIO(render_policy_audit_csv(export))))
    assert csv_rows[0]["row_type"] == "metadata"
    assert csv_rows[0]["record_count"] == "2"
    assert [row["run_id"] for row in csv_rows[1:]] == ["run-a", "run-b"]


def test_json_and_csv_api_exports_all_matching_filtered_records(monkeypatch, tmp_path):
    store, service, policy, experiment_id, first, client = _setup_exports(tmp_path, monkeypatch)
    reports = [first]
    for _ in range(22):
        reports.append(service.evaluate(policy["policy_id"], 1, experiment_id))
    other_policy = service.create_version(
        policy_id=None,
        name="Different policy",
        description="Excluded by the filter",
        tags=[],
        criteria=_criteria(0.8),
    )
    service.evaluate(other_policy["policy_id"], 1, experiment_id)

    bad_event = store.get_policy_audit(reports[0]["run_id"])["event"]
    bad_event["request_body"] = "private request data"
    bad_event["credential"] = "private credential"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE ml_policy_evaluation_audit SET event_json = ? WHERE run_id = ?",
            (json.dumps(bad_event), reports[0]["run_id"]),
        )
        connection.execute(
            "UPDATE ml_policy_evaluation_audit SET event_json = ? WHERE run_id = ?",
            ("not-json", reports[1]["run_id"]),
        )

    query = urlencode({"policy_id": policy["policy_id"], "experiment_id": experiment_id})
    response = client.get(f"/ml-experiments/policy-audit/export.json?{query}")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert 'filename="policy-evaluation-audit.json"' in response.headers["content-disposition"]
    export = response.json()
    assert export["export_schema_version"] == "1.0"
    assert datetime.fromisoformat(export["generated_at_utc"])
    assert export["applied_filters"] == {
        "experiment_id": experiment_id,
        "policy_id": policy["policy_id"],
    }
    assert export["record_count"] == 23  # More than one history page; no pagination truncation.
    assert {row["verification_status"] for row in export["records"]} == {
        "VERIFIED",
        "INVALID",
        "UNVERIFIABLE",
    }
    assert all(row["policy_id"] == policy["policy_id"] for row in export["records"])
    first_ids = [row["audit_id"] for row in export["records"]]
    repeated = client.get(f"/ml-experiments/policy-audit/export.json?{query}").json()
    assert [row["audit_id"] for row in repeated["records"]] == first_ids
    assert "private request data" not in response.text
    assert "private credential" not in response.text
    assert "request_body" not in response.text
    assert "credential" not in response.text

    csv_response = client.get(f"/ml-experiments/policy-audit/export.csv?{query}")
    assert csv_response.status_code == 200
    assert csv_response.headers["content-type"].startswith("text/csv")
    assert 'filename="policy-evaluation-audit.csv"' in csv_response.headers["content-disposition"]
    rows = list(csv.DictReader(io.StringIO(csv_response.text)))
    assert rows[0]["row_type"] == "metadata"
    assert rows[0]["record_count"] == "23"
    assert sum(row["row_type"] == "audit_record" for row in rows) == 23
    assert {row["verification_status"] for row in rows[1:]} == {
        "VERIFIED",
        "INVALID",
        "UNVERIFIABLE",
    }
    assert "private request data" not in csv_response.text
    assert "private credential" not in csv_response.text


def test_empty_exports_keep_metadata_and_invalid_filter_ranges_return_422(monkeypatch, tmp_path):
    _, _, _, _, _, client = _setup_exports(tmp_path, monkeypatch)
    empty = client.get("/ml-experiments/policy-audit/export.json?policy_id=missing")
    assert empty.status_code == 200
    assert empty.json()["record_count"] == 0
    assert empty.json()["records"] == []
    empty_csv = client.get("/ml-experiments/policy-audit/export.csv?policy_id=missing")
    empty_rows = list(csv.DictReader(io.StringIO(empty_csv.text)))
    assert len(empty_rows) == 1
    assert empty_rows[0]["row_type"] == "metadata"
    assert empty_rows[0]["record_count"] == "0"

    invalid_range = client.get(
        "/ml-experiments/policy-audit/export.json"
        "?created_from_utc=2026-12-02T00%3A00%3A00Z&created_to_utc=2026-12-01T00%3A00%3A00Z"
    )
    assert invalid_range.status_code == 422
    malformed_version = client.get("/ml-experiments/policy-audit/export.csv?policy_version=bad")
    assert malformed_version.status_code == 422


def test_export_digest_stays_the_original_local_consistency_value(tmp_path, monkeypatch):
    _, service, policy, _, report, _ = _setup_exports(tmp_path, monkeypatch)
    export = service.export_audits(policy_id=policy["policy_id"])
    row = export["records"][0]
    assert row["digest_sha256"] == audit_digest(
        service.store.get_policy_audit(report["run_id"])["event"]
    )
    assert row["verification_status"] == "VERIFIED"


def test_v86_audit_database_migration_preserves_rows_and_allows_bad_json_inspection(tmp_path):
    database = tmp_path / "existing-v86.sqlite"
    event_json = '{"audit_schema_version":"1.0","run_id":"historical-run"}'
    with sqlite3.connect(database) as connection:
        connection.execute("""CREATE TABLE ml_policy_evaluation_audit (
                audit_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL UNIQUE,
                timestamp_utc TEXT NOT NULL,
                execution_source TEXT NOT NULL,
                outcome TEXT NOT NULL,
                event_json TEXT NOT NULL,
                digest_sha256 TEXT NOT NULL
            )""")
        connection.execute(
            "CREATE INDEX idx_ml_policy_audit_policy ON "
            "ml_policy_evaluation_audit(json_extract(event_json, '$.policy_id'), timestamp_utc DESC)"
        )
        connection.execute(
            "INSERT INTO ml_policy_evaluation_audit VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "audit-1",
                "historical-run",
                "2026-01-01T00:00:00Z",
                "api",
                "PASSED",
                event_json,
                "digest",
            ),
        )

    MLExperimentStore(database)
    with sqlite3.connect(database) as connection:
        saved = connection.execute(
            "SELECT event_json, digest_sha256 FROM ml_policy_evaluation_audit WHERE audit_id = ?",
            ("audit-1",),
        ).fetchone()
        indexes = {
            row[0] for row in connection.execute("PRAGMA index_list(ml_policy_evaluation_audit)")
        }
        connection.execute(
            "UPDATE ml_policy_evaluation_audit SET event_json = ? WHERE audit_id = ?",
            ("malformed-json", "audit-1"),
        )
    assert saved == (event_json, "digest")
    assert "idx_ml_policy_audit_policy" not in indexes
