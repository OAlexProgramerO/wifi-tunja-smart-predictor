"""V5 scenario-capacity and experiment-persistence contracts."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from api.main import app
from api.routes import experiments as experiment_routes
from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.capacity import plan_capacity
from wifi_tunja_smart_predictor.experiments.store import ExperimentStore
from wifi_tunja_smart_predictor.reporting.service import (
    build_experiment_report,
    serialize_experiment_json,
)
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService


def test_capacity_plan_conserves_budget_and_uses_stable_tie_break() -> None:
    result = plan_capacity(
        [
            {"zone_id": "b", "demand": 20, "capacity": 10},
            {"zone_id": "a", "demand": 20, "capacity": 10},
        ],
        resource_budget=5,
    )
    assert result["allocated_connections"] == 5
    assert result["remaining_budget_connections"] == 0
    assert result["zones"][0]["zone_id"] == "a"
    assert result["zones"][0]["allocated_capacity"] == 5
    assert result["zones"][1]["allocated_capacity"] == 0
    assert result["aggregate"]["estimated_demand_connections"] == 40
    assert result["aggregate"]["overload_connections"] == 15


def test_capacity_zero_capacity_has_no_infinite_utilization() -> None:
    result = plan_capacity([{"zone_id": "z", "demand": 0, "capacity": 0}])
    assert result["zones"][0]["utilization_pct"] is None
    assert result["zones"][0]["overload_connections"] == 0


def test_capacity_rejects_invalid_values() -> None:
    import pytest

    with pytest.raises(ValueError, match="non-negative"):
        plan_capacity([{"zone_id": "z", "demand": 1, "capacity": 1}], -1)


def test_capacity_prioritizes_largest_deficit_and_conserves_budget() -> None:
    result = plan_capacity(
        [
            {"zone_id": "small-gap", "demand": 20, "capacity": 19},
            {"zone_id": "large-gap", "demand": 15, "capacity": 0},
        ],
        resource_budget=5,
    )
    zones = {row["zone_id"]: row for row in result["zones"]}
    assert zones["large-gap"]["allocated_capacity"] == 5
    assert zones["small-gap"]["allocated_capacity"] == 0
    assert result["allocated_connections"] <= result["resource_budget_connections"]


def test_experiment_store_persists_and_paginates(tmp_path) -> None:
    store = ExperimentStore(tmp_path / "experiments.sqlite3")
    first = store.save("First", {"seed": 42}, {"score": 1.0})
    store.save("Second", {}, {})
    assert store.get(first["id"])["configuration"] == {"seed": 42}
    assert len(store.list(limit=1, offset=0)) == 1
    assert store.list(limit=1, offset=1)[0]["name"] == "First"


def test_experiment_store_migrates_legacy_records_without_data_loss(tmp_path) -> None:
    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_version (version TEXT NOT NULL)")
        connection.execute("INSERT INTO schema_version VALUES ('1.0')")
        connection.execute("""CREATE TABLE experiments (
                id TEXT PRIMARY KEY, created_at_utc TEXT NOT NULL,
                experiment_type TEXT NOT NULL, name TEXT NOT NULL,
                schema_version TEXT NOT NULL, configuration_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            )""")
        connection.execute(
            "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-id",
                "2025-01-01T00:00:00+00:00",
                "capacity_scenario",
                "legacy",
                "1.0",
                "{}",
                "{}",
            ),
        )
    record = ExperimentStore(path).get("legacy-id")
    assert record["name"] == "legacy"
    assert record["schema_version"] == "1.2"
    assert record["updated_at_utc"] == record["created_at_utc"]


def test_experiment_store_rejects_corrupt_persisted_json(tmp_path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    store = ExperimentStore(path)
    record = store.save("corrupt me", {}, {})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE experiments SET result_json = ? WHERE id = ?", ("{bad json", record["id"])
        )
    import pytest

    with pytest.raises(ValueError, match="invalid JSON"):
        store.get(record["id"])


def test_dataset_fingerprint_tracks_the_actual_scenario_frame():
    frame = make_mini_frame(8)
    original = ScenarioPredictionService(frame, classifier=object(), regressor=object())
    changed_frame = frame.copy()
    changed_frame.loc[changed_frame.index[0], "connections_previous_hour"] += 1
    changed = ScenarioPredictionService(changed_frame, classifier=object(), regressor=object())
    assert len(original.dataset_sha256) == 64
    assert original.dataset_sha256 != changed.dataset_sha256


def test_scenario_experiment_api_saves_model_trace_and_exports(monkeypatch, tmp_path) -> None:
    class FakePrediction:
        def to_dict(self):
            return {
                "scenario": {
                    "model_version": "model-v",
                    "dataset_version": "synthetic-v1",
                    "dataset_sha256": "d" * 64,
                    "dataset_fingerprint_algorithm": "pandas_hash_pandas_object_sha256_v1",
                    "scenario_time": "2025-01-01T10:00:00",
                },
                "regression": {"predicted_connections_next_hour": 12.5},
                "traceability": {
                    "classifier": {"artifact_sha256": "a" * 64, "model_version": "model-v"},
                    "regressor": {"artifact_sha256": "b" * 64, "model_version": "model-v"},
                },
            }

    class FakeService:
        def predict(self, request):
            return FakePrediction()

    monkeypatch.setattr(experiment_routes, "_scenario_service", lambda: FakeService())
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(tmp_path / "experiments.sqlite3"))
    experiment_routes._store.cache_clear()
    zone_id = str(make_mini_frame(8)["zone_id"].iloc[0])
    client = TestClient(app)
    response = client.post(
        "/experiments/scenarios",
        json={
            "name": "test run",
            "datetime": "2025-01-01T10:00:00",
            "zones": [{"zone_id": zone_id, "capacity_connections": 10}],
            "resource_budget_connections": 5,
        },
    )
    assert response.status_code == 201
    experiment = response.json()
    assert experiment["result"]["zones"][0]["demand"] == 12.5
    assert experiment["status"] == "COMPLETED"
    assert (
        len(experiment["result"]["provenance"]["models"][0]["classifier"]["artifact_sha256"]) == 64
    )
    assert client.get(f"/experiments/{experiment['id']}").status_code == 200
    report = client.get(f"/experiments/{experiment['id']}/report?format=markdown")
    assert report.status_code == 200
    assert (
        "All observations, locations, predictions, and capacity values are synthetic" in report.text
    )
    assert "Model artifact provenance" in report.text
    json_report = client.get(f"/experiments/{experiment['id']}/report?format=json").json()
    assert json_report["report_schema_version"] == "1.0"
    assert json_report["dataset_provenance"] == "d" * 64
    fixed_time = datetime(2026, 1, 1, tzinfo=UTC)
    fixed_report = build_experiment_report(experiment, generated_at_utc=fixed_time)
    repeated_report = build_experiment_report(experiment, generated_at_utc=fixed_time)
    assert serialize_experiment_json(fixed_report) == serialize_experiment_json(repeated_report)
    second = experiment_routes._store().save(
        "comparison",
        {},
        {
            "time_window": experiment["result"]["time_window"],
            "zones": [
                {
                    "zone_id": zone_id,
                    "demand": 12.5,
                    "effective_capacity": 20,
                    "overload_connections": 0,
                }
            ],
            "provenance": experiment["result"]["provenance"],
        },
    )
    compared = client.post(
        "/experiments/compare", json={"experiment_ids": [experiment["id"], second["id"]]}
    )
    assert compared.status_code == 200
    assert compared.json()["comparisons"][0]["changes"][zone_id]["capacity_connections"] == 7.5
    incompatible = experiment_routes._store().save(
        "different time", {}, {"time_window": "2025-02-01T10:00:00", "zones": [], "provenance": {}}
    )
    conflict = client.post(
        "/experiments/compare", json={"experiment_ids": [experiment["id"], incompatible["id"]]}
    )
    assert conflict.status_code == 409
    experiment_routes._store.cache_clear()


def test_assistant_reports_empty_experiment_history_bilingually(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(tmp_path / "missing.sqlite3"))
    service = AssistantService(frame=make_mini_frame(8))
    english = service.handle(ChatRequest(message="Show experiment history"))
    spanish = service.handle(ChatRequest(message="¿Qué muestra el historial de experimentos?"))
    assert english.intent == "EXPERIMENT_HISTORY"
    assert "no saved capacity experiments" in english.answer
    assert "No hay experimentos" in spanish.answer


def test_assistant_compares_compatible_saved_capacity_experiments(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(tmp_path / "history.sqlite3"))
    store = ExperimentStore(tmp_path / "history.sqlite3")
    provenance = {
        "dataset_version": "synthetic-v1",
        "dataset_sha256": "d" * 64,
        "dataset_fingerprint_algorithm": "pandas_hash_pandas_object_sha256_v1",
        "models": [
            {
                "classifier": {"artifact_sha256": "a" * 64, "model_version": "v"},
                "regressor": {"artifact_sha256": "b" * 64, "model_version": "v"},
            }
        ],
    }
    base = {
        "time_window": "2025-01-01T10:00:00",
        "provenance": provenance,
        "capacity_plan": {"aggregate": {"overload_connections": 8.0}},
    }
    changed = {
        "time_window": "2025-01-01T10:00:00",
        "provenance": provenance,
        "capacity_plan": {"aggregate": {"overload_connections": 3.0}},
    }
    store.save("baseline", {}, base)
    store.save("alternative", {}, changed)
    result = AssistantService(frame=make_mini_frame(8)).handle(
        ChatRequest(message="Compare experiments")
    )
    assert "changed by -5.0 connections" in result.answer
    assert "synthetic scenarios" in result.answer


def test_report_rejects_malformed_record() -> None:
    import pytest

    with pytest.raises(ValueError, match="configuration or result"):
        build_experiment_report({"configuration": [], "result": {}})
