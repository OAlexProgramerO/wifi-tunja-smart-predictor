"""V5 scenario-capacity and experiment-persistence contracts."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.routes import experiments as experiment_routes
from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor import config
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


def test_experiment_store_closes_connections_after_each_operation(monkeypatch, tmp_path) -> None:
    store = ExperimentStore(tmp_path / "closed.sqlite3")
    connections = []
    connect = sqlite3.connect

    def tracked_connect(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(
        "wifi_tunja_smart_predictor.experiments.store.sqlite3.connect", tracked_connect
    )
    record = store.save("close handles", {}, {})
    store.get(record["id"])
    store.list()

    assert len(connections) == 3
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed database"):
            connection.execute("SELECT 1")


def test_experiment_database_default_is_shared_and_independent_of_cwd(
    monkeypatch, tmp_path
) -> None:
    project_root = tmp_path / "checkout"
    monkeypatch.setattr(config, "PROJECT_ROOT", project_root)
    monkeypatch.delenv("EXPERIMENT_DB_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    expected = project_root / "reports" / "experiments.sqlite3"
    assert config.experiment_database_path() == expected
    ExperimentStore(expected).save("created outside cwd", {}, {"zones": []})
    experiment_routes._store.cache_clear()
    assert experiment_routes._store().path == expected
    response = AssistantService(frame=make_mini_frame(8)).handle(
        ChatRequest(message="Show experiment history")
    )
    assert "created outside cwd" in response.answer
    experiment_routes._store.cache_clear()

    monkeypatch.setenv("EXPERIMENT_DB_PATH", "data/experiment-history.sqlite3")
    assert config.experiment_database_path() == project_root / "data" / "experiment-history.sqlite3"


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
                    "application_version": "0.5.3",
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
    client = TestClient(app, raise_server_exceptions=False)
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
    assert json_report["report_schema_version"] == "1.4"
    assert json_report["integrity"]["status"] == "verified"
    assert json_report["dataset_provenance"] == "d" * 64
    fixed_time = datetime(2026, 1, 1, tzinfo=UTC)
    fixed_report = build_experiment_report(experiment, generated_at_utc=fixed_time)
    repeated_report = build_experiment_report(experiment, generated_at_utc=fixed_time)
    assert serialize_experiment_json(fixed_report) == serialize_experiment_json(repeated_report)
    second = experiment_routes._store().save(
        "comparison",
        {
            "datetime": "2025-01-01T10:00:00",
            "zones": [{"zone_id": zone_id, "capacity_connections": 20}],
        },
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
    assert compared.json()["comparisons"][0]["comparison_status"] == "comparable"
    incompatible = experiment_routes._store().save(
        "different time",
        {
            "datetime": "2025-02-01T10:00:00",
            "zones": [{"zone_id": zone_id, "capacity_connections": 20}],
        },
        {
            "time_window": "2025-02-01T10:00:00",
            "zones": [],
            "provenance": experiment["result"]["provenance"],
        },
    )
    conflict = client.post(
        "/experiments/compare", json={"experiment_ids": [experiment["id"], incompatible["id"]]}
    )
    assert conflict.status_code == 409
    assert conflict.json()["reason_code"] == "INCOMPATIBLE_PROVENANCE"
    assert "INCOMPATIBLE_TIME_WINDOW" in conflict.json()["reason_codes"]
    malformed_a = experiment_routes._store().save(
        "malformed A",
        {"datetime": "2025-01-01", "zones": [{"zone_id": zone_id, "capacity_connections": 1}]},
        {"provenance": ["invalid"], "time_window": "2025-01-01"},
    )
    malformed_b = experiment_routes._store().save(
        "malformed B",
        {"datetime": "2025-01-01", "zones": [{"zone_id": zone_id, "capacity_connections": 1}]},
        {"provenance": ["invalid"], "time_window": "2025-01-01"},
    )
    malformed_response = client.post(
        "/experiments/compare",
        json={"experiment_ids": [malformed_a["id"], malformed_b["id"]]},
    )
    assert malformed_response.status_code == 409
    malformed_trace_provenance = {
        **experiment["result"]["provenance"],
        "models": ["invalid trace"],
    }
    malformed_trace_a = experiment_routes._store().save(
        "malformed trace A",
        {
            "datetime": "2025-01-01T10:00:00",
            "zones": [{"zone_id": zone_id, "capacity_connections": 1}],
        },
        {
            "time_window": "2025-01-01T10:00:00",
            "provenance": {
                **malformed_trace_provenance,
                "models": [
                    {
                        **experiment["result"]["provenance"]["models"][0],
                        "classifier": "invalid",
                    }
                ],
            },
            "zones": [],
        },
    )
    malformed_trace_b = experiment_routes._store().save(
        "malformed trace B",
        {
            "datetime": "2025-01-01T10:00:00",
            "zones": [{"zone_id": zone_id, "capacity_connections": 1}],
        },
        {
            "time_window": "2025-01-01T10:00:00",
            "provenance": {
                **malformed_trace_provenance,
                "models": [
                    {
                        **experiment["result"]["provenance"]["models"][0],
                        "classifier": "invalid",
                    }
                ],
            },
            "zones": [],
        },
    )
    assert (
        client.post(
            "/experiments/compare",
            json={"experiment_ids": [malformed_trace_a["id"], malformed_trace_b["id"]]},
        ).status_code
        == 409
    )
    disjoint_zone = "SYNTHETIC_ZONE_WITH_NO_SHARED_ID"
    no_overlap = experiment_routes._store().save(
        "different zone",
        {
            "datetime": "2025-01-01T10:00:00",
            "zones": [{"zone_id": disjoint_zone, "capacity_connections": 8}],
        },
        {
            "time_window": experiment["result"]["time_window"],
            "provenance": experiment["result"]["provenance"],
            "zones": [
                {
                    "zone_id": disjoint_zone,
                    "demand": 8,
                    "effective_capacity": 8,
                    "overload_connections": 0,
                }
            ],
        },
    )
    no_overlap_response = client.post(
        "/experiments/compare", json={"experiment_ids": [experiment["id"], no_overlap["id"]]}
    )
    no_overlap_result = no_overlap_response.json()["comparisons"][0]
    assert no_overlap_response.status_code == 200
    assert no_overlap_result["shared_zones"] == []
    assert no_overlap_result["changes"] == {}
    assert no_overlap_result["comparison_status"] == "no_shared_zones"
    assert "no zone-level results" in no_overlap_result["interpretation_detail"]
    assert no_overlap_result["scientific_comparability"] == "not_assessable"
    assert no_overlap_result["comparability_reasons"] == ["The experiments share no zone IDs."]
    assert no_overlap_result["comparability_reason_codes"] == ["NO_SHARED_ZONES"]
    malformed_zone = experiment_routes._store().save(
        "malformed zone",
        {
            "datetime": "2025-01-01T10:00:00",
            "zones": [{"zone_id": zone_id, "capacity_connections": 1}],
        },
        {
            "time_window": experiment["result"]["time_window"],
            "provenance": experiment["result"]["provenance"],
            "zones": [{"zone_id": zone_id, "demand": "unknown"}],
        },
    )
    assert (
        client.post(
            "/experiments/compare",
            json={"experiment_ids": [experiment["id"], malformed_zone["id"]]},
        ).status_code
        == 409
    )
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
        "application_version": "0.5.3",
        "models": [
            {
                "application_version": "0.5.3",
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


def test_assistant_rejects_malformed_saved_experiment_comparison(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(tmp_path / "corrupt-history.sqlite3"))
    store = ExperimentStore(tmp_path / "corrupt-history.sqlite3")
    malformed = {
        "time_window": "2025-01-01T10:00:00",
        "provenance": {
            "dataset_version": "synthetic-v1",
            "dataset_sha256": "d" * 64,
            "dataset_fingerprint_algorithm": "pandas_hash_pandas_object_sha256_v1",
            "models": ["malformed trace"],
        },
        "capacity_plan": {"aggregate": {"overload_connections": 0}},
    }
    store.save("malformed A", {}, malformed)
    store.save("malformed B", {}, malformed)

    result = AssistantService(frame=make_mini_frame(8)).handle(
        ChatRequest(message="Compare experiments")
    )

    assert "should not be compared directly" in result.answer


def test_report_rejects_malformed_record() -> None:
    import pytest

    with pytest.raises(ValueError, match="configuration or result"):
        build_experiment_report({"configuration": [], "result": {}})


def test_legacy_experiment_is_incomplete_and_report_labels_missing_metadata(tmp_path) -> None:
    store = ExperimentStore(tmp_path / "legacy.sqlite3")
    record = store.save("legacy", {"datetime": "2025-01-01", "zones": []}, {"zones": []})
    assert record["integrity"]["status"] == "incomplete"
    report = build_experiment_report(record)
    assert report["integrity"]["assessment_status"] == "not_assessable"
    assert report["dataset_provenance"] == "Not recorded"


def test_integrity_invalid_for_bad_fingerprint_and_malformed_zone(tmp_path) -> None:
    store = ExperimentStore(tmp_path / "invalid.sqlite3")
    record = store.save(
        "bad provenance",
        {"datetime": "2025-01-01", "zones": [{"zone_id": "z", "capacity_connections": 1}]},
        {
            "zones": [{"zone_id": "z", "demand": "unknown"}],
            "provenance": {"dataset_sha256": "bad"},
        },
    )
    assert record["integrity"]["status"] == "invalid"
    assert "DATASET_FINGERPRINT_INVALID" in record["integrity"]["reason_codes"]
    with pytest.raises(ValueError, match="integrity"):
        build_experiment_report(record)


def test_experiment_list_skips_corrupt_row_and_returns_warning(tmp_path, monkeypatch) -> None:
    path = tmp_path / "partial.sqlite3"
    store = ExperimentStore(path)
    valid = store.save("valid row", {}, {})
    corrupt = store.save("corrupt row", {}, {})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE experiments SET result_json = ? WHERE id = ?", ("{bad", corrupt["id"])
        )
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(path))
    experiment_routes._store.cache_clear()
    response = TestClient(app).get("/experiments")
    assert response.status_code == 200
    assert [item["id"] for item in response.json()["items"]] == [valid["id"]]
    assert response.json()["integrity_warnings"][0]["reason_code"] == "INVALID_PERSISTED_RECORD"
    assert response.json()["analytics"]["total_experiment_count"] == 1
    experiment_routes._store.cache_clear()


def test_experiment_history_filters_before_pagination_and_reports_filtered_total(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "filtered.sqlite3"
    store = ExperimentStore(path)
    records = [
        store.save(
            name,
            {"zones": [{"zone_id": zone}]},
            {},
        )
        for name, zone in (("Alpha", "zone-a"), ("Beta", "zone-b"), ("Alpha second", "zone-a"))
    ]
    with sqlite3.connect(path) as connection:
        for record, created in zip(
            records,
            (
                "2026-01-01T01:00:00+00:00",
                "2026-01-02T01:00:00+00:00",
                "2026-01-03T01:00:00+00:00",
            ),
        ):
            connection.execute(
                "UPDATE experiments SET created_at_utc = ? WHERE id = ?",
                (created, record["id"]),
            )
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(path))
    experiment_routes._store.cache_clear()
    client = TestClient(app)
    response = client.get(
        "/experiments?search=alpha&zone_id=zone-a&start_date=2026-01-01&end_date=2026-01-03&limit=1&offset=1"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 2
    assert len(payload["items"]) == 1
    assert payload["items"][0]["name"] == "Alpha"
    date_response = client.get("/experiments?start_date=2026-01-02&end_date=2026-01-02")
    assert date_response.status_code == 200
    assert date_response.json()["total_count"] == 1
    assert date_response.json()["items"][0]["name"] == "Beta"
    unfiltered = client.get("/experiments")
    assert unfiltered.json()["total_count"] == 3
    assert client.get("/experiments?start_date=2026-01-03&end_date=2026-01-01").status_code == 422
    assert client.get("/experiments?search=").status_code == 422
    experiment_routes._store.cache_clear()


def test_experiment_analytics_cover_all_filtered_records_and_group_in_utc(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "analytics.sqlite3"
    store = ExperimentStore(path)
    model_reference = {
        "selected_model": "test-model",
        "model_version": "model-v1",
        "artifact_sha256": "b" * 64,
    }

    def save_analytics_record(name, created_at, zones, budget, demands):
        record_zones = [
            {"zone_id": zone_id, "capacity_connections": capacity} for zone_id, capacity in zones
        ]
        result_zones = [
            {
                "zone_id": zone_id,
                "demand": demand,
                "effective_capacity": capacity,
                "overload_connections": max(0, demand - capacity),
            }
            for (zone_id, capacity), demand in zip(zones, demands)
        ]
        record = store.save(
            name,
            {
                "datetime": "2026-01-01T18:00:00+00:00",
                "zones": record_zones,
                "resource_budget_connections": budget,
            },
            {
                "time_window": "2026-01-01T18:00:00+00:00",
                "zones": result_zones,
                "provenance": {
                    "dataset_version": "synthetic-v1",
                    "dataset_sha256": "a" * 64,
                    "dataset_fingerprint_algorithm": "sha256-canonical-v1",
                    "models": [
                        {
                            "application_version": "0.7.1",
                            "classifier": model_reference,
                            "regressor": model_reference,
                        }
                    ],
                },
            },
        )
        with sqlite3.connect(path) as connection:
            connection.execute(
                "UPDATE experiments SET created_at_utc = ? WHERE id = ?",
                (created_at, record["id"]),
            )
        return record

    save_analytics_record(
        "Alpha",
        "2026-01-01T23:30:00-02:00",
        [("zone-1", 10)],
        0,
        [0],
    )
    save_analytics_record(
        "Beta",
        "2026-01-02T02:00:00+00:00",
        [("zone-1", 20), ("zone-2", 30)],
        10,
        [10, 14],
    )
    save_analytics_record(
        "Invalid result",
        "2026-01-03T02:00:00+00:00",
        [("zone-3", 50)],
        20,
        [-1],
    )
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(path))
    experiment_routes._store.cache_clear()
    client = TestClient(app)

    response = client.get(
        "/experiments?zone_id=zone-1&start_date=2026-01-02&end_date=2026-01-02&limit=1&offset=0"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 2
    assert len(payload["items"]) == 1
    analytics = payload["analytics"]
    assert analytics["total_experiment_count"] == 2
    assert analytics["experiments_by_zone"] == [
        {"zone_id": "zone-1", "experiment_count": 2},
        {"zone_id": "zone-2", "experiment_count": 1},
    ]
    assert analytics["experiments_by_creation_date_utc"] == [
        {"date_utc": "2026-01-02", "experiment_count": 2}
    ]
    capacity_summary = analytics["numeric_summaries"]["configured_capacity_connections"]
    assert capacity_summary["observation_count"] == 3
    assert capacity_summary["statistics"] == {
        "minimum": 10.0,
        "maximum": 30.0,
        "mean": 20.0,
        "median": 20.0,
    }
    budget_summary = analytics["numeric_summaries"]["allocation_budget_connections"]
    assert budget_summary["statistics"]["minimum"] == 0.0
    estimate_summary = analytics["numeric_summaries"]["estimated_next_hour_connections"]
    assert estimate_summary["observation_count"] == 3
    assert estimate_summary["statistics"]["mean"] == 8.0
    assert analytics["record_integrity_counts"] == {"verified": 2}

    unfiltered = client.get("/experiments?limit=1&offset=0").json()["analytics"]
    assert unfiltered["total_experiment_count"] == 3
    assert unfiltered["record_integrity_counts"]["invalid"] == 1
    assert unfiltered["experiments_by_zone"] == [
        {"zone_id": "zone-1", "experiment_count": 2},
        {"zone_id": "zone-2", "experiment_count": 1},
    ]
    experiment_routes._store.cache_clear()


def test_experiment_analytics_marks_missing_fields_and_empty_filters(tmp_path, monkeypatch) -> None:
    path = tmp_path / "incomplete-analytics.sqlite3"
    ExperimentStore(path).save("Incomplete", {}, {})
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(path))
    experiment_routes._store.cache_clear()
    client = TestClient(app)

    incomplete = client.get("/experiments").json()["analytics"]
    assert incomplete["total_experiment_count"] == 1
    assert incomplete["numeric_summaries"]["configured_capacity_connections"] == {
        "observation_count": 0,
        "statistics": None,
    }
    assert incomplete["missing_experiment_counts"]["zone_configuration"] == 1
    assert incomplete["missing_experiment_counts"]["allocation_budget"] == 1
    assert incomplete["missing_experiment_counts"]["connection_estimates"] == 1

    empty = client.get("/experiments?zone_id=missing-zone").json()["analytics"]
    assert empty["total_experiment_count"] == 0
    assert empty["experiments_by_zone"] == []
    assert empty["numeric_summaries"]["allocation_budget_connections"] == {
        "observation_count": 0,
        "statistics": None,
    }
    experiment_routes._store.cache_clear()


def test_experiment_analytics_excludes_integrity_invalid_numeric_overflow(tmp_path, monkeypatch):
    path = tmp_path / "overflow-analytics.sqlite3"
    huge = 10**400
    ExperimentStore(path).save(
        "Overflow synthetic record",
        {
            "datetime": "2026-01-01T00:00:00+00:00",
            "zones": [{"zone_id": "zone-overflow", "capacity_connections": huge}],
            "resource_budget_connections": huge,
        },
        {
            "zones": [
                {
                    "zone_id": "zone-overflow",
                    "demand": huge,
                    "effective_capacity": huge,
                    "overload_connections": 0,
                }
            ]
        },
    )
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(path))
    experiment_routes._store.cache_clear()
    analytics = TestClient(app).get("/experiments").json()["analytics"]
    assert analytics["total_experiment_count"] == 1
    assert analytics["record_integrity_counts"] == {"invalid": 1}
    assert analytics["experiments_by_zone"] == []
    assert analytics["numeric_summaries"]["configured_capacity_connections"] == {
        "observation_count": 0,
        "statistics": None,
    }
    assert (
        analytics["missing_experiment_counts"]["invalid_records_excluded_from_field_analytics"] == 1
    )
    experiment_routes._store.cache_clear()


def test_experiment_analytics_handles_malformed_persisted_configuration(tmp_path, monkeypatch):
    path = tmp_path / "malformed-analytics.sqlite3"
    store = ExperimentStore(path)
    record = store.save("Malformed", {}, {})
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE experiments SET configuration_json = ? WHERE id = ?",
            ('{"scenario_kind": []}', record["id"]),
        )
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(path))
    experiment_routes._store.cache_clear()

    response = TestClient(app).get("/experiments")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_count"] == 1
    assert payload["items"][0]["integrity"]["reason_codes"] == ["INTEGRITY_CHECK_FAILED"]
    assert payload["analytics"]["record_integrity_counts"] == {"invalid": 1}
    assert payload["analytics"]["experiments_by_zone"] == []
    experiment_routes._store.cache_clear()


def test_malformed_markdown_report_returns_safe_service_unavailable(tmp_path, monkeypatch) -> None:
    store = ExperimentStore(tmp_path / "malformed-report.sqlite3")
    record = store.save(
        "bad metric",
        {"datetime": "2025-01-01", "zones": [{"zone_id": "z", "capacity_connections": 1}]},
        {
            "zones": [
                {
                    "zone_id": "z",
                    "demand": "bad",
                    "effective_capacity": 1,
                    "overload_connections": 0,
                }
            ]
        },
    )
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(store.path))
    experiment_routes._store.cache_clear()
    response = TestClient(app, raise_server_exceptions=False).get(
        f"/experiments/{record['id']}/report?format=markdown"
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "Experiment report data is invalid."
    experiment_routes._store.cache_clear()
