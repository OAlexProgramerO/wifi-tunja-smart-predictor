"""What-if simulation contracts, deterministic calculations, and API integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

from api.main import app
from api.routes import experiments as experiment_routes
from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.experiments.store import ExperimentStore
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.simulation.metrics import METRIC_REGISTRY
from wifi_tunja_smart_predictor.simulation.schemas import WhatIfSimulationRequest
from wifi_tunja_smart_predictor.simulation.service import WhatIfSimulationService


class FakePredictionService:
    def __init__(self, zone_demand: dict[str, float]) -> None:
        self.zone_demand = zone_demand
        self.calls = []

    def predict(self, request: ScenarioRequest):
        self.calls.append(request)
        demand = self.zone_demand[request.zone_id]
        result = {
            "scenario": {
                "dataset_version": "synthetic-test-v1",
                "dataset_sha256": "d" * 64,
                "dataset_fingerprint_algorithm": "pandas_hash_pandas_object_sha256_v1",
            },
            "classification": {"predicted_demand_level": "HIGH" if demand >= 50 else "LOW"},
            "regression": {"predicted_connections_next_hour": demand},
            "traceability": {
                "application_version": "0.6.0",
                "classifier": {"artifact_sha256": "a" * 64, "model_version": "classifier-v"},
                "regressor": {"artifact_sha256": "b" * 64, "model_version": "regressor-v"},
            },
        }
        return SimpleNamespace(to_dict=lambda: result)


def _request() -> WhatIfSimulationRequest:
    return WhatIfSimulationRequest.model_validate(
        {
            "name": "campus capacity comparison",
            "description": "Synthetic assumptions only",
            "datetime": "2025-01-01T10:00:00",
            "baseline": {
                "name": "baseline",
                "zones": [
                    {"zone_id": "zone-a", "capacity_connections": 40},
                    {"zone_id": "zone-b", "capacity_connections": 0},
                ],
                "resource_budget_connections": 30,
            },
            "alternatives": [
                {
                    "name": "alternative",
                    "zones": [
                        {"zone_id": "zone-a", "capacity_connections": 90},
                        {"zone_id": "zone-b", "capacity_connections": 0},
                    ],
                    "resource_budget_connections": 5,
                }
            ],
        }
    )


def test_what_if_separates_model_demand_from_capacity_and_budget_rules():
    service = WhatIfSimulationService(FakePredictionService({"zone-a": 100.0, "zone-b": 0.0}))
    result = service.run(_request())
    baseline = result.baseline
    comparison = result.comparisons[0]
    alternative = comparison.alternative

    assert result.assessment_status == "assessable"
    assert len(service.prediction_service.calls) == 2
    assert baseline.aggregate.predicted_demand_connections == 100
    assert alternative.aggregate.predicted_demand_connections == 100
    assert comparison.aggregate_differences["predicted_demand_difference_connections"] == 0
    assert baseline.aggregate.allocated_budget_connections <= 30
    assert alternative.aggregate.allocated_budget_connections <= 5
    assert baseline.aggregate.overload_connections == 30
    assert alternative.aggregate.overload_connections == 5
    zero_demand = next(row for row in comparison.zones if row.zone_id == "zone-b")
    assert zero_demand.predicted_demand_difference_pct is None
    assert zero_demand.reason_codes == ["BASELINE_DEMAND_IS_ZERO"]
    assert baseline.aggregate.budget_constraint_satisfied
    assert result.disclaimer.startswith("All locations")


def test_what_if_outputs_are_deterministic_and_metric_registry_is_versioned():
    service = WhatIfSimulationService(FakePredictionService({"zone-a": 80.0, "zone-b": 20.0}))
    first = service.run(_request()).model_dump(mode="json")
    second = service.run(_request()).model_dump(mode="json")
    assert first == second
    assert result_has_metric_registry(first)


def test_what_if_provenance_fingerprints_require_sha256_hex_and_versions():
    reasons = WhatIfSimulationService._provenance_reasons(
        {
            "dataset_sha256": "z" * 64,
            "models": [
                {
                    "classifier": {"artifact_sha256": "x" * 64},
                    "regressor": {"artifact_sha256": "b" * 64, "model_version": "r1"},
                }
            ],
        }
    )
    assert "DATASET_FINGERPRINT_UNAVAILABLE" in reasons
    assert "DATASET_VERSION_UNAVAILABLE" in reasons
    assert "DATASET_FINGERPRINT_ALGORITHM_UNAVAILABLE" in reasons
    assert "CLASSIFIER_FINGERPRINT_UNAVAILABLE" in reasons
    assert "CLASSIFIER_VERSION_UNAVAILABLE" in reasons
    assert "APPLICATION_VERSION_UNAVAILABLE" in reasons


def result_has_metric_registry(result: dict) -> bool:
    return (
        result["metric_registry_version"] == "1.0"
        and result["metric_definitions"]["metrics"] == METRIC_REGISTRY
        and "utilization_proxy_pct" in METRIC_REGISTRY
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data["baseline"].update(
            {"zones": [{"zone_id": "zone-a", "capacity_connections": float("inf")}]}
        ),
        lambda data: data["alternatives"][0].update(
            {
                "zones": [
                    {"zone_id": "different-zone", "capacity_connections": 10},
                    {"zone_id": "zone-b", "capacity_connections": 10},
                ]
            }
        ),
        lambda data: data.update({"unexpected": "field"}),
        lambda data: data["baseline"].update(
            {
                "zones": [
                    {"zone_id": "zone-a", "capacity_connections": 1},
                    {"zone_id": "ZONE-A", "capacity_connections": 2},
                ]
            }
        ),
    ],
)
def test_what_if_request_rejects_invalid_bounds_zone_sets_and_unknown_fields(mutate):
    data = _request().model_dump(mode="json")
    mutate(data)
    with pytest.raises(ValidationError):
        WhatIfSimulationRequest.model_validate(data)


def test_what_if_api_persists_retrieves_and_exports_report(monkeypatch, tmp_path):
    frame = make_mini_frame(8)
    zone_ids = sorted(frame["zone_id"].unique().tolist())[:1]
    fake_service = FakePredictionService({zone_ids[0]: 40.0})
    monkeypatch.setattr(experiment_routes, "_scenario_service", lambda: fake_service)
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(tmp_path / "what-if.sqlite3"))
    experiment_routes._store.cache_clear()
    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/experiments/what-if",
        json={
            "name": "V6 API workbench run",
            "datetime": "2025-01-01T10:00:00",
            "baseline": {
                "name": "baseline",
                "zones": [{"zone_id": zone_ids[0], "capacity_connections": 10}],
            },
            "alternatives": [
                {
                    "name": "more capacity",
                    "zones": [{"zone_id": zone_ids[0], "capacity_connections": 100}],
                }
            ],
        },
    )
    assert response.status_code == 201, response.text
    assert response.headers.get("x-request-id")
    record = response.json()
    assert record["integrity"]["status"] == "verified"
    assert record["configuration"]["scenario_kind"] == "what_if_comparison"
    assert client.get(f"/experiments/{record['id']}").json()["id"] == record["id"]
    json_report = client.get(f"/experiments/{record['id']}/report?format=json")
    assert json_report.status_code == 200
    assert json_report.json()["report_schema_version"] == "1.2"
    assert (
        json_report.json()["scenario_analysis"]["comparisons"][0]["alternative_name"]
        == "more capacity"
    )
    markdown = client.get(f"/experiments/{record['id']}/report?format=markdown")
    assert markdown.status_code == 200
    assert "What-if scenario comparison" in markdown.text
    assert "does not change predicted demand" in markdown.text
    assert client.get("/experiments?limit=10&offset=0").json()["items"][0]["id"] == record["id"]
    experiment_routes._store.cache_clear()


def test_what_if_api_returns_safe_validation_error(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(tmp_path / "invalid-what-if.sqlite3"))
    experiment_routes._store.cache_clear()
    response = TestClient(app, raise_server_exceptions=False).post(
        "/experiments/what-if",
        json={
            "name": "invalid",
            "datetime": "2025-01-01T10:00:00",
            "baseline": {},
            "alternatives": [],
        },
    )
    assert response.status_code == 422
    assert response.headers.get("x-request-id")
    assert "traceback" not in response.text.casefold()
    experiment_routes._store.cache_clear()


def test_existing_deterministic_assistant_still_reads_experiment_history(monkeypatch, tmp_path):
    monkeypatch.setenv("EXPERIMENT_DB_PATH", str(tmp_path / "assistant-v6.sqlite3"))
    store = ExperimentStore(tmp_path / "assistant-v6.sqlite3")
    store.save("historical baseline", {"datetime": "2025-01-01", "zones": []}, {"zones": []})
    answer = AssistantService(frame=make_mini_frame(8)).handle(
        ChatRequest(message="Show experiment history")
    )
    assert answer.intent == "EXPERIMENT_HISTORY"
    assert "historical baseline" in answer.answer


def test_scenario_workbench_renders_without_api_call():
    app = AppTest.from_string(
        "from app.dashboard import render_scenario_workbench\n"
        "from tests.conftest import make_mini_frame\n"
        "render_scenario_workbench(make_mini_frame(8))\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any(widget.label == "Language / Idioma" for widget in app.radio)
    assert any(widget.label == "Synthetic zones (1–12)" for widget in app.multiselect)
