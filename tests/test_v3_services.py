"""V3 scenario and deterministic assistant coverage."""

from datetime import datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from wifi_tunja_smart_predictor.assistant.queries import DatasetQuery, DatasetQueryEngine
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.exceptions import ScenarioBuildError
from wifi_tunja_smart_predictor.geospatial.locations import LocationResolver
from wifi_tunja_smart_predictor.models.regression import conformal_residual_radius
from wifi_tunja_smart_predictor.scenarios.builder import HistoricalAnalogEngine, ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService


class DummyClassifier:
    classes_ = np.array(["LOW", "HIGH"])

    def predict(self, frame):
        return np.array(["HIGH"] * len(frame))

    def predict_proba(self, frame):
        return np.tile([0.2, 0.8], (len(frame), 1))


class DummyRegressor:
    def predict(self, frame):
        return np.full(len(frame), 80.0)


def test_resolver_zone_and_coordinates(mini_frame):
    resolver = LocationResolver(mini_frame)
    by_zone = resolver.resolve(zone="downtown")
    by_coordinates = resolver.resolve(latitude=5.5353, longitude=-73.3678)
    assert by_zone.zone_id == "ZONE_01"
    assert by_coordinates.distance_to_ap_km == pytest.approx(0)


def test_historical_analogs_exclude_future_rows_and_are_deterministic(mini_frame):
    resolver = LocationResolver(mini_frame)
    location = resolver.resolve(zone_id="ZONE_01")
    when = datetime(2024, 6, 2, 0)
    engine = HistoricalAnalogEngine(mini_frame)
    rows, strategy = engine.select(location, when)
    assert (rows["timestamp"] < when).all()
    assert len(rows) > 0
    assert engine.select(location, when)[1] == strategy
    with pytest.raises(ScenarioBuildError, match="No earlier"):
        engine.select(location, datetime(2024, 6, 1, 0))


def test_scenario_prediction_returns_both_models_interval_and_capacity(mini_frame):
    service = ScenarioPredictionService(mini_frame, DummyClassifier(), DummyRegressor())
    result = service.predict(
        ScenarioRequest(datetime=datetime(2026, 9, 30, 18), zone="downtown")
    ).to_dict()
    assert result["scenario"]["scenario_mode"] == "SYNTHETIC_SCENARIO"
    assert result["classification"]["predicted_demand_level"] == "HIGH"
    assert result["regression"]["predicted_connections_next_hour"] == 80
    assert result["regression"]["prediction_interval_lower"] <= 80
    assert result["regression"]["prediction_interval_upper"] >= 80
    assert "not causes" in result["explanation"]["disclaimer"].lower()
    assert "demand_level" not in result["scenario"].get("features", {})


def test_assistant_is_deterministic_and_supports_followup_context(mini_frame):
    assistant = AssistantService(mini_frame)
    first = assistant.handle(ChatRequest(message="I am in downtown", session_id="test"))
    followup = assistant.handle(
        ChatRequest(message="What about 7 PM?", session_id=first.session_id)
    )
    assert first.intent == "LOCATION_INFO"
    assert "Synthetic" in first.answer
    assert followup.intent == "PREDICTION"
    assert followup.scenario is not None
    assert followup.scenario["scenario_mode"] == "SYNTHETIC_SCENARIO"


def test_assistant_answers_access_point_count(mini_frame):
    response = AssistantService(mini_frame).handle(
        ChatRequest(message="How many synthetic access points are in the dataset?")
    )
    assert response.intent == "DATA_QUERY"
    assert "2 simulated access points" in response.answer


def test_assistant_reports_historical_synthetic_zone_and_recent_window(mini_frame):
    assistant = AssistantService(mini_frame)
    historic = assistant.handle(
        ChatRequest(message="Which synthetic zone historically has highest demand at 7 PM?")
    )
    recent = assistant.handle(ChatRequest(message="What happened in the previous hours?"))
    assert historic.intent == "DATA_QUERY"
    assert "Synthetic Downtown Core" in historic.answer
    assert recent.intent == "DATA_QUERY"
    assert "hourly groups" in recent.answer


def test_assistant_supports_methodology_and_available_locations(mini_frame):
    assistant = AssistantService(mini_frame)
    methodology = assistant.handle(ChatRequest(message="Explain the methodology"))
    zones = assistant.handle(ChatRequest(message="List available zones"))
    assert methodology.intent == "METHODOLOGY"
    assert "ordered train/validation/test" in methodology.answer
    assert "Synthetic Downtown Core" in zones.answer


def test_dataset_query_allowlist_and_safe_aggregate(mini_frame):
    rows = DatasetQueryEngine(mini_frame).execute(
        DatasetQuery(metric="mean", column="connections_next_hour", group_by=["hour"])
    )
    assert len(rows) == 24
    with pytest.raises(ValidationError):
        DatasetQuery(metric="mean", column="__import__", group_by=[])


def test_conformal_radius_uses_validation_residual_quantile():
    radius = conformal_residual_radius([0, 1, 2, 3, 4, 5, 6, 7, 8], [0] * 9, confidence=0.8)
    assert radius == 7
    with pytest.raises(ValueError, match="confidence"):
        conformal_residual_radius([1], [0], confidence=1)


def test_assistant_api_health_chat_and_safe_dataset_query(mini_frame, monkeypatch):
    import assistant_api.main as assistant_api

    service = AssistantService(mini_frame)
    monkeypatch.setattr(assistant_api, "_assistant", lambda: service)
    client = TestClient(assistant_api.app)
    assert client.get("/health").json()["llm_enabled"] is False
    response = client.post("/chat", json={"message": "How many rows are in the dataset?"})
    assert response.status_code == 200
    assert "48 rows" in response.json()["answer"]
    aggregate = client.post(
        "/dataset/query",
        json={"metric": "mean", "column": "connections_next_hour", "group_by": ["hour"]},
    )
    assert aggregate.status_code == 200
    assert len(aggregate.json()["result"]) == 24
    unsafe = client.post("/dataset/query", json={"metric": "mean", "column": "__import__"})
    assert unsafe.status_code == 422


def test_main_api_scenario_endpoint_uses_scenario_service(mini_frame, monkeypatch):
    import api.routes.prediction as routes
    from api.main import app as main_api

    service = ScenarioPredictionService(mini_frame, DummyClassifier(), DummyRegressor())
    monkeypatch.setattr(routes, "_scenario_service", lambda: service)
    client = TestClient(main_api)
    response = client.post(
        "/scenario/predict",
        json={"datetime": "2026-09-30T18:00:00-05:00", "location": {"zone": "downtown"}},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["scenario"]["scenario_mode"] == "SYNTHETIC_SCENARIO"
    assert payload["regression"]["predicted_connections_next_hour"] == 80
