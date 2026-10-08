"""Request correlation, JSON event format, and privacy regression tests."""

from __future__ import annotations

import json
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.main import app as main_app
from api.routes import prediction as prediction_routes
from api.schemas import PredictRequest
from api.security import install_api_security
from assistant_api import main as assistant_api
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.observability.logging import (
    JsonLogFormatter,
    configure_structured_logging,
)
from wifi_tunja_smart_predictor.observability.middleware import current_request_id


def _event_records(caplog, event: str):
    return [
        record
        for record in caplog.records
        if getattr(record, "event_fields", {}).get("event") == event
    ]


def test_request_id_is_generated_returned_logged_and_context_is_reset(caplog):
    caplog.set_level(logging.INFO)
    client = TestClient(main_app)

    response = client.get("/health")

    request_id = response.headers["X-Request-ID"]
    assert response.status_code == 200
    assert len(request_id) == 36
    record = _event_records(caplog, "http.request.completed")[-1]
    fields = record.event_fields
    assert fields["request_id"] == request_id
    assert fields["service"] == "main_api"
    assert fields["route"] == "/health"
    assert fields["status_code"] == 200
    assert fields["duration_ms"] >= 0
    assert current_request_id() is None
    assert json.loads(JsonLogFormatter().format(record))["request_id"] == request_id
    lifecycle_records = [
        item
        for item in _event_records(caplog, "http.request.completed")
        if item.event_fields["request_id"] == request_id
    ]
    assert len(lifecycle_records) == 1


def test_valid_request_id_is_preserved_and_invalid_id_is_replaced(caplog):
    caplog.set_level(logging.INFO)
    client = TestClient(assistant_api.app)

    preserved = client.get("/health", headers={"X-Request-ID": "client.trace-42"})
    replaced = client.get("/health", headers={"X-Request-ID": "unsafe\r\nvalue"})

    assert preserved.headers["X-Request-ID"] == "client.trace-42"
    replacement_id = replaced.headers["X-Request-ID"]
    assert replacement_id != "unsafe\r\nvalue"
    assert len(replacement_id) == 36
    records = _event_records(caplog, "http.request.completed")
    assert records[-2].event_fields["request_id"] == "client.trace-42"
    assert records[-1].event_fields["request_id"] == replacement_id
    assert records[-2].event_fields["service"] == "assistant_api"


def test_oversized_request_id_is_replaced():
    response = TestClient(main_app).get("/health", headers={"X-Request-ID": "x" * 65})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] != "x" * 65
    assert len(response.headers["X-Request-ID"]) == 36


def test_request_ids_do_not_leak_between_sequential_requests(caplog):
    caplog.set_level(logging.INFO)
    client = TestClient(main_app)

    first = client.get("/health", headers={"X-Request-ID": "request-one"})
    second = client.get("/health")

    assert first.headers["X-Request-ID"] == "request-one"
    assert second.headers["X-Request-ID"] != first.headers["X-Request-ID"]
    assert current_request_id() is None


def test_assistant_logs_operation_without_user_message_or_context(caplog):
    caplog.set_level(logging.INFO)
    secret_like_message = "private-user-message-DO-NOT-LOG"

    response = TestClient(assistant_api.app).post(
        "/chat",
        headers={"X-Request-ID": "privacy-check"},
        json={"message": secret_like_message, "session_id": "private-session-DO-NOT-LOG"},
    )

    assert response.status_code == 200
    operations = _event_records(caplog, "assistant.chat.completed")
    assert operations
    assert operations[-1].event_fields["request_id"] == "privacy-check"
    assert operations[-1].event_fields["intent"]
    assert secret_like_message not in caplog.text
    assert "private-session-DO-NOT-LOG" not in caplog.text


def test_dataset_query_operation_logs_only_allowlisted_query_metadata(caplog):
    caplog.set_level(logging.INFO)
    response = TestClient(assistant_api.app).post(
        "/dataset/query",
        headers={"X-Request-ID": "query-check"},
        json={"metric": "count", "group_by": ["zone_type"]},
    )

    assert response.status_code == 200
    event = _event_records(caplog, "assistant.dataset_query.completed")[-1]
    assert event.event_fields["request_id"] == "query-check"
    assert event.event_fields["group_by"] == ["zone_type"]
    assert "rows" not in event.event_fields


def test_historical_assistant_event_is_distinct_from_model_prediction(caplog):
    caplog.set_level(logging.INFO)
    response = TestClient(assistant_api.app).post(
        "/chat",
        headers={"X-Request-ID": "historical-check"},
        json={"message": "How is demand historically downtown?"},
    )

    assert response.status_code == 200
    event = _event_records(caplog, "assistant.historical_query.completed")[-1]
    assert event.event_fields["request_id"] == "historical-check"
    assert event.event_fields["operation"] == "historical_query"


def test_standard_prediction_operation_is_traceable(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    monkeypatch.setattr(prediction_routes, "load_model", lambda: object())
    monkeypatch.setattr(prediction_routes, "predict_demand", lambda _features, model: ["HIGH"])
    monkeypatch.setattr(
        prediction_routes,
        "predict_probability",
        lambda _features, model: [{"LOW": 0.2, "HIGH": 0.8}],
    )
    monkeypatch.setattr(
        prediction_routes,
        "_metadata",
        lambda: {"project_version": "model-test", "selected_model": "test-classifier"},
    )
    payload = PredictRequest.model_construct(
        **{field: None for field in PredictRequest.model_fields}
    )

    result = prediction_routes.predict(payload)

    assert result.prediction == "HIGH"
    event = _event_records(caplog, "prediction.standard.completed")[-1]
    assert event.event_fields["model_version"] == "model-test"
    assert event.event_fields["selected_model"] == "test-classifier"
    assert "request_id" not in event.event_fields


def test_scenario_prediction_operation_is_traceable(caplog, monkeypatch):
    caplog.set_level(logging.INFO)

    class _PredictionResult:
        @staticmethod
        def to_dict():
            return {
                "scenario": {},
                "classification": {},
                "regression": {},
                "capacity": {},
                "explanation": {},
                "disclaimer": "Synthetic data only.",
            }

    class _PredictionService:
        @staticmethod
        def predict(_request):
            return _PredictionResult()

    monkeypatch.setattr(prediction_routes, "_scenario_service", lambda: _PredictionService())
    response = TestClient(main_app).post(
        "/scenario/predict",
        headers={"X-Request-ID": "scenario-check"},
        json={"location": {"zone": "downtown"}, "datetime": "2026-01-12T18:00:00"},
    )

    assert response.status_code == 200
    event = _event_records(caplog, "prediction.scenario.completed")[-1]
    assert event.event_fields["request_id"] == "scenario-check"
    assert event.event_fields["operation"] == "scenario"
    assert event.event_fields["model_version"]


def test_direct_assistant_service_call_has_no_fabricated_request_id(mini_frame, caplog):
    caplog.set_level(logging.INFO)
    response = AssistantService(mini_frame).handle(ChatRequest(message="hi"))

    assert response.intent == "GREETING"
    event = _event_records(caplog, "assistant.chat.completed")[-1]
    assert "request_id" not in event.event_fields


def test_sanitized_internal_error_is_correlated_without_exception_message(caplog, monkeypatch):
    caplog.set_level(logging.INFO)
    private_error = "sensitive-internal-detail"

    def fail_service():
        raise RuntimeError(private_error)

    monkeypatch.setattr(assistant_api, "_assistant", fail_service)
    response = TestClient(assistant_api.app).post(
        "/chat",
        headers={"X-Request-ID": "failure-check"},
        json={"message": "hello"},
    )

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "failure-check"
    assert private_error not in response.text
    assert private_error not in caplog.text
    failures = _event_records(caplog, "assistant.chat.failed")
    assert failures[-1].event_fields["exception_type"] == "RuntimeError"
    assert failures[-1].event_fields["request_id"] == "failure-check"


def test_validation_error_keeps_request_id_and_safe_response(caplog):
    caplog.set_level(logging.INFO)
    response = TestClient(assistant_api.app).post(
        "/chat",
        headers={"X-Request-ID": "validation-check"},
        json={"message": ""},
    )

    assert response.status_code == 422
    assert response.headers["X-Request-ID"] == "validation-check"
    assert response.json()["detail"]["message"] == "Request validation failed."
    assert (
        _event_records(caplog, "http.request.completed")[-1].event_fields["outcome"]
        == "client_error"
    )


def test_logging_configuration_does_not_duplicate_handlers():
    configure_structured_logging()
    root = logging.getLogger()
    original_handlers = tuple(root.handlers)

    configure_structured_logging()

    assert tuple(root.handlers) == original_handlers
    assert logging.getLogger("uvicorn.access").disabled


def test_middleware_adds_correlation_to_body_limit_rejection():
    app = FastAPI()
    install_api_security(app, max_body_bytes=32, service="test_api")
    client = TestClient(app)

    response = client.post(
        "/not-found",
        headers={"X-Request-ID": "bounded-request", "Content-Length": "100"},
        content=b"x" * 100,
    )

    assert response.status_code == 413
    assert response.headers["X-Request-ID"] == "bounded-request"


def test_cors_allows_request_id_header_and_exposes_response_id():
    response = TestClient(main_app).options(
        "/health",
        headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "x-request-id",
        },
    )

    assert response.status_code == 200
    assert "x-request-id" in response.headers["access-control-allow-headers"].lower()

    response = TestClient(main_app).get(
        "/health",
        headers={"Origin": "http://localhost:8501", "X-Request-ID": "cors-request"},
    )
    assert response.headers["X-Request-ID"] == "cors-request"
    assert response.headers["access-control-expose-headers"] == "X-Request-ID"
