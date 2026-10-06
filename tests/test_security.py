"""Security regression checks for bounded schemas, query allowlists, and API middleware."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from api.security import _cors_origins
from wifi_tunja_smart_predictor.assistant.queries import DatasetQuery
from wifi_tunja_smart_predictor.assistant.schemas import AssistantContext, ChatRequest
from wifi_tunja_smart_predictor.assistant.service import ChatSessionStore


def test_chat_schema_rejects_empty_oversized_and_unsafe_session_values():
    for message in ("", " \t\n ", "x" * 1001):
        with pytest.raises(ValidationError):
            ChatRequest(message=message)
    with pytest.raises(ValidationError, match="session_id"):
        ChatRequest(message="hello", session_id="../secret")


def test_chat_schema_accepts_normal_unicode_and_boundary_length():
    assert ChatRequest(message="  ¿Qué puedes hacer?  ").message.startswith("  ¿Qué")
    assert len(ChatRequest(message="x" * 1000).message) == 1000


@pytest.mark.parametrize(
    "context",
    [
        {"dashboard_section": "Unlisted panel"},
        {"unexpected": "value"},
        {"latitude": 5.5},
        {"zone_id": "../../model"},
    ],
)
def test_context_rejects_unrecognized_or_malformed_values(context):
    with pytest.raises(ValidationError):
        AssistantContext.model_validate(context)


def test_scenario_context_is_bounded_and_validates_result_envelope():
    valid = {
        "scenario": {},
        "classification": {"predicted_demand_level": "HIGH", "probability_high": 0.7},
        "regression": {"predicted_connections_next_hour": 42.0},
    }
    assert AssistantContext(scenario_result=valid).scenario_result == valid
    with pytest.raises(ValidationError, match="result structure"):
        AssistantContext(scenario_result={"arbitrary": {"data": "x"}})
    unsafe = {
        "scenario": {},
        "classification": {},
        "regression": {},
        "explanation": {"top_factors": [{"label": "<script>alert(1)</script>"}]},
    }
    with pytest.raises(ValidationError, match="unsafe factor label"):
        AssistantContext(scenario_result=unsafe)
    inconsistent = {
        "scenario": {},
        "classification": {},
        "regression": {"prediction_interval_lower": 25, "prediction_interval_upper": 12},
    }
    with pytest.raises(ValidationError, match="inconsistent"):
        AssistantContext(scenario_result=inconsistent)
    nested = {"x": "too deep"}
    for _ in range(8):
        nested = {"x": nested}
    with pytest.raises(ValidationError, match="nesting"):
        AssistantContext(
            scenario_result={"scenario": nested, "classification": {}, "regression": {}}
        )


@pytest.mark.parametrize(
    "filters",
    [
        {"hour": 24},
        {"year": 2030},
        {"is_weekend": True},
        {"demand_level": "MEDIUM"},
        {"time_period": "MIDNIGHT"},
        {"hour": {"$gt": 2}},
        {"unknown_column": 1},
        {"date_start": "not-a-date"},
        {"date_start": "2025-01-01", "date_end": "2024-12-31"},
        {"zone_id": "../../data"},
    ],
)
def test_dataset_query_rejects_unsupported_filter_values(filters):
    with pytest.raises(ValidationError):
        DatasetQuery(filters=filters)


def test_dataset_query_rejects_expression_columns_and_duplicate_groups():
    with pytest.raises(ValidationError):
        DatasetQuery(metric="mean", column="__import__")
    with pytest.raises(ValidationError, match="unique"):
        DatasetQuery(group_by=["hour", "hour"])


def test_cors_origins_are_explicit_and_configurable(monkeypatch):
    monkeypatch.delenv("CORS_ALLOWED_ORIGINS", raising=False)
    assert _cors_origins() == ["http://localhost:8501", "http://127.0.0.1:8501"]
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://dashboard.example, https://admin.example")
    assert _cors_origins() == ["https://dashboard.example", "https://admin.example"]
    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "*")
    with pytest.raises(ValueError, match=r"explicit HTTP\(S\) origins"):
        _cors_origins()


def test_configured_cors_origin_is_applied_to_a_new_app(monkeypatch):
    from fastapi import FastAPI

    from api.security import install_api_security

    monkeypatch.setenv("CORS_ALLOWED_ORIGINS", "https://dashboard.example")
    app = FastAPI()
    app.get("/health")(lambda: {"status": "ok"})
    install_api_security(app, max_body_bytes=4096)
    client = TestClient(app)

    allowed = client.get("/health", headers={"Origin": "https://dashboard.example"})
    denied = client.get("/health", headers={"Origin": "https://other.example"})
    assert allowed.headers["access-control-allow-origin"] == "https://dashboard.example"
    assert "access-control-allow-origin" not in denied.headers


def test_assistant_session_store_evicts_at_its_configured_bound():
    sessions = ChatSessionStore(max_sessions=2)
    sessions.put("first", {"zone": "ZONE_01"})
    sessions.put("second", {"zone": "ZONE_02"})
    sessions.put("third", {"zone": "ZONE_03"})

    assert len(sessions._items) == 2
    assert sessions.get("first") == {}


def test_main_api_headers_cors_docs_and_sanitized_validation():
    from api.main import app

    client = TestClient(app)
    response = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "strict-origin-when-cross-origin"
    assert "access-control-allow-credentials" not in response.headers
    assert client.get("/docs").status_code == 200
    invalid = client.post("/predict", json={"hour": 99, "secret_value": "dont-echo"})
    assert invalid.status_code == 422
    assert "dont-echo" not in invalid.text
    assert "secret_value" not in invalid.text
    too_large = client.post("/predict", content=b"{" + b" " * (256 * 1024))
    assert too_large.status_code == 413


def test_assistant_api_bodies_headers_cors_and_safe_query_error(mini_frame, monkeypatch):
    import assistant_api.main as assistant_api
    from wifi_tunja_smart_predictor.assistant.service import AssistantService

    monkeypatch.setattr(assistant_api, "_assistant", lambda: AssistantService(mini_frame))
    client = TestClient(assistant_api.app)
    normal = client.post("/chat", json={"message": "hola"})
    assert normal.status_code == 200
    assert normal.json()["intent"] == "GREETING"

    oversized = client.post("/chat", content=b"{" + b" " * (64 * 1024))
    assert oversized.status_code == 413

    invalid = client.post("/chat", json={"message": "", "ignored": "dont-echo"})
    assert invalid.status_code == 422
    assert "dont-echo" not in invalid.text

    context = client.post(
        "/chat", json={"message": "Explain this", "context": {"dashboard_section": "Secret"}}
    )
    assert context.status_code == 422
    too_long = client.post("/chat", json={"message": "x" * 1001})
    assert too_long.status_code == 422

    query = client.post("/dataset/query", json={"filters": {"hour": 24}})
    assert query.status_code == 422
    assert "24" not in query.text
    for unsafe_query in (
        {"metric": "mean", "column": "__import__"},
        {"metric": "exec"},
        {"filters": {"hour": {"$gt": 4}}},
    ):
        unsafe = client.post("/dataset/query", json=unsafe_query)
        assert unsafe.status_code == 422

    allowed = client.get("/health", headers={"Origin": "http://localhost:8501"})
    denied = client.get("/health", headers={"Origin": "https://untrusted.example"})
    assert allowed.headers["access-control-allow-origin"] == "http://localhost:8501"
    assert "access-control-allow-origin" not in denied.headers
    assert allowed.headers["permissions-policy"] == "camera=(), microphone=(), geolocation=()"
    assert client.get("/docs").status_code == 200


def test_unexpected_assistant_failure_does_not_expose_internal_paths(monkeypatch):
    import assistant_api.main as assistant_api

    class FailingAssistant:
        def handle(self, request):
            raise RuntimeError(r"C:\private\models\classifier.joblib")

    monkeypatch.setattr(assistant_api, "_assistant", lambda: FailingAssistant())
    response = TestClient(assistant_api.app).post("/chat", json={"message": "hello"})

    assert response.status_code == 500
    assert "classifier.joblib" not in response.text
    assert "C:\\private" not in response.text


def test_locations_endpoint_returns_the_local_synthetic_catalog(monkeypatch):
    import api.routes.prediction as routes
    from api.main import app

    monkeypatch.setattr(
        routes,
        "_location_catalog",
        lambda: [{"wifi_id": "WIFI_01", "zone_id": "ZONE_01"}],
    )
    response = TestClient(app).get("/locations")
    assert response.status_code == 200
    assert response.json()["locations"][0]["zone_id"] == "ZONE_01"
