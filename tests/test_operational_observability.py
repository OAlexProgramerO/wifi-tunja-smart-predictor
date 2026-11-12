from __future__ import annotations

import json
from io import BytesIO
from urllib.error import URLError

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

import api.routes.observability as diagnostics_route
import wifi_tunja_smart_predictor.observability.middleware as middleware_module
from api.main import app as main_api
from assistant_api.main import app as assistant_api
from wifi_tunja_smart_predictor.observability.metrics import MetricsRegistry


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self):
        return self.status

    def read(self, _size):
        return json.dumps(
            {
                "service": "assistant_api",
                "availability": "available",
                "readiness": "ready",
                "metrics": {
                    "service": "assistant_api",
                    "requests_total": 4,
                    "errors_total": 1,
                    "endpoints": {},
                    "metrics_scope": "process_local_since_start",
                },
            }
        ).encode()


def test_metrics_middleware_counts_routes_statuses_duration_and_omits_request_data(monkeypatch):
    registry = MetricsRegistry()
    monkeypatch.setattr(middleware_module, "request_metrics", registry)
    app = FastAPI()

    @app.get("/items/{item_id}")
    def get_item(item_id: str):
        return {"item": item_id}

    @app.post("/failure")
    def fail():
        return JSONResponse(status_code=503, content={"detail": "private body"})

    app.add_middleware(middleware_module.RequestObservabilityMiddleware, service="main_api")
    client = TestClient(app)
    client.get("/items/42?token=query-secret", headers={"X-Private": "header-secret"})
    client.post("/failure", json={"chat": "body-secret"})
    client.get("/missing?password=unretained")

    snapshot = registry.snapshot("main_api")
    assert snapshot["requests_total"] == 3
    assert snapshot["errors_total"] == 2
    assert snapshot["endpoints"]["/items/{item_id}"]["responses_by_status"] == {"200": 1}
    assert snapshot["endpoints"]["/failure"]["responses_by_status"] == {"503": 1}
    assert snapshot["endpoints"]["unmatched"]["responses_by_status"] == {"404": 1}
    assert snapshot["endpoints"]["/items/{item_id}"]["duration_ms"]["count"] == 1
    serialized = json.dumps(snapshot)
    for secret in ("query-secret", "header-secret", "body-secret", "unretained", "private body"):
        assert secret not in serialized


def test_metrics_registry_aggregates_duration_and_bounds_endpoint_cardinality():
    registry = MetricsRegistry()
    registry.record(service="main_api", route="/health", status_code=200, duration_ms=12.5)
    registry.record(service="main_api", route="/health", status_code=500, duration_ms=3.25)
    for index in range(100):
        registry.record(
            service="main_api", route=f"/custom/{index}", status_code=600, duration_ms=-4
        )

    snapshot = registry.snapshot("main_api")
    assert len(snapshot["endpoints"]) == 64
    assert snapshot["endpoints"]["/health"]["errors_total"] == 1
    assert snapshot["endpoints"]["/health"]["duration_ms"] == {
        "count": 2,
        "total": 15.75,
        "max": 12.5,
    }
    assert snapshot["endpoints"]["other"]["responses_by_status"] == {"500": 38}
    assert snapshot["requests_total"] == 102


def test_assistant_metrics_endpoint_returns_aggregate_only(monkeypatch):
    import assistant_api.observability_routes as assistant_metrics_route

    monkeypatch.setattr(
        assistant_metrics_route,
        "metrics_snapshot",
        lambda _service: {
            "service": "assistant_api",
            "requests_total": 2,
            "errors_total": 0,
            "endpoints": {},
            "metrics_scope": "process_local_since_start",
        },
    )
    monkeypatch.setattr(
        assistant_metrics_route.deployment, "runtime_assets_available", lambda: True
    )
    monkeypatch.setattr(
        assistant_metrics_route.deployment, "model_artifacts_available", lambda: True
    )
    response = TestClient(assistant_api).get("/observability/metrics")
    assert response.status_code == 200
    assert response.json()["availability"] == "available"
    assert response.json()["readiness"] == "ready"
    assert response.json()["metrics"]["requests_total"] == 2


def test_diagnostics_endpoint_reports_reliable_assistant_status_and_metrics(monkeypatch):
    monkeypatch.setenv("ASSISTANT_API_URL", "http://assistant-api:8001")
    monkeypatch.setattr(diagnostics_route, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(diagnostics_route, "_readiness_status", lambda: "ready")
    response = TestClient(main_api).get("/observability/diagnostics")

    assert response.status_code == 200
    services = response.json()["services"]
    assert services["main_api"]["availability"] == "available"
    assert services["assistant_api"]["availability"] == "available"
    assert services["assistant_api"]["readiness"] == "ready"
    assert services["assistant_api"]["metrics"]["requests_total"] == 4
    assert response.json()["privacy"]["request_content_retained"] is False


def test_diagnostics_marks_unreachable_assistant_unavailable_without_leaking_url_or_error(
    monkeypatch,
):
    monkeypatch.setenv("ASSISTANT_API_URL", "http://private-host:9000")
    monkeypatch.setattr(diagnostics_route, "_readiness_status", lambda: "ready")

    def unreachable(*_args, **_kwargs):
        raise URLError("private-host and connection detail")

    monkeypatch.setattr(diagnostics_route, "urlopen", unreachable)
    response = TestClient(main_api).get("/observability/diagnostics")

    assistant = response.json()["services"]["assistant_api"]
    assert assistant["availability"] == "unknown"
    assert assistant["check_status"] == "unavailable"
    assert assistant["metrics"] is None
    assert "private-host" not in response.text
    assert "connection detail" not in response.text


def test_diagnostics_rejects_credential_bearing_assistant_url_without_echoing_secret(monkeypatch):
    monkeypatch.setenv("ASSISTANT_API_URL", "http://user:credential-secret@assistant-api:8001")
    monkeypatch.setattr(diagnostics_route, "_readiness_status", lambda: "ready")
    monkeypatch.setattr(
        diagnostics_route,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not probe")),
    )
    response = TestClient(main_api).get("/observability/diagnostics")

    assistant = response.json()["services"]["assistant_api"]
    assert assistant["availability"] == "unknown"
    assert assistant["check_status"] == "unavailable"
    assert "credential-secret" not in response.text


def test_diagnostics_reports_assistant_http_503_as_not_ready(monkeypatch):
    from urllib.error import HTTPError

    monkeypatch.setenv("ASSISTANT_API_URL", "http://assistant-api:8001")
    monkeypatch.setattr(diagnostics_route, "_readiness_status", lambda: "ready")

    def not_ready(*_args, **_kwargs):
        raise HTTPError(
            "http://assistant-api:8001/observability/metrics", 503, "private", {}, BytesIO()
        )

    monkeypatch.setattr(diagnostics_route, "urlopen", not_ready)
    assistant = (
        TestClient(main_api).get("/observability/diagnostics").json()["services"]["assistant_api"]
    )
    assert assistant["availability"] == "not_ready"
    assert assistant["readiness"] == "not_ready"
    assert "private" not in json.dumps(assistant)


def test_diagnostics_distinguishes_malformed_assistant_response_from_unavailable(monkeypatch):
    class Malformed(FakeResponse):
        def read(self, _size):
            return BytesIO(b"not json").read()

    monkeypatch.setenv("ASSISTANT_API_URL", "http://assistant-api:8001")
    monkeypatch.setattr(diagnostics_route, "urlopen", lambda *_args, **_kwargs: Malformed())
    assistant = (
        TestClient(main_api).get("/observability/diagnostics").json()["services"]["assistant_api"]
    )
    assert assistant["availability"] == "failed"
    assert assistant["check_status"] == "verified"
    assert assistant["metrics"] is None
