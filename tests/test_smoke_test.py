from __future__ import annotations

from scripts import smoke_test


def _stub_services(monkeypatch, *, main_ready="ready"):
    def main_request(path, _payload=None):
        responses = {
            "/health": {"status": "ok"},
            "/ready": {"status": main_ready},
            "/model-info": {"model_loaded": True, "regression_model_loaded": True},
            "/predict": {"prediction": "HIGH"},
            "/scenario/predict": {"classification": {"predicted_demand_level": "HIGH"}},
        }
        return responses[path]

    def assistant_request(path, _payload=None):
        responses = {
            "/health": {"status": "ok"},
            "/ready": {"status": "ready"},
            "/chat": {"intent": "GREETING", "answer": "Hello!"},
        }
        return responses[path]

    monkeypatch.setattr(smoke_test, "main_api_request", main_request)
    monkeypatch.setattr(smoke_test, "assistant_api_request", assistant_request)
    monkeypatch.setattr(smoke_test, "_dashboard_health", lambda _url: None)


def test_smoke_test_passes_when_all_documented_http_checks_succeed(monkeypatch, capsys):
    _stub_services(monkeypatch)

    assert smoke_test.run_smoke_tests("http://127.0.0.1:8501") == 0
    output = capsys.readouterr().out
    assert "PASS Standard prediction" in output
    assert "PASS Dashboard HTTP and backend network" in output


def test_smoke_test_returns_nonzero_for_unready_service(monkeypatch, capsys):
    _stub_services(monkeypatch, main_ready="not_ready")

    assert smoke_test.run_smoke_tests("http://127.0.0.1:8501") == 1
    assert "FAIL Main API readiness" in capsys.readouterr().err
