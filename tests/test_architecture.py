from __future__ import annotations

import json
import sqlite3
from io import BytesIO
from urllib.error import HTTPError, URLError

from fastapi.testclient import TestClient

import api.main
import wifi_tunja_smart_predictor.architecture as architecture


class FakeResponse:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def getcode(self):
        return self.status

    def read(self, _size):
        return json.dumps({"status": "ready"}).encode()


def test_assistant_probe_marks_unconfigured_and_unreachable_as_unavailable(monkeypatch):
    unconfigured = architecture.probe_assistant_api(None)
    assert unconfigured["status"] == "unknown"
    assert unconfigured["check_status"] == "unavailable"

    def unreachable(*_args, **_kwargs):
        raise URLError("private host detail")

    monkeypatch.setattr(architecture, "urlopen", unreachable)
    result = architecture.probe_assistant_api("http://assistant-api:8001")
    assert result["status"] == "unknown"
    assert result["check_status"] == "unavailable"
    assert "private host detail" not in result["message"]


def test_assistant_probe_distinguishes_not_ready_from_invalid_configuration(monkeypatch):
    def not_ready(*_args, **_kwargs):
        raise HTTPError("http://assistant-api/ready", 503, "not ready", {}, BytesIO())

    monkeypatch.setattr(architecture, "urlopen", not_ready)
    result = architecture.probe_assistant_api("http://assistant-api:8001")
    assert result["status"] == "not_ready"
    assert result["check_status"] == "verified"

    invalid = architecture.probe_assistant_api("http://user:secret@assistant-api:8001")
    assert invalid["status"] == "unknown"
    assert invalid["check_status"] == "unavailable"
    assert "secret" not in json.dumps(invalid)


def test_assistant_probe_accepts_only_a_valid_readiness_response(monkeypatch):
    monkeypatch.setattr(architecture, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    assert architecture.probe_assistant_api("http://assistant-api:8001") == {
        "status": "available",
        "check_status": "verified",
        "message": "The assistant API readiness route reports ready.",
    }

    class InvalidResponse(FakeResponse):
        def read(self, _size):
            return b"not json"

    monkeypatch.setattr(architecture, "urlopen", lambda *_args, **_kwargs: InvalidResponse())
    invalid = architecture.probe_assistant_api("http://assistant-api:8001")
    assert invalid["status"] == "failed"
    assert invalid["check_status"] == "verified"


def test_sqlite_component_status_distinguishes_missing_and_corrupt_stores(tmp_path):
    missing = architecture._inspect_sqlite_store(tmp_path / "missing.sqlite3")
    assert missing["status"] == "unavailable"
    assert missing["check_status"] == "unavailable"

    corrupt_path = tmp_path / "corrupt.sqlite3"
    corrupt_path.write_text("not a database", encoding="utf-8")
    corrupt = architecture._inspect_sqlite_store(corrupt_path)
    assert corrupt["status"] == "failed"
    assert corrupt["check_status"] == "verified"


def test_architecture_diagnostics_api_reports_checked_and_unavailable_components(
    tmp_path, monkeypatch
):
    experiment_db = tmp_path / "experiments.sqlite3"
    with sqlite3.connect(experiment_db):
        pass
    official_db = tmp_path / "official.sqlite3"
    with sqlite3.connect(official_db):
        pass

    monkeypatch.setattr(architecture, "experiment_database_path", lambda: experiment_db)
    monkeypatch.setattr(architecture.mintic, "_database_path", lambda: official_db)
    monkeypatch.setattr(
        architecture.mintic, "load_snapshot", lambda: {"stale": False, "records": []}
    )
    monkeypatch.setattr(architecture.deployment, "runtime_assets_available", lambda: True)
    monkeypatch.setattr(architecture.deployment, "model_artifacts_available", lambda: True)
    monkeypatch.setattr(
        architecture,
        "probe_assistant_api",
        lambda _url: {
            "status": "available",
            "check_status": "verified",
            "message": "The assistant API readiness route reports ready.",
        },
    )
    monkeypatch.setenv("ASSISTANT_API_URL", "http://user:private-secret@assistant-api:8001")

    response = TestClient(api.main.app).get("/architecture/diagnostics")
    assert response.status_code == 200
    payload = response.json()
    components = {item["id"]: item for item in payload["components"]}
    assert components["main_api"]["status"] == "available"
    assert components["synthetic_ml_pipeline"]["status"] == "available"
    assert components["assistant_api"]["status"] == "available"
    assert components["official_mintic_snapshot"]["status"] == "available_cached"
    assert components["sqlite_persistence"]["status"] == "available"
    assert components["streamlit_dashboard"]["status"] == "unknown"
    assert components["streamlit_dashboard"]["check_status"] == "unavailable"
    assert components["docker_compose"]["status"] == "unknown"
    assert str(tmp_path) not in response.text
    assert "private-secret" not in response.text


def test_architecture_diagnostics_reports_unavailable_dependencies_as_unknown(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(architecture, "experiment_database_path", lambda: tmp_path / "missing.db")
    monkeypatch.setattr(architecture.mintic, "_database_path", lambda: tmp_path / "missing2.db")
    monkeypatch.setattr(architecture.mintic, "load_snapshot", lambda: None)
    monkeypatch.setattr(architecture.deployment, "runtime_assets_available", lambda: False)
    monkeypatch.setattr(architecture.deployment, "model_artifacts_available", lambda: False)
    monkeypatch.setattr(
        architecture,
        "probe_assistant_api",
        lambda _url: {
            "status": "unknown",
            "check_status": "unavailable",
            "message": "The configured assistant readiness check could not reach the service.",
        },
    )
    monkeypatch.delenv("ASSISTANT_API_URL", raising=False)

    payload = TestClient(api.main.app).get("/architecture/diagnostics").json()
    components = {item["id"]: item for item in payload["components"]}
    assert components["assistant_api"]["status"] == "unknown"
    assert components["assistant_api"]["check_status"] == "unavailable"
    assert components["synthetic_ml_pipeline"]["status"] == "not_ready"
    assert components["official_mintic_snapshot"]["status"] == "unavailable"
    assert components["sqlite_persistence"]["status"] == "unavailable"
