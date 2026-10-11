from fastapi.testclient import TestClient

from api.main import app as main_app
from assistant_api import main as assistant_main
from wifi_tunja_smart_predictor import deployment


def _set_runtime_assets(monkeypatch, tmp_path, *, missing=()):
    paths = tuple(
        tmp_path / f"asset-{index}" for index in range(len(deployment.REQUIRED_DEMO_ASSETS))
    )
    for index, path in enumerate(paths):
        if index not in missing:
            path.touch()
        elif path.exists():
            path.unlink()
    monkeypatch.setattr(deployment, "REQUIRED_DEMO_ASSETS", paths)


def test_readiness_requires_dataset_and_both_models(monkeypatch, tmp_path):
    _set_runtime_assets(monkeypatch, tmp_path, missing=(0,))
    assert deployment.runtime_assets_available() is False

    _set_runtime_assets(monkeypatch, tmp_path)
    assert deployment.runtime_assets_available() is True

    _set_runtime_assets(monkeypatch, tmp_path, missing=(5,))
    assert deployment.runtime_assets_available() is False


def test_main_api_readiness_is_sanitized_and_health_contract_remains(monkeypatch, tmp_path):
    _set_runtime_assets(monkeypatch, tmp_path, missing=(0,))
    client = TestClient(main_app)

    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "message": "Required synthetic data or model artifacts are unavailable.",
    }
    assert str(tmp_path) not in response.text
    assert client.get("/health").status_code == 200

    _set_runtime_assets(monkeypatch, tmp_path)
    assert client.get("/ready").json() == {"status": "ready"}


def test_assistant_readiness_uses_same_assets_and_preserves_health(monkeypatch, tmp_path):
    _set_runtime_assets(monkeypatch, tmp_path, missing=(2,))
    client = TestClient(assistant_main.app)

    response = client.get("/ready")
    assert response.status_code == 503
    assert str(tmp_path) not in response.text
    assert client.get("/health").json()["status"] == "ok"

    _set_runtime_assets(monkeypatch, tmp_path)
    assert client.get("/ready").json() == {"status": "ready"}
