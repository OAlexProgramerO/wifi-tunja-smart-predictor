from fastapi.testclient import TestClient

from api.main import app as main_app
from api.routes import prediction
from assistant_api import main as assistant_main


def _set_runtime_assets(monkeypatch, tmp_path, *, dataset=True, classifier=True, regressor=True):
    paths = {
        "RAW_DATASET_PATH": dataset,
        "MODEL_PATH": classifier,
        "REGRESSION_MODEL_PATH": regressor,
    }
    for name, present in paths.items():
        path = tmp_path / name
        if present:
            path.touch()
        elif path.exists():
            path.unlink()
        monkeypatch.setattr(prediction, name, path)
        monkeypatch.setattr(assistant_main, name, path)


def test_readiness_requires_dataset_and_both_models(monkeypatch, tmp_path):
    _set_runtime_assets(monkeypatch, tmp_path, dataset=False)
    assert prediction.runtime_assets_available() is False

    _set_runtime_assets(monkeypatch, tmp_path)
    assert prediction.runtime_assets_available() is True

    _set_runtime_assets(monkeypatch, tmp_path, regressor=False)
    assert prediction.runtime_assets_available() is False


def test_main_api_readiness_is_sanitized_and_health_contract_remains(monkeypatch, tmp_path):
    _set_runtime_assets(monkeypatch, tmp_path, dataset=False)
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
    _set_runtime_assets(monkeypatch, tmp_path, classifier=False)
    client = TestClient(assistant_main.app)

    response = client.get("/ready")
    assert response.status_code == 503
    assert str(tmp_path) not in response.text
    assert client.get("/health").json()["status"] == "ok"

    _set_runtime_assets(monkeypatch, tmp_path)
    assert client.get("/ready").json() == {"status": "ready"}
