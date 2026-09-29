"""FastAPI contract tests using an in-memory trained pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.config import MODEL_INPUT_COLUMNS, TARGET_COLUMN
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    frame = make_mini_frame(40)
    pipeline = build_classifier_pipeline(
        LogisticRegression(max_iter=200, random_state=42), scale_numeric=True
    )
    pipeline.fit(frame[MODEL_INPUT_COLUMNS], frame[TARGET_COLUMN])
    artifact = tmp_path / "model.joblib"
    joblib.dump(pipeline, artifact)

    import wifi_tunja_smart_predictor.config as config_mod
    import wifi_tunja_smart_predictor.models.predict as predict_mod

    monkeypatch.setattr(config_mod, "MODEL_PATH", artifact)
    monkeypatch.setattr(predict_mod, "MODEL_PATH", artifact)
    predict_mod.load_model.cache_clear()

    from api.main import app

    return TestClient(app)


def _valid_payload() -> dict:
    frame = make_mini_frame(8)
    return json.loads(frame[MODEL_INPUT_COLUMNS].iloc[[0]].to_json(orient="records"))[0]


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info(client: TestClient) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert "feature_count" in body
    assert body["model_loaded"] is True


def test_predict_valid(client: TestClient) -> None:
    payload = _valid_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"LOW", "HIGH"}
    if body["prediction_probability"] is not None:
        assert abs(sum(body["prediction_probability"].values()) - 1.0) < 1e-5


def test_predict_invalid_rejected(client: TestClient) -> None:
    payload = _valid_payload()
    payload["hour"] = 99
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
