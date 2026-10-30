"""FastAPI contract tests using an in-memory trained pipeline."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

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

    import api.routes.prediction as prediction_routes
    import wifi_tunja_smart_predictor.config as config_mod
    import wifi_tunja_smart_predictor.models.predict as predict_mod

    monkeypatch.setattr(config_mod, "MODEL_PATH", artifact)
    monkeypatch.setattr(predict_mod, "MODEL_PATH", artifact)
    metadata_path = tmp_path / "model_metadata.json"
    classification_metadata_path = tmp_path / "classification_metadata.json"
    metadata = {
        "project_version": "0.3.8",
        "selected_model": "test_classifier",
        "feature_count": len(MODEL_INPUT_COLUMNS),
        "feature_names": MODEL_INPUT_COLUMNS,
        "regression": {
            "project_version": "0.3.8",
            "selected_model": "test_regressor",
        },
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    classification_metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(prediction_routes, "MODEL_METADATA_PATH", metadata_path)
    monkeypatch.setattr(
        prediction_routes, "CLASSIFICATION_METADATA_PATH", classification_metadata_path
    )
    predict_mod.clear_model_cache()

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


def test_model_diagnostics_endpoint_returns_validated_report(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    monkeypatch.setattr(
        prediction_routes,
        "load_valid_diagnostics",
        lambda: {
            "schema_version": "1.0",
            "generated_at_utc": "2026-10-20T10:00:00+00:00",
            "project_version": "0.5.2",
            "model_version": "0.5.0",
            "dataset": {},
            "data_quality": {},
            "artifacts": {},
            "method": {},
            "test_period": {},
            "test": {},
            "rolling_windows": [],
            "warnings": ["synthetic data"],
        },
    )
    response = client.get("/model-diagnostics")
    assert response.status_code == 200
    assert response.json()["schema_version"] == "1.0"
    assert response.json()["warnings"] == ["synthetic data"]


def test_model_diagnostics_endpoint_fails_closed_when_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    def unavailable():
        raise ValueError("stale")

    monkeypatch.setattr(prediction_routes, "load_valid_diagnostics", unavailable)
    response = client.get("/model-diagnostics")
    assert response.status_code == 503
    assert "stale" not in response.text


def test_temporal_evaluation_api_returns_versioned_report(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sklearn.linear_model import LinearRegression

    import api.routes.prediction as prediction_routes
    from tests.conftest import make_mini_frame
    from wifi_tunja_smart_predictor.config import REGRESSION_TARGET_COLUMN

    frame = make_mini_frame(400)
    classifier = build_classifier_pipeline(
        LogisticRegression(max_iter=200, random_state=42), scale_numeric=True
    )
    regressor = build_classifier_pipeline(LinearRegression(), scale_numeric=False)
    classifier.fit(frame[MODEL_INPUT_COLUMNS], frame[TARGET_COLUMN])
    regressor.fit(frame[MODEL_INPUT_COLUMNS], frame[REGRESSION_TARGET_COLUMN])
    dataset = tmp_path / "synthetic.csv"
    dataset.write_text("synthetic test input", encoding="utf-8")
    monkeypatch.setattr(prediction_routes, "PROCESSED_DATASET_PATH", dataset)
    monkeypatch.setattr(prediction_routes, "load_raw_dataset", lambda _path: frame)
    monkeypatch.setattr(
        prediction_routes, "_metadata", lambda: {"preprocessing_version": "scenario-features-v1"}
    )
    monkeypatch.setattr(
        prediction_routes,
        "load_regression_metadata",
        lambda: {"preprocessing_version": "scenario-features-v1"},
    )
    monkeypatch.setattr(prediction_routes, "artifact_matches_metadata", lambda *_: True)
    monkeypatch.setattr(prediction_routes, "model_bundle_metadata_matches", lambda *_: True)
    monkeypatch.setattr(
        prediction_routes,
        "load_model_artifact",
        lambda: SimpleNamespace(estimator=classifier, sha256="a" * 64),
    )
    monkeypatch.setattr(
        prediction_routes,
        "load_regression_artifact",
        lambda _path: SimpleNamespace(estimator=regressor, sha256="b" * 64),
    )

    response = client.post(
        "/evaluation/temporal",
        json={"window_count": 2, "features": ["hour", "zone_type"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["project_version"] == "0.6.5"
    assert body["feature_version"] == "scenario-features-v1"
    assert body["dataset"]["sha256"] == hashlib.sha256(b"synthetic test input").hexdigest()
    assert len(body["windows"]) == 2
    assert body["artifacts"]["classifier_sha256"] == "a" * 64
    markdown = client.post(
        "/evaluation/temporal?report_format=markdown",
        json={"window_count": 2, "features": ["hour"]},
    )
    assert markdown.status_code == 200
    assert markdown.headers["content-type"].startswith("text/markdown")
    assert "Feature distribution diagnostics" in markdown.text


def test_temporal_evaluation_api_rejects_excessive_window_count(client: TestClient) -> None:
    response = client.post("/evaluation/temporal", json={"window_count": 6})
    assert response.status_code == 422
    assert "internal" not in response.text


def test_model_info_rejects_classifier_with_incompatible_prediction_labels(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    class IncompatibleClassifier:
        classes_ = ("MEDIUM",)

        def predict(self, _features):
            return ["MEDIUM"]

        def predict_proba(self, _features):
            return [[1.0]]

    from wifi_tunja_smart_predictor.models.loaded_artifact import LoadedArtifact

    monkeypatch.setattr(
        prediction_routes,
        "load_model_artifact",
        lambda: LoadedArtifact(IncompatibleClassifier(), "a" * 64, 10),
    )

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert response.json()["model_version"] is None


def test_model_info_marks_classifier_unavailable_when_fingerprint_mismatches(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    monkeypatch.setattr(
        prediction_routes,
        "_metadata",
        lambda: {
            "project_version": "0.4.3",
            "selected_model": "test_classifier",
            "artifact_sha256": "0" * 64,
            "artifact_size_bytes": 1,
        },
    )

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert response.json()["model_version"] is None


def test_model_info_rejects_inconsistent_regression_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    class Regressor:
        def predict(self, _features):
            return [1.0]

    monkeypatch.setattr(
        prediction_routes,
        "load_regression_metadata",
        lambda: {
            "project_version": "0.3.7",
            "selected_model": "outdated_regressor",
            "feature_count": len(MODEL_INPUT_COLUMNS),
            "feature_names": MODEL_INPUT_COLUMNS,
        },
    )
    from wifi_tunja_smart_predictor.models.loaded_artifact import LoadedArtifact

    monkeypatch.setattr(
        prediction_routes,
        "load_regression_artifact",
        lambda: LoadedArtifact(Regressor(), "a" * 64, 10),
    )

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["regression_model_loaded"] is False
    assert response.json()["regression_model"] is None


def test_predict_valid(client: TestClient) -> None:
    payload = _valid_payload()
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"LOW", "HIGH"}
    if body["prediction_probability"] is not None:
        assert abs(sum(body["prediction_probability"].values()) - 1.0) < 1e-5


def test_predict_reports_version_from_trained_model_metadata(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    monkeypatch.setattr(
        prediction_routes,
        "_metadata",
        lambda: {"project_version": "0.3.7", "selected_model": "stored_classifier"},
    )

    response = client.post("/predict", json=_valid_payload(), headers={"X-Request-ID": "v42-test"})

    assert response.status_code == 200
    assert response.json()["model_version"] == "0.3.7"
    assert response.json()["selected_model"] == "stored_classifier"
    trace = response.json()["traceability"]
    assert trace["request_id"] == "v42-test"
    assert trace["classifier"]["model_version"] == "0.3.7"
    assert trace["classifier"]["selected_model"] == "stored_classifier"
    assert len(trace["classifier"]["artifact_sha256"]) == 64
    assert trace["classifier"]["artifact_size_bytes"] > 0
    assert trace["regressor"] is None
    timestamp = datetime.fromisoformat(trace["predicted_at_utc"].replace("Z", "+00:00"))
    assert timestamp.tzinfo == timezone.utc


def test_predict_trace_identity_is_sha256_of_loaded_artifact(client: TestClient) -> None:
    import wifi_tunja_smart_predictor.models.predict as predict_mod

    artifact = predict_mod.MODEL_PATH
    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 200
    assert (
        response.json()["traceability"]["classifier"]["artifact_sha256"]
        == hashlib.sha256(artifact.read_bytes()).hexdigest()
    )


def test_predict_rejects_model_bytes_that_disagree_with_recorded_fingerprint(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    monkeypatch.setattr(
        prediction_routes,
        "_metadata",
        lambda: {
            "project_version": "0.4.3",
            "selected_model": "test_classifier",
            "artifact_sha256": "0" * 64,
            "artifact_size_bytes": 99,
        },
    )

    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "model_metadata_unavailable"
    assert "path" not in response.text.lower()


def test_predict_rejects_classifier_with_incompatible_labels(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    import api.routes.prediction as prediction_routes

    class IncompatibleClassifier:
        classes_ = ("MEDIUM",)

        def predict(self, _features):
            return ["LOW"]

        def predict_proba(self, _features):
            return [[1.0]]

    from wifi_tunja_smart_predictor.models.loaded_artifact import LoadedArtifact

    monkeypatch.setattr(
        prediction_routes,
        "load_model_artifact",
        lambda: LoadedArtifact(IncompatibleClassifier(), "a" * 64, 10),
    )

    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "model_unavailable",
            "message": "Required model artifacts are unavailable.",
        }
    }


@pytest.mark.parametrize(
    "metadata_content",
    [
        None,
        "{ malformed json",
        json.dumps({"project_version": "0.3.7", "selected_model": "stored_classifier"}),
    ],
    ids=["missing", "malformed", "incomplete"],
)
def test_predict_rejects_missing_or_invalid_artifact_metadata(
    client: TestClient, metadata_content: str | None
) -> None:
    import api.routes.prediction as prediction_routes

    metadata_path = prediction_routes.MODEL_METADATA_PATH
    if metadata_content is None:
        metadata_path.unlink()
    else:
        metadata_path.write_text(metadata_content, encoding="utf-8")

    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "code": "model_metadata_unavailable",
            "message": "Required model metadata is unavailable or invalid.",
        }
    }
    assert str(metadata_path) not in response.text
    info = client.get("/model-info")
    assert info.status_code == 200
    assert info.json()["model_loaded"] is False
    assert info.json()["model_version"] is None
    assert client.get("/health").json()["status"] == "ok"


def test_predict_invalid_rejected(client: TestClient) -> None:
    payload = _valid_payload()
    payload["hour"] = 99
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_unexpected_prediction_error_is_sanitized_and_logged(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import api.routes.prediction as prediction_routes

    def fail_prediction(*_args, **_kwargs):
        raise RuntimeError("internal test detail")

    monkeypatch.setattr(prediction_routes, "predict_demand", fail_prediction)

    response = client.post("/predict", json=_valid_payload())

    assert response.status_code == 500
    assert "internal test detail" not in response.text
    assert "stack_frames" in caplog.text
