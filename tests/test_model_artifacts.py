from __future__ import annotations

import json

import pytest

from wifi_tunja_smart_predictor.config import MODEL_INPUT_COLUMNS
from wifi_tunja_smart_predictor.exceptions import ModelMetadataError
from wifi_tunja_smart_predictor.models import artifacts


class _Classifier:
    classes_ = ("LOW", "HIGH")

    def predict(self, _features):
        return ["LOW"]

    def predict_proba(self, _features):
        return [[0.8, 0.2]]


class _Regressor:
    def predict(self, _features):
        return [10.0]


class _UnexpectedClassifier(_Classifier):
    classes_ = ("MEDIUM",)


def _metadata(version: str, model: str) -> dict:
    return {
        "project_version": version,
        "selected_model": model,
        "feature_count": len(MODEL_INPUT_COLUMNS),
        "feature_names": MODEL_INPUT_COLUMNS,
    }


def test_classifier_metadata_requires_version_model_and_feature_contract(tmp_path):
    combined_path = tmp_path / "model.json"
    classifier_path = tmp_path / "classifier.json"
    valid = _metadata("0.3.7", "random_forest")
    combined_path.write_text(json.dumps(valid), encoding="utf-8")
    classifier_path.write_text(json.dumps(valid), encoding="utf-8")

    assert artifacts.load_classifier_metadata(combined_path, classifier_path) == valid

    classifier_path.write_text(json.dumps(_metadata("0.3.6", "random_forest")), encoding="utf-8")
    with pytest.raises(ModelMetadataError, match="inconsistent"):
        artifacts.load_classifier_metadata(combined_path, classifier_path)


@pytest.mark.parametrize("contents", [None, "{ broken", json.dumps({"project_version": "0.3.8"})])
def test_classifier_metadata_rejects_missing_malformed_or_incomplete_file(tmp_path, contents):
    metadata_path = tmp_path / "model.json"
    classifier_path = tmp_path / "classifier.json"
    valid = json.dumps(_metadata("0.3.8", "random_forest"))
    classifier_path.write_text(valid, encoding="utf-8")
    if contents is not None:
        metadata_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ModelMetadataError):
        artifacts.load_classifier_metadata(metadata_path, classifier_path)


@pytest.mark.parametrize("contents", [None, "{ broken", json.dumps({"project_version": "0.3.8"})])
def test_regression_metadata_rejects_missing_malformed_or_incomplete_file(tmp_path, contents):
    metadata_path = tmp_path / "regression.json"
    if contents is not None:
        metadata_path.write_text(contents, encoding="utf-8")

    with pytest.raises(ModelMetadataError):
        artifacts.load_regression_metadata(metadata_path)


def test_runtime_readiness_rejects_classifier_that_cannot_be_deserialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    import wifi_tunja_smart_predictor.models.predict as prediction

    metadata = _metadata("0.3.8", "random_forest")
    combined = {**metadata, "regression": _metadata("0.3.8", "random_forest_regressor")}
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: combined)
    monkeypatch.setattr(
        artifacts,
        "load_regression_metadata",
        lambda: _metadata("0.3.8", "random_forest_regressor"),
    )
    monkeypatch.setattr(artifacts, "load_regression_model", lambda: _Regressor())
    corrupt_artifact = tmp_path / "classifier.joblib"
    corrupt_artifact.write_text("not a joblib artifact", encoding="utf-8")
    monkeypatch.setattr(prediction, "MODEL_PATH", corrupt_artifact)
    prediction.clear_model_cache()

    assert artifacts.model_artifacts_available() is False


def test_runtime_readiness_rejects_regressor_that_cannot_be_deserialized(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    metadata = _metadata("0.3.8", "random_forest")
    regression = _metadata("0.3.8", "random_forest_regressor")
    monkeypatch.setattr(
        artifacts, "load_classifier_metadata", lambda: {**metadata, "regression": regression}
    )
    monkeypatch.setattr(artifacts, "load_regression_metadata", lambda: regression)
    monkeypatch.setattr(artifacts, "load_model", lambda: _Classifier())
    corrupt_artifact = tmp_path / "regressor.joblib"
    corrupt_artifact.write_text("not a joblib artifact", encoding="utf-8")
    monkeypatch.setattr(artifacts, "REGRESSION_MODEL_PATH", corrupt_artifact)
    artifacts._deserialize_regression.cache_clear()

    assert artifacts.model_artifacts_available() is False
    artifacts._deserialize_regression.cache_clear()


def test_runtime_readiness_rejects_models_with_different_training_versions(
    monkeypatch: pytest.MonkeyPatch,
):
    classifier = {
        **_metadata("0.3.8", "random_forest"),
        "regression": _metadata("0.3.8", "random_forest_regressor"),
    }
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: classifier)
    monkeypatch.setattr(
        artifacts,
        "load_regression_metadata",
        lambda: _metadata("0.3.7", "random_forest_regressor"),
    )

    assert artifacts.model_artifacts_available() is False


def test_runtime_readiness_rejects_inconsistent_combined_regression_feature_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    regression = _metadata("0.3.8", "random_forest_regressor")
    combined = {**_metadata("0.3.8", "random_forest"), "regression": regression.copy()}
    combined["regression"]["feature_count"] -= 1
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: combined)
    monkeypatch.setattr(artifacts, "load_regression_metadata", lambda: regression)

    assert artifacts.model_artifacts_available() is False


def test_runtime_readiness_accepts_loadable_versioned_model_bundle(monkeypatch: pytest.MonkeyPatch):
    classifier = {
        **_metadata("0.3.8", "random_forest"),
        "regression": _metadata("0.3.8", "random_forest_regressor"),
    }
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: classifier)
    monkeypatch.setattr(
        artifacts,
        "load_regression_metadata",
        lambda: _metadata("0.3.8", "random_forest_regressor"),
    )
    monkeypatch.setattr(artifacts, "load_model", lambda: _Classifier())
    monkeypatch.setattr(artifacts, "load_regression_model", lambda: _Regressor())

    assert artifacts.model_artifacts_available() is True


def test_runtime_readiness_rejects_classifier_with_incompatible_labels(
    monkeypatch: pytest.MonkeyPatch,
):
    classifier = {
        **_metadata("0.3.8", "unexpected_classifier"),
        "regression": _metadata("0.3.8", "random_forest_regressor"),
    }
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: classifier)
    monkeypatch.setattr(
        artifacts,
        "load_regression_metadata",
        lambda: _metadata("0.3.8", "random_forest_regressor"),
    )
    monkeypatch.setattr(artifacts, "load_model", lambda: _UnexpectedClassifier())
    monkeypatch.setattr(artifacts, "load_regression_model", lambda: _Regressor())

    assert artifacts.model_artifacts_available() is False
