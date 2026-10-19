from __future__ import annotations

import json

import pytest

from wifi_tunja_smart_predictor.config import MODEL_INPUT_COLUMNS
from wifi_tunja_smart_predictor.exceptions import ModelMetadataError
from wifi_tunja_smart_predictor.models import artifacts
from wifi_tunja_smart_predictor.models.loaded_artifact import (
    LoadedArtifact,
    metadata_matches_artifact,
)


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


def test_classifier_metadata_rejects_different_artifact_fingerprints(tmp_path):
    combined_path = tmp_path / "model.json"
    classifier_path = tmp_path / "classifier.json"
    combined = {
        **_metadata("0.4.3", "random_forest"),
        "artifact_sha256": "a" * 64,
        "artifact_size_bytes": 10,
    }
    classifier = {
        **_metadata("0.4.3", "random_forest"),
        "artifact_sha256": "b" * 64,
        "artifact_size_bytes": 10,
    }
    combined_path.write_text(json.dumps(combined), encoding="utf-8")
    classifier_path.write_text(json.dumps(classifier), encoding="utf-8")

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


def test_runtime_readiness_rejects_classifier_and_regressor_from_different_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    classifier = {
        **_metadata("0.4.2", "random_forest"),
        "training_run_id": "run-a",
        "regression": {
            **_metadata("0.4.2", "random_forest_regressor"),
            "training_run_id": "run-b",
        },
    }
    regressor = {
        **_metadata("0.4.2", "random_forest_regressor"),
        "training_run_id": "run-b",
    }
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: classifier)
    monkeypatch.setattr(artifacts, "load_regression_metadata", lambda: regressor)

    assert artifacts.model_artifacts_available() is False


def test_runtime_readiness_accepts_legacy_bundle_with_matching_training_timestamp(
    monkeypatch: pytest.MonkeyPatch,
):
    training_time = "2026-10-01T00:00:00+00:00"
    classifier = {
        **_metadata("0.3.8", "random_forest"),
        "training_timestamp_utc": training_time,
        "regression": {
            **_metadata("0.3.8", "random_forest_regressor"),
            "training_timestamp_utc": training_time,
        },
    }
    regressor = {
        **_metadata("0.3.8", "random_forest_regressor"),
        "training_timestamp_utc": training_time,
    }
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: classifier)
    monkeypatch.setattr(artifacts, "load_regression_metadata", lambda: regressor)
    monkeypatch.setattr(artifacts, "load_model", lambda: _Classifier())
    monkeypatch.setattr(artifacts, "load_regression_model", lambda: _Regressor())

    assert artifacts.model_artifacts_available() is True


def test_regression_artifact_identity_matches_loaded_bytes(tmp_path):
    import hashlib

    import joblib

    artifact_path = tmp_path / "regressor.joblib"
    joblib.dump({"selected": "test-regressor"}, artifact_path)

    loaded = artifacts.load_regression_artifact(artifact_path)

    assert loaded.estimator == {"selected": "test-regressor"}
    assert loaded.sha256 == hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    assert loaded.size_bytes == artifact_path.stat().st_size
    artifacts._deserialize_regression.cache_clear()


def test_artifact_metadata_verification_accepts_legacy_and_rejects_partial_or_mismatched():
    artifact = LoadedArtifact(estimator=object(), sha256="a" * 64, size_bytes=128)

    assert metadata_matches_artifact({}, artifact)
    assert metadata_matches_artifact(
        {"artifact_sha256": "a" * 64, "artifact_size_bytes": 128}, artifact
    )
    assert not metadata_matches_artifact({"artifact_sha256": "a" * 64}, artifact)
    assert not metadata_matches_artifact(
        {"artifact_sha256": "b" * 64, "artifact_size_bytes": 128}, artifact
    )
    assert not metadata_matches_artifact(
        {"artifact_sha256": "a" * 64, "artifact_size_bytes": 127}, artifact
    )


@pytest.mark.parametrize("mismatch", ["classifier", "regressor"])
def test_runtime_readiness_rejects_artifact_fingerprint_mismatch(monkeypatch, mismatch):
    import wifi_tunja_smart_predictor.models.predict as prediction

    classifier_artifact = LoadedArtifact(_Classifier(), "a" * 64, 10)
    regressor_artifact = LoadedArtifact(_Regressor(), "b" * 64, 12)
    classifier_hash = "c" * 64 if mismatch == "classifier" else "a" * 64
    regressor_hash = "c" * 64 if mismatch == "regressor" else "b" * 64
    classifier_metadata = {
        **_metadata("0.4.3", "random_forest"),
        "artifact_sha256": classifier_hash,
        "artifact_size_bytes": 10,
        "regression": {
            **_metadata("0.4.3", "random_forest_regressor"),
            "artifact_sha256": regressor_hash,
            "artifact_size_bytes": 12,
        },
    }
    regressor_metadata = classifier_metadata["regression"]
    monkeypatch.setattr(artifacts, "load_classifier_metadata", lambda: classifier_metadata)
    monkeypatch.setattr(artifacts, "load_regression_metadata", lambda: regressor_metadata)
    monkeypatch.setattr(prediction, "load_model_artifact", lambda: classifier_artifact)
    monkeypatch.setattr(artifacts, "load_regression_artifact", lambda: regressor_artifact)

    assert artifacts.model_artifacts_available() is False
