"""Model pipeline tests on a tiny in-memory frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from wifi_tunja_smart_predictor.config import MODEL_INPUT_COLUMNS, TARGET_COLUMN, TARGET_LABELS
from wifi_tunja_smart_predictor.exceptions import ModelArtifactError
from wifi_tunja_smart_predictor.models import predict as prediction
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.predict import predict_demand, predict_probability
from wifi_tunja_smart_predictor.models.train import temporal_split
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline


def test_pipeline_fits_and_predicts_labels(mini_frame: pd.DataFrame) -> None:
    x = mini_frame[MODEL_INPUT_COLUMNS]
    y = mini_frame[TARGET_COLUMN]
    pipeline = build_classifier_pipeline(
        LogisticRegression(max_iter=200, random_state=42), scale_numeric=True
    )
    pipeline.fit(x, y)
    preds = pipeline.predict(x)
    assert set(preds).issubset(set(TARGET_LABELS))
    metrics = evaluate_classifier(pipeline, x, y)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["confusion_matrix"] is not None


def test_probabilities_are_valid(mini_frame: pd.DataFrame) -> None:
    x = mini_frame[MODEL_INPUT_COLUMNS]
    y = mini_frame[TARGET_COLUMN]
    pipeline = build_classifier_pipeline(
        LogisticRegression(max_iter=200, random_state=42), scale_numeric=True
    )
    pipeline.fit(x, y)
    proba = predict_probability(x, model=pipeline)
    assert proba is not None
    for row in proba:
        total = sum(row.values())
        assert abs(total - 1.0) < 1e-6
        assert all(0.0 <= v <= 1.0 for v in row.values())
    labels = predict_demand(x, model=pipeline)
    assert set(labels) <= set(TARGET_LABELS)


def test_predict_probability_rejects_invalid_estimator_probabilities(mini_frame):
    class InvalidProbabilityClassifier:
        classes_ = np.asarray(["LOW", "HIGH"])

        def predict_proba(self, features):
            return np.tile([1.2, -0.2], (len(features), 1))

    with pytest.raises(ModelArtifactError, match="invalid probability"):
        predict_probability(mini_frame[MODEL_INPUT_COLUMNS], model=InvalidProbabilityClassifier())


def test_temporal_split_order() -> None:
    # Build a compact frame that crosses configured cutoffs via explicit dates.
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-03-01 00:00:00",
                    "2025-06-30 12:00:00",
                    "2025-07-15 00:00:00",
                    "2025-09-30 12:00:00",
                    "2025-10-01 00:00:00",
                    "2025-12-01 00:00:00",
                ]
            ),
            "x": np.arange(6),
        }
    )
    train, validation, test = temporal_split(frame)
    assert train["timestamp"].max() <= pd.Timestamp("2025-06-30 23:59:59")
    assert test["timestamp"].min() > pd.Timestamp("2025-09-30 23:59:59")
    assert validation["timestamp"].min() > train["timestamp"].max()


def test_classifier_cache_refreshes_when_artifact_changes(tmp_path, monkeypatch) -> None:
    from wifi_tunja_smart_predictor.models.loaded_artifact import LoadedArtifact

    artifact = tmp_path / "classifier.joblib"
    artifact.write_text("first artifact", encoding="utf-8")
    loaded_paths = []
    monkeypatch.setattr(prediction, "MODEL_PATH", artifact)
    monkeypatch.setattr(
        prediction,
        "deserialize_artifact",
        lambda path: loaded_paths.append(path)
        or LoadedArtifact(path.read_text(encoding="utf-8"), "f" * 64, path.stat().st_size),
    )
    prediction.clear_model_cache()

    first = prediction.load_model()
    artifact.write_text("replacement classifier artifact", encoding="utf-8")
    second = prediction.load_model()

    assert first == "first artifact"
    assert second == "replacement classifier artifact"
    assert len(loaded_paths) == 2
    prediction.clear_model_cache()


def test_classifier_artifact_identity_matches_loaded_bytes(tmp_path) -> None:
    import hashlib

    import joblib

    artifact = tmp_path / "classifier.joblib"
    joblib.dump({"selected": "test"}, artifact)

    loaded = prediction.load_model_artifact(str(artifact))

    assert loaded.estimator == {"selected": "test"}
    assert loaded.sha256 == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert loaded.size_bytes == artifact.stat().st_size
