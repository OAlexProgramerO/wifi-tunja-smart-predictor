"""Model pipeline tests on a tiny in-memory frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from wifi_tunja_smart_predictor.config import MODEL_INPUT_COLUMNS, TARGET_COLUMN, TARGET_LABELS
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
