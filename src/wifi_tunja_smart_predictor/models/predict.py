"""Inference helpers shared by the API and the dashboard."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    MODEL_PATH,
    TARGET_LABELS,
)
from wifi_tunja_smart_predictor.exceptions import ModelNotFoundError, PredictionError

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model(path: str | None = None) -> Any:
    """Load the persisted sklearn pipeline (preprocessing + classifier)."""
    model_path = Path(path) if path else MODEL_PATH
    if not model_path.is_file():
        raise ModelNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Train it from the repository root with: python scripts/train_model.py"
        )
    logger.info("Loading model from %s", model_path)
    return joblib.load(model_path)


def _as_frame(payload: pd.DataFrame | dict[str, Any]) -> pd.DataFrame:
    if isinstance(payload, pd.DataFrame):
        frame = payload.copy()
    else:
        frame = pd.DataFrame([payload])
    missing = [col for col in MODEL_INPUT_COLUMNS if col not in frame.columns]
    if missing:
        raise PredictionError(f"Missing required prediction fields: {missing}")
    return frame[MODEL_INPUT_COLUMNS]


def predict_demand(
    payload: pd.DataFrame | dict[str, Any],
    model: Any | None = None,
) -> list[str]:
    """Return LOW/HIGH labels for each input row."""
    estimator = model if model is not None else load_model()
    frame = _as_frame(payload)
    predictions = [str(label) for label in estimator.predict(frame)]
    invalid = [label for label in predictions if label not in TARGET_LABELS]
    if invalid:
        raise PredictionError(f"Model returned unexpected labels: {invalid}")
    return predictions


def predict_probability(
    payload: pd.DataFrame | dict[str, Any],
    model: Any | None = None,
) -> list[dict[str, float]] | None:
    """Return class probabilities when the estimator supports ``predict_proba``.

    Returns None when probabilities are not available rather than inventing them.
    """
    estimator = model if model is not None else load_model()
    if not hasattr(estimator, "predict_proba"):
        return None
    frame = _as_frame(payload)
    proba = estimator.predict_proba(frame)
    classes = [str(c) for c in estimator.classes_]
    return [
        {classes[i]: float(row[i]) for i in range(len(classes))}
        for row in proba
    ]


def clear_model_cache() -> None:
    """Drop the cached pipeline (used by tests)."""
    load_model.cache_clear()
