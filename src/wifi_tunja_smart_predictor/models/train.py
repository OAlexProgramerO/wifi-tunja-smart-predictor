"""Temporal splitting and baseline classifier training."""

from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    POSITIVE_LABEL,
    RANDOM_SEED,
    TARGET_COLUMN,
    TRAIN_END,
    VALIDATION_END,
)
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline

logger = logging.getLogger(__name__)


def temporal_split(
    frame: pd.DataFrame,
    *,
    train_end: str = TRAIN_END,
    validation_end: str = VALIDATION_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by timestamp so later periods never influence earlier ones.

    Train: timestamps <= train_end
    Validation: train_end < timestamps <= validation_end
    Test: timestamps > validation_end
    """
    ordered = frame.copy()
    ordered["timestamp"] = pd.to_datetime(
        ordered["timestamp"], format="mixed", errors="raise"
    )
    ordered = ordered.sort_values("timestamp")
    ts = ordered["timestamp"]
    train_mask = ts <= pd.Timestamp(train_end)
    val_mask = (ts > pd.Timestamp(train_end)) & (ts <= pd.Timestamp(validation_end))
    test_mask = ts > pd.Timestamp(validation_end)
    train = ordered.loc[train_mask]
    validation = ordered.loc[val_mask]
    test = ordered.loc[test_mask]
    if train.empty or test.empty:
        raise ValueError(
            "Temporal split produced an empty train or test set. "
            f"Train rows={len(train)}, validation rows={len(validation)}, test rows={len(test)}."
        )
    logger.info(
        "Temporal split: train=%s (%s -> %s), validation=%s (%s -> %s), test=%s (%s -> %s)",
        len(train),
        train["timestamp"].min(),
        train["timestamp"].max(),
        len(validation),
        validation["timestamp"].min() if len(validation) else None,
        validation["timestamp"].max() if len(validation) else None,
        len(test),
        test["timestamp"].min(),
        test["timestamp"].max(),
    )
    return train, validation, test


def _baseline_specs() -> list[dict[str, Any]]:
    """Return the bounded baseline configurations used in VERSION 0.2."""
    return [
        {
            "name": "logistic_regression",
            "scale_numeric": True,
            "estimator": LogisticRegression(
                max_iter=400,
                random_state=RANDOM_SEED,
                solver="lbfgs",
            ),
        },
        {
            "name": "decision_tree",
            "scale_numeric": False,
            "estimator": DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=20,
                random_state=RANDOM_SEED,
            ),
        },
        {
            "name": "random_forest",
            "scale_numeric": False,
            "estimator": RandomForestClassifier(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        },
        {
            "name": "knn",
            "scale_numeric": True,
            "estimator": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        },
        {
            "name": "naive_bayes",
            "scale_numeric": True,
            "estimator": GaussianNB(),
        },
    ]


def train_baseline_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Fit each baseline on the training period and score validation + test.

    Model selection uses validation F1 for the HIGH class. Test metrics are
    reported for every model and are never used to choose the persisted artifact.
    """
    x_train = train[MODEL_INPUT_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(str)
    x_val = validation[MODEL_INPUT_COLUMNS] if len(validation) else None
    y_val = validation[TARGET_COLUMN].astype(str) if len(validation) else None
    x_test = test[MODEL_INPUT_COLUMNS]
    y_test = test[TARGET_COLUMN].astype(str)

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    val_f1: dict[str, float] = {}

    for spec in _baseline_specs():
        name = spec["name"]
        pipeline = build_classifier_pipeline(
            spec["estimator"], scale_numeric=spec["scale_numeric"]
        )
        logger.info("Training %s", name)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_seconds = time.perf_counter() - started
        fitted[name] = pipeline

        test_metrics = evaluate_classifier(pipeline, x_test, y_test)
        row = {
            "model": name,
            "accuracy": test_metrics["accuracy"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "roc_auc": test_metrics["roc_auc"],
            "training_time_seconds": round(training_seconds, 3),
        }
        if x_val is not None and y_val is not None and len(x_val):
            val_metrics = evaluate_classifier(pipeline, x_val, y_val)
            val_f1[name] = val_metrics["f1"]
            row["validation_f1"] = val_metrics["f1"]
        rows.append(row)
        logger.info("Finished %s in %.2fs (test F1=%.4f)", name, training_seconds, row["f1"])

    comparison = pd.DataFrame(rows)
    if val_f1:
        selected_name = max(val_f1, key=val_f1.get)
    else:
        # Fallback if validation is empty: do not invent a composite score;
        # persist the first trained model and document the limitation.
        selected_name = comparison.iloc[0]["model"]
    logger.info(
        "Selected model '%s' using validation F1 for class %s.",
        selected_name,
        POSITIVE_LABEL,
    )
    return comparison, fitted, selected_name
