"""Regression training, metrics, and split-conformal residual calibration."""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    RANDOM_SEED,
    REGRESSION_TARGET_COLUMN,
)
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline

logger = logging.getLogger(__name__)


def regression_specs() -> list[dict[str, Any]]:
    """Return bounded regression candidates for next-hour connection counts."""
    return [
        {
            "name": "random_forest_regressor",
            "estimator": RandomForestRegressor(
                n_estimators=100,
                max_depth=14,
                min_samples_leaf=8,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        },
        {
            "name": "hist_gradient_boosting_regressor",
            "estimator": HistGradientBoostingRegressor(
                max_iter=140,
                max_leaf_nodes=15,
                learning_rate=0.08,
                l2_regularization=1.0,
                random_state=RANDOM_SEED,
            ),
        },
    ]


def evaluate_regressor(estimator: Any, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Compute standard regression metrics from an estimator's predictions."""
    actual = np.asarray(target, dtype=float)
    predicted = np.asarray(estimator.predict(features), dtype=float)
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)),
    }


def conformal_residual_radius(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    confidence: float = 0.90,
) -> float:
    """Return the finite-sample split-conformal absolute residual quantile."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be strictly between zero and one")
    residuals = np.abs(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    residuals = residuals[np.isfinite(residuals)]
    if not len(residuals):
        raise ValueError("At least one finite calibration residual is required")
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * confidence)))
    return float(np.sort(residuals)[rank - 1])


def train_regression_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Fit candidates on train, select by validation MAE, and report test metrics."""
    x_train = train[MODEL_INPUT_COLUMNS]
    y_train = pd.to_numeric(train[REGRESSION_TARGET_COLUMN], errors="raise")
    x_validation = validation[MODEL_INPUT_COLUMNS]
    y_validation = pd.to_numeric(validation[REGRESSION_TARGET_COLUMN], errors="raise")
    x_test = test[MODEL_INPUT_COLUMNS]
    y_test = pd.to_numeric(test[REGRESSION_TARGET_COLUMN], errors="raise")

    rows: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for spec in regression_specs():
        name = spec["name"]
        pipeline = build_classifier_pipeline(spec["estimator"], scale_numeric=False)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        fitted[name] = pipeline
        val_metrics = evaluate_regressor(pipeline, x_validation, y_validation)
        test_metrics = evaluate_regressor(pipeline, x_test, y_test)
        rows.append(
            {
                "model": name,
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                **test_metrics,
                "training_time_seconds": round(elapsed, 3),
            }
        )
        logger.info("Finished %s in %.2fs (validation MAE=%.3f)", name, elapsed, val_metrics["mae"])
    comparison = pd.DataFrame(rows).sort_values("validation_mae", kind="stable")
    return comparison, fitted, str(comparison.iloc[0]["model"])
