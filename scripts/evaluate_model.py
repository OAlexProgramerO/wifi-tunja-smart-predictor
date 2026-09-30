#!/usr/bin/env python3
"""Evaluate both persisted models and create compact reproducible reports."""

from __future__ import annotations

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from wifi_tunja_smart_predictor.config import (
    CONFUSION_MATRIX_PATH,
    FIGURES_DIR,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    PROCESSED_DATASET_PATH,
    REGRESSION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_PREDICTIONS_FIGURE_PATH,
    REGRESSION_RESIDUALS_FIGURE_PATH,
    REGRESSION_TARGET_COLUMN,
    SELECTED_MODEL_METRICS_PATH,
    SELECTED_REGRESSION_METRICS_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
    TARGET_COLUMN,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.predict import load_model
from wifi_tunja_smart_predictor.models.regression import evaluate_regressor
from wifi_tunja_smart_predictor.models.train import temporal_split
from wifi_tunja_smart_predictor.visualization.plots import plot_confusion_matrix

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Evaluate held-out test rows and write metrics, residual analysis, and figures."""
    frame = load_raw_dataset(PROCESSED_DATASET_PATH)
    _train, _validation, test = temporal_split(frame)
    x_test = test[MODEL_INPUT_COLUMNS]
    y_class = test[TARGET_COLUMN]
    classifier = load_model()
    class_metrics = evaluate_classifier(classifier, x_test, y_class)

    class_payload = {
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "note": "Performance measured on the synthetic temporal test period.",
        "test_rows": int(len(test)),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        **class_metrics,
    }
    SELECTED_MODEL_METRICS_PATH.write_text(
        json.dumps(class_payload, indent=2, default=str), encoding="utf-8"
    )
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(class_metrics["confusion_matrix"], class_metrics["labels"]).savefig(
        CONFUSION_MATRIX_PATH, dpi=140
    )

    regression_meta = json.loads(REGRESSION_METADATA_PATH.read_text(encoding="utf-8"))
    if not REGRESSION_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Regression model missing at {REGRESSION_MODEL_PATH}; run scripts/train_model.py first."
        )
    import joblib

    regressor = joblib.load(REGRESSION_MODEL_PATH)
    y_true = test[REGRESSION_TARGET_COLUMN].astype(float).to_numpy()
    predictions = np.maximum(0.0, np.asarray(regressor.predict(x_test), dtype=float))
    reg_metrics = evaluate_regressor(
        regressor, x_test, test[REGRESSION_TARGET_COLUMN].astype(float)
    )
    radius = float(regression_meta["calibration_radius"])
    lower = np.maximum(0.0, predictions - radius)
    upper = predictions + radius
    residuals = y_true - predictions
    errors = pd.DataFrame(
        {
            "zone_type": test["zone_type"].to_numpy(),
            "hour": test["hour"].to_numpy(),
            "absolute_error": np.abs(residuals),
            "actual": y_true,
            "predicted": predictions,
        }
    )
    reg_payload = {
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "note": "Regression performance measured on the synthetic temporal test period.",
        "test_rows": int(len(test)),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
        **reg_metrics,
        "interval_confidence": regression_meta["interval_confidence"],
        "interval_radius": radius,
        "interval_test_coverage": float(np.mean((y_true >= lower) & (y_true <= upper))),
        "mean_interval_width": float(np.mean(upper - lower)),
        "mean_absolute_error_by_zone_type": errors.groupby("zone_type")["absolute_error"]
        .mean()
        .round(3)
        .to_dict(),
        "mean_absolute_error_by_hour": errors.groupby("hour")["absolute_error"]
        .mean()
        .round(3)
        .to_dict(),
        "residual_quantiles": {
            str(q): float(np.quantile(np.abs(residuals), q)) for q in (0.5, 0.9, 0.95)
        },
    }
    SELECTED_REGRESSION_METRICS_PATH.write_text(
        json.dumps(reg_payload, indent=2, default=str), encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.hist(residuals, bins=40, color="#2a6f97", edgecolor="white")
    axis.axvline(0, color="#333333", linewidth=1)
    axis.set(
        title="Regression residuals (synthetic test period)",
        xlabel="Actual − predicted connections",
        ylabel="Rows",
    )
    fig.tight_layout()
    fig.savefig(REGRESSION_RESIDUALS_FIGURE_PATH, dpi=140)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(y_true, predictions, alpha=0.3, s=12, color="#2a6f97")
    max_value = max(float(y_true.max()), float(predictions.max()))
    axis.plot([0, max_value], [0, max_value], linestyle="--", color="#555555")
    axis.set(
        title="Predicted vs actual connections (synthetic test period)",
        xlabel="Actual",
        ylabel="Predicted",
    )
    fig.tight_layout()
    fig.savefig(REGRESSION_PREDICTIONS_FIGURE_PATH, dpi=140)
    plt.close(fig)

    combined = json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    combined["classification"]["test_metrics"] = class_metrics
    combined["regression"]["test_metrics"] = reg_payload
    MODEL_METADATA_PATH.write_text(json.dumps(combined, indent=2, default=str), encoding="utf-8")
    logger.info("Classification test metrics: %s", class_metrics)
    logger.info("Regression test metrics: %s", reg_payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
