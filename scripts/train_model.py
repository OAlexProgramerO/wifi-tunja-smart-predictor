#!/usr/bin/env python3
"""Train and persist temporally validated classification and regression pipelines."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np

from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    DATASET_VERSION,
    INTERVAL_CONFIDENCE,
    MODEL_COMPARISON_PATH,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PREPROCESSING_VERSION,
    PROCESSED_DATASET_PATH,
    PROJECT_NAME,
    PROJECT_ROOT,
    PROJECT_VERSION,
    REGRESSION_COMPARISON_PATH,
    REGRESSION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_TARGET_COLUMN,
    SYNTHETIC_DATA_DISCLAIMER,
    TARGET_COLUMN,
    TRAIN_END,
    VALIDATION_END,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.regression import (
    conformal_residual_radius,
    evaluate_regressor,
    train_regression_models,
)
from wifi_tunja_smart_predictor.models.train import temporal_split, train_baseline_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _period(frame) -> dict[str, object]:
    return {
        "start": str(frame["timestamp"].min()),
        "end": str(frame["timestamp"].max()),
        "rows": int(len(frame)),
    }


def main() -> int:
    """Train classifier and regressor using the same chronological partitions."""
    source = PROCESSED_DATASET_PATH
    if not source.is_file():
        from wifi_tunja_smart_predictor.config import RAW_DATASET_PATH

        source = RAW_DATASET_PATH
        logger.info("Processed dataset missing; using immutable raw source %s", source)
        frame = load_raw_dataset(source).drop_duplicates()
    else:
        frame = load_raw_dataset(source)

    train, validation, test = temporal_split(frame)
    logger.info(
        "Temporal partitions train=%s validation=%s test=%s",
        _period(train),
        _period(validation),
        _period(test),
    )
    comparison, fitted, selected_name = train_baseline_models(train, validation, test)
    selected = fitted[selected_name]
    comparison.to_csv(MODEL_COMPARISON_PATH, index=False)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected, MODEL_PATH)
    classifier_metrics = evaluate_classifier(
        selected, test[MODEL_INPUT_COLUMNS], test[TARGET_COLUMN]
    )
    classifier_validation = evaluate_classifier(
        selected, validation[MODEL_INPUT_COLUMNS], validation[TARGET_COLUMN]
    )
    classifier_frame = comparison.set_index("model")
    trained_at = datetime.now(timezone.utc).isoformat()
    train_period = {**_period(train), "cutoff": TRAIN_END}
    validation_period = {**_period(validation), "cutoff": VALIDATION_END}
    test_period = _period(test)

    classifier_metadata = {
        "project": PROJECT_NAME,
        "project_version": PROJECT_VERSION,
        "model_type": type(selected.named_steps["model"]).__name__,
        "selected_model": selected_name,
        "selection_rule": "Highest validation F1 for class HIGH; test set was not used for selection.",
        "training_timestamp_utc": trained_at,
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "target_column": TARGET_COLUMN,
        "feature_count": len(MODEL_INPUT_COLUMNS),
        "feature_names": MODEL_INPUT_COLUMNS,
        "model_parameters": selected.named_steps["model"].get_params(),
        "training_time_seconds": float(
            classifier_frame.loc[selected_name, "training_time_seconds"]
        ),
        "train_period": train_period,
        "validation_period": validation_period,
        "test_period": test_period,
        "validation_metrics": {
            key: value for key, value in classifier_validation.items() if key != "confusion_matrix"
        },
        "test_metrics": {
            key: value for key, value in classifier_metrics.items() if key != "confusion_matrix"
        },
        "confusion_matrix": classifier_metrics["confusion_matrix"],
        "confusion_matrix_labels": classifier_metrics["labels"],
        "model_path": MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
    }
    CLASSIFICATION_METADATA_PATH.write_text(
        json.dumps(classifier_metadata, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "Selected classifier %s; test metrics %s",
        selected_name,
        classifier_metadata["test_metrics"],
    )

    regression_comparison, regression_models, regression_name = train_regression_models(
        train, validation, test
    )
    regressor = regression_models[regression_name]
    regression_comparison.to_csv(REGRESSION_COMPARISON_PATH, index=False)
    joblib.dump(regressor, REGRESSION_MODEL_PATH)
    x_validation = validation[MODEL_INPUT_COLUMNS]
    y_validation = validation[REGRESSION_TARGET_COLUMN].astype(float)
    validation_predictions = np.asarray(regressor.predict(x_validation), dtype=float)
    radius = conformal_residual_radius(
        y_validation, validation_predictions, confidence=INTERVAL_CONFIDENCE
    )
    x_test = test[MODEL_INPUT_COLUMNS]
    y_test = test[REGRESSION_TARGET_COLUMN].astype(float)
    test_predictions = np.maximum(0.0, np.asarray(regressor.predict(x_test), dtype=float))
    regression_metrics = evaluate_regressor(regressor, x_test, y_test)
    test_residuals = np.abs(y_test.to_numpy() - test_predictions)
    lower = np.maximum(0.0, test_predictions - radius)
    upper = test_predictions + radius
    regression_frame = regression_comparison.set_index("model")
    regression_metadata = {
        "project": PROJECT_NAME,
        "project_version": PROJECT_VERSION,
        "model_type": type(regressor.named_steps["model"]).__name__,
        "selected_model": regression_name,
        "selection_rule": "Lowest validation MAE; test set was not used for selection.",
        "training_timestamp_utc": trained_at,
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "target_column": REGRESSION_TARGET_COLUMN,
        "feature_count": len(MODEL_INPUT_COLUMNS),
        "feature_names": MODEL_INPUT_COLUMNS,
        "model_parameters": regressor.named_steps["model"].get_params(),
        "training_time_seconds": float(
            regression_frame.loc[regression_name, "training_time_seconds"]
        ),
        "train_period": train_period,
        "validation_period": validation_period,
        "test_period": test_period,
        "validation_metrics": evaluate_regressor(regressor, x_validation, y_validation),
        "test_metrics": regression_metrics,
        "interval_method": "split conformal absolute-residual quantile calibrated on validation predictions",
        "interval_confidence": INTERVAL_CONFIDENCE,
        "calibration_rows": int(len(validation)),
        "calibration_radius": radius,
        "test_interval_coverage": float(
            np.mean((y_test.to_numpy() >= lower) & (y_test.to_numpy() <= upper))
        ),
        "test_mean_interval_width": float(np.mean(upper - lower)),
        "test_absolute_residual_quantiles": {
            str(q): float(np.quantile(test_residuals, q)) for q in (0.5, 0.9, 0.95)
        },
        "model_path": REGRESSION_MODEL_PATH.relative_to(PROJECT_ROOT).as_posix(),
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
    }
    REGRESSION_METADATA_PATH.write_text(
        json.dumps(regression_metadata, indent=2, default=str), encoding="utf-8"
    )
    logger.info(
        "Selected regressor %s; test metrics %s; conformal radius %.3f (test coverage %.3f)",
        regression_name,
        regression_metrics,
        radius,
        regression_metadata["test_interval_coverage"],
    )

    # Preserve the former top-level classifier fields so older /model-info clients keep working.
    combined_metadata = {
        **classifier_metadata,
        "synthetic_data_disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "classification": classifier_metadata,
        "regression": regression_metadata,
        "artifacts": {
            "classifier": classifier_metadata["model_path"],
            "regressor": regression_metadata["model_path"],
            "classification_metadata": CLASSIFICATION_METADATA_PATH.relative_to(
                PROJECT_ROOT
            ).as_posix(),
            "regression_metadata": REGRESSION_METADATA_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
    }
    MODEL_METADATA_PATH.write_text(
        json.dumps(combined_metadata, indent=2, default=str), encoding="utf-8"
    )
    print("Synthetic evaluation only; no real-world performance is implied.")
    print("Classification:", classifier_metadata["test_metrics"])
    print("Regression:", regression_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
