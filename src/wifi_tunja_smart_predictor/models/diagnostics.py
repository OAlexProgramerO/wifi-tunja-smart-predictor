"""Build and validate reproducible synthetic model quality diagnostics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    MODEL_PATH,
    MODEL_DIAGNOSTICS_PATH,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    PROJECT_VERSION,
    REGRESSION_MODEL_PATH,
    REGRESSION_TARGET_COLUMN,
    SYNTHETIC_DATA_DISCLAIMER,
    TARGET_COLUMN,
    TARGET_LABELS,
)

SCHEMA_VERSION = "1.0"


def sha256_file(path: Path) -> str:
    """Return the content fingerprint for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _class_metrics(matrix: np.ndarray) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for index, label in enumerate(TARGET_LABELS):
        support = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        true_positive = int(matrix[index, index])
        precision = true_positive / predicted if predicted else None
        recall = true_positive / support if support else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else (0.0 if precision == 0 or recall == 0 else None)
        )
        result[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return result


def _metrics(
    classifier: Any,
    regressor: Any,
    frame: pd.DataFrame,
    regression_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("Model diagnostics require at least one evaluation row.")
    features = frame[MODEL_INPUT_COLUMNS]
    labels = frame[TARGET_COLUMN].astype(str).to_numpy()
    predicted_labels = np.asarray(classifier.predict(features)).astype(str)
    matrix = confusion_matrix(labels, predicted_labels, labels=list(TARGET_LABELS))
    probabilities = None
    roc_auc = None
    if callable(getattr(classifier, "predict_proba", None)):
        probabilities = np.asarray(classifier.predict_proba(features), dtype=float)
        classes = list(classifier.classes_)
        if TARGET_LABELS[1] in classes and len(set(labels)) == 2:
            roc_auc = float(
                roc_auc_score(
                    labels == TARGET_LABELS[1], probabilities[:, classes.index(TARGET_LABELS[1])]
                )
            )
    class_total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / class_total) if class_total else None
    actual = frame[REGRESSION_TARGET_COLUMN].to_numpy(dtype=float)
    prediction = np.maximum(0.0, np.asarray(regressor.predict(features), dtype=float))
    residual = prediction - actual
    classes = _class_metrics(matrix)
    regression_metrics = {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)) if len(actual) > 1 else None,
        "mean_prediction_bias": float(residual.mean()),
        "absolute_error_quantiles": {
            str(q): float(np.quantile(np.abs(residual), q)) for q in (0.5, 0.9, 0.95)
        },
    }
    if regression_metadata and isinstance(
        regression_metadata.get("calibration_radius"), (int, float)
    ):
        radius = float(regression_metadata["calibration_radius"])
        lower = np.maximum(0.0, prediction - radius)
        upper = prediction + radius
        regression_metrics.update(
            interval_confidence=regression_metadata.get("interval_confidence"),
            interval_radius=radius,
            interval_coverage=float(np.mean((actual >= lower) & (actual <= upper))),
        )
    return {
        "rows": len(frame),
        "classification": {
            "accuracy": accuracy,
            "precision": classes[TARGET_LABELS[1]]["precision"],
            "recall": classes[TARGET_LABELS[1]]["recall"],
            "f1": classes[TARGET_LABELS[1]]["f1"],
            "roc_auc": roc_auc,
            "roc_auc_status": (
                "available" if roc_auc is not None else "undefined: both classes are required"
            ),
            "class_metrics": classes,
            "confusion_labels": list(TARGET_LABELS),
            "confusion_matrix": matrix.tolist(),
            "class_counts": {label: int((labels == label).sum()) for label in TARGET_LABELS},
        },
        "regression": regression_metrics,
    }


def _rolling_windows(frame: pd.DataFrame, classifier: Any, regressor: Any) -> list[dict[str, Any]]:
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = pd.Index(ordered["timestamp"].drop_duplicates().sort_values())
    windows = []
    # Three expanding-window folds. Each fold fits a fresh clone on strictly earlier
    # rows and evaluates only on the immediately following chronological block.
    for index, (train_fraction, test_fraction) in enumerate(
        ((0.5, 0.6), (0.6, 0.7), (0.7, 0.8)), 1
    ):
        train_end = min(len(timestamps) - 1, int(len(timestamps) * train_fraction) - 1)
        test_end = min(len(timestamps) - 1, int(len(timestamps) * test_fraction) - 1)
        train_cutoff, test_cutoff = timestamps[train_end], timestamps[test_end]
        train = ordered.loc[ordered.timestamp <= train_cutoff]
        test = ordered.loc[(ordered.timestamp > train_cutoff) & (ordered.timestamp <= test_cutoff)]
        entry: dict[str, Any] = {
            "window": index,
            "train_start": str(train.timestamp.min()) if len(train) else None,
            "train_end": str(train.timestamp.max()) if len(train) else None,
            "test_start": str(test.timestamp.min()) if len(test) else None,
            "test_end": str(test.timestamp.max()) if len(test) else None,
            "train_rows": len(train),
            "test_rows": len(test),
        }
        if len(train) < 100 or len(test) < 20 or train.timestamp.max() >= test.timestamp.min():
            entry.update(status="skipped", reason="insufficient or non-chronological fold rows")
        elif train[TARGET_COLUMN].nunique() < 2:
            entry.update(status="skipped", reason="training fold contains fewer than two classes")
        else:
            try:
                fold_classifier, fold_regressor = clone(classifier), clone(regressor)
                fold_classifier.fit(train[MODEL_INPUT_COLUMNS], train[TARGET_COLUMN])
                fold_regressor.fit(train[MODEL_INPUT_COLUMNS], train[REGRESSION_TARGET_COLUMN])
                entry.update(status="passed", **_metrics(fold_classifier, fold_regressor, test))
            except (ValueError, TypeError) as exc:
                entry.update(
                    status="skipped", reason=f"fold could not be fitted: {type(exc).__name__}"
                )
        windows.append(entry)
    return windows


def build_diagnostics(
    frame: pd.DataFrame,
    classifier: Any,
    regressor: Any,
    classifier_metadata: dict[str, Any],
    regressor_metadata: dict[str, Any],
    dataset_path: Path,
) -> dict[str, Any]:
    """Build the shared report from exact validated model artifacts and data."""
    ordered = frame.sort_values("timestamp", kind="stable").reset_index(drop=True)
    test_start = pd.Timestamp(classifier_metadata["test_period"]["start"])
    test_end = pd.Timestamp(classifier_metadata["test_period"]["end"])
    test = ordered.loc[ordered.timestamp.between(test_start, test_end)]
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_version": PROJECT_VERSION,
        "model_version": classifier_metadata.get("project_version"),
        "dataset": {
            "sha256": sha256_file(dataset_path),
            "rows": len(frame),
            "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        },
        "data_quality": {
            "exact_duplicate_rows": int(frame.duplicated().sum()),
            "missing_value_count": int(frame.isna().sum().sum()),
            "columns_with_missing_values": {
                str(column): int(count) for column, count in frame.isna().sum().items() if count
            },
        },
        "artifacts": {
            "classifier_sha256": sha256_file(MODEL_PATH),
            "regressor_sha256": sha256_file(REGRESSION_MODEL_PATH),
            "classifier_model": classifier_metadata.get("selected_model"),
            "regressor_model": regressor_metadata.get("selected_model"),
        },
        "method": {
            "holdout": "Existing chronological train/validation/test split; final test rows are held out.",
            "rolling_evaluation": "Three expanding windows: fresh cloned pipelines are fit on earlier rows only and tested on the immediately following block.",
            "prediction_probability": "Estimator probabilities are not presented as calibrated confidence.",
        },
        "test_period": classifier_metadata.get("test_period"),
        "test": _metrics(classifier, regressor, test, regressor_metadata),
        "rolling_windows": _rolling_windows(ordered, classifier, regressor),
        "warnings": [
            "All rows, labels, coordinates, and measurements are synthetic; this report does not validate real WiFi demand in Tunja.",
            "Window scores may vary with distribution shift and do not establish operational forecast quality.",
        ],
    }


def write_diagnostics(report: dict[str, Any], path: Path = MODEL_DIAGNOSTICS_PATH) -> None:
    """Persist a complete JSON report atomically in its destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_valid_diagnostics(path: Path = MODEL_DIAGNOSTICS_PATH) -> dict[str, Any]:
    """Load report only if its schema, dataset, package and model artifacts are current."""
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report, dict):
            raise ValueError("diagnostics report must be a JSON object")
        if report.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported diagnostics schema")
        if report.get("project_version") != PROJECT_VERSION:
            raise ValueError("diagnostics report belongs to another project version")
        dataset = report["dataset"]
        artifacts = report["artifacts"]
        if (
            not isinstance(dataset, dict)
            or not isinstance(artifacts, dict)
            or not isinstance(report.get("test"), dict)
            or not isinstance(report["test"].get("classification"), dict)
            or not isinstance(report["test"].get("regression"), dict)
            or not isinstance(report.get("rolling_windows"), list)
            or not isinstance(report.get("warnings"), list)
            or not isinstance(report.get("method"), dict)
            or not isinstance(report.get("data_quality"), dict)
            or not isinstance(report.get("test_period"), dict)
        ):
            raise ValueError("diagnostics report structure is invalid")
        current_dataset = next(
            (
                candidate
                for candidate in (PROCESSED_DATASET_PATH, RAW_DATASET_PATH)
                if candidate.is_file()
            ),
            None,
        )
        if current_dataset is None or sha256_file(current_dataset) != dataset.get("sha256"):
            raise ValueError("diagnostics dataset fingerprint is stale")
        if sha256_file(MODEL_PATH) != artifacts.get("classifier_sha256"):
            raise ValueError("diagnostics classifier fingerprint is stale")
        if sha256_file(REGRESSION_MODEL_PATH) != artifacts.get("regressor_sha256"):
            raise ValueError("diagnostics regressor fingerprint is stale")
        return report
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Model diagnostics are missing, invalid, or stale.") from exc
