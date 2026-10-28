"""Bounded chronological evaluation and synthetic feature-shift diagnostics."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone

from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    PROJECT_VERSION,
    REGRESSION_TARGET_COLUMN,
    TARGET_COLUMN,
)
from wifi_tunja_smart_predictor.models.diagnostics import _metrics

SCHEMA_VERSION = "1.0"
MAX_WINDOWS = 5
MAX_FEATURES = 12


def _safe(value: Any) -> Any:
    """Convert numpy values and non-finite floats to JSON-safe deterministic values."""
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _distribution_change(reference: pd.Series, comparison: pd.Series) -> dict[str, Any]:
    ref_missing = float(reference.isna().mean())
    cmp_missing = float(comparison.isna().mean())
    ref = reference.dropna()
    cmp = comparison.dropna()
    result: dict[str, Any] = {
        "reference_missing_proportion": ref_missing,
        "comparison_missing_proportion": cmp_missing,
        "missing_proportion_difference": abs(cmp_missing - ref_missing),
        "reference_non_missing_rows": len(ref),
        "comparison_non_missing_rows": len(cmp),
    }
    if len(ref) < 2 or len(cmp) < 2:
        result.update(statistic=None, method=None, status="insufficient_samples")
    elif pd.api.types.is_numeric_dtype(reference.dtype) and pd.api.types.is_numeric_dtype(
        comparison.dtype
    ):
        pooled = float(np.sqrt((ref.var(ddof=1) + cmp.var(ddof=1)) / 2))
        difference = abs(float(ref.mean()) - float(cmp.mean()))
        result.update(
            statistic=(
                difference / (difference + pooled) if pooled else (0.0 if difference == 0 else 1.0)
            ),
            method="bounded_normalized_mean_difference",
            status="available",
        )
    else:
        ref_dist = ref.astype(str).value_counts(normalize=True)
        cmp_dist = cmp.astype(str).value_counts(normalize=True)
        categories = ref_dist.index.union(cmp_dist.index)
        statistic = float(
            0.5 * sum(abs(ref_dist.get(key, 0.0) - cmp_dist.get(key, 0.0)) for key in categories)
        )
        result.update(
            statistic=statistic,
            method="categorical_total_variation_distance",
            status="available",
        )
    return result


def evaluate_temporal_windows(
    frame: pd.DataFrame,
    classifier: Any,
    regressor: Any,
    *,
    window_count: int = 3,
    min_train_rows: int = 100,
    min_test_rows: int = 20,
    feature_names: list[str] | None = None,
    dataset_fingerprint: str | None = None,
    classifier_fingerprint: str | None = None,
    regressor_fingerprint: str | None = None,
    feature_version: str | None = None,
) -> dict[str, Any]:
    """Fit cloned estimators on earlier timestamps and evaluate later blocks only."""
    if not 2 <= window_count <= MAX_WINDOWS:
        raise ValueError(f"window_count must be between 2 and {MAX_WINDOWS}.")
    if min_train_rows < 2 or min_test_rows < 2:
        raise ValueError("Minimum train and test rows must each be at least 2.")
    if not isinstance(frame, pd.DataFrame) or frame.empty or "timestamp" not in frame:
        raise ValueError("A non-empty frame with timestamps is required.")
    try:
        ordered = frame.copy()
        ordered["timestamp"] = pd.to_datetime(ordered["timestamp"], format="mixed", errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Timestamps must be valid and non-missing.") from exc
    if ordered.timestamp.isna().any():
        raise ValueError("Timestamps must be valid and non-missing.")
    required = set(MODEL_INPUT_COLUMNS + [TARGET_COLUMN, REGRESSION_TARGET_COLUMN])
    if not required.issubset(ordered.columns):
        raise ValueError("Dataset is missing required evaluation columns.")
    ordered = ordered.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = pd.Index(ordered.timestamp.drop_duplicates())
    first_test_index = max(1, len(timestamps) // 2)
    if len(timestamps) - first_test_index < window_count:
        raise ValueError("Insufficient distinct timestamps for requested windows.")
    features = feature_names or list(MODEL_INPUT_COLUMNS[:MAX_FEATURES])
    if not 1 <= len(features) <= MAX_FEATURES or len(set(features)) != len(features):
        raise ValueError(f"Select between 1 and {MAX_FEATURES} unique features.")
    if not set(features).issubset(MODEL_INPUT_COLUMNS):
        raise ValueError("Feature selection contains unsupported columns.")

    # Reserve the first half for initial training; partition the remainder into
    # non-overlapping test blocks. Training expands only with rows before each block.
    remaining = len(timestamps) - first_test_index
    edges = [first_test_index + (remaining * i // window_count) for i in range(window_count + 1)]
    windows = []
    feature_diagnostics = []
    for index in range(window_count):
        start, end = edges[index], edges[index + 1]
        test_times = timestamps[start:end]
        if len(test_times) == 0:
            continue
        cutoff = test_times[0]
        train = ordered.loc[ordered.timestamp < cutoff]
        test = ordered.loc[ordered.timestamp.isin(test_times)]
        entry: dict[str, Any] = {
            "window": index + 1,
            "train_start": train.timestamp.min().isoformat() if len(train) else None,
            "train_end": train.timestamp.max().isoformat() if len(train) else None,
            "test_start": test.timestamp.min().isoformat(),
            "test_end": test.timestamp.max().isoformat(),
            "train_rows": len(train),
            "test_rows": len(test),
        }
        if len(train) < min_train_rows or len(test) < min_test_rows:
            entry.update(status="skipped", reason="insufficient_train_or_test_rows")
        elif train[TARGET_COLUMN].nunique() < 2:
            entry.update(status="skipped", reason="training_window_has_one_target_class")
        else:
            try:
                fitted_classifier, fitted_regressor = clone(classifier), clone(regressor)
                fitted_classifier.fit(train[MODEL_INPUT_COLUMNS], train[TARGET_COLUMN])
                fitted_regressor.fit(train[MODEL_INPUT_COLUMNS], train[REGRESSION_TARGET_COLUMN])
                entry.update(
                    status="evaluated", metrics=_metrics(fitted_classifier, fitted_regressor, test)
                )
                reference = train
                for feature in features:
                    diagnostic = _distribution_change(reference[feature], test[feature])
                    feature_diagnostics.append(
                        {
                            "window": index + 1,
                            "reference_start": entry["train_start"],
                            "reference_end": entry["train_end"],
                            "comparison_start": entry["test_start"],
                            "comparison_end": entry["test_end"],
                            "reference_rows": len(reference),
                            "comparison_rows": len(test),
                            "feature": feature,
                            **diagnostic,
                        }
                    )
            except (ValueError, TypeError) as exc:
                entry.update(status="skipped", reason=f"evaluation_failed:{type(exc).__name__}")
        windows.append(entry)
    available = [item for item in feature_diagnostics if item["statistic"] is not None]
    top_features = sorted(
        available, key=lambda item: (-item["statistic"], item["window"], item["feature"])
    )[:5]
    evaluated = [item for item in windows if item["status"] == "evaluated"]
    metric_changes = {}
    for group, metric_names in (
        ("classification", ("accuracy", "precision", "recall", "f1", "roc_auc")),
        ("regression", ("mae", "rmse", "r2")),
    ):
        for metric_name in metric_names:
            values = [
                item["metrics"][group].get(metric_name)
                for item in evaluated
                if item["metrics"][group].get(metric_name) is not None
            ]
            metric_changes[f"{group}.{metric_name}"] = {
                "observed": len(values) > 1 and any(value != values[0] for value in values[1:]),
                "values": values,
            }
    data_limitations = [
        f"Window {item['window']} was skipped: {item['reason']}"
        for item in windows
        if item["status"] == "skipped"
    ]
    if any(item["status"] == "insufficient_samples" for item in feature_diagnostics):
        data_limitations.append(
            "One or more feature comparisons had insufficient non-missing samples."
        )
    fingerprint = (
        dataset_fingerprint
        or hashlib.sha256(
            pd.util.hash_pandas_object(ordered, index=True).values.tobytes()
        ).hexdigest()
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "project_version": PROJECT_VERSION,
        "feature_version": feature_version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "synthetic_chronological_expanding_windows",
        "parameters": {
            "window_count": window_count,
            "min_train_rows": min_train_rows,
            "min_test_rows": min_test_rows,
            "features": features,
        },
        "dataset": {"sha256": fingerprint, "rows": len(ordered), "synthetic": True},
        "artifacts": {
            "classifier_sha256": classifier_fingerprint,
            "regressor_sha256": regressor_fingerprint,
        },
        "windows": windows,
        "feature_diagnostics": feature_diagnostics,
        "summary": {
            "evaluated_windows": sum(item["status"] == "evaluated" for item in windows),
            "skipped_windows": sum(item["status"] == "skipped" for item in windows),
            "metric_changes": metric_changes,
            "metric_changes_observed": any(item["observed"] for item in metric_changes.values()),
            "top_distribution_changes": top_features,
            "data_quality_limitations": data_limitations,
            "limitations": [
                "All observations and outcomes are synthetic.",
                "Distribution differences do not establish model degradation, causality, or production drift.",
                "These results do not validate real WiFi demand in Tunja.",
            ],
        },
    }
    # Validate strict JSON encoding; represent undefined/non-finite values as null.
    return _safe(json.loads(json.dumps(payload, allow_nan=False, default=str)))


def render_temporal_markdown(report: dict[str, Any]) -> str:
    """Render the deterministic, readable portion of a temporal report."""
    lines = [
        "# Temporal evaluation report",
        "",
        f"Schema: {report['schema_version']}  ",
        f"Project version: {report['project_version']} · feature version: {report.get('feature_version') or 'unavailable'}  ",
        f"Generated at UTC: {report['generated_at_utc']}  ",
        f"Dataset SHA-256: {report['dataset']['sha256']}  ",
        f"Classifier SHA-256: {report['artifacts'].get('classifier_sha256') or 'unavailable'}  ",
        f"Regressor SHA-256: {report['artifacts'].get('regressor_sha256') or 'unavailable'}  ",
        f"Parameters: {json.dumps(report['parameters'], sort_keys=True, ensure_ascii=False)}",
        f"Evaluated windows: {report['summary']['evaluated_windows']} / {report['parameters']['window_count']}",
        f"Metric changes observed: {report['summary']['metric_changes_observed']}",
        "Changed metrics: "
        + ", ".join(
            key for key, result in report["summary"]["metric_changes"].items() if result["observed"]
        ),
        "",
        "All observations and outcomes are synthetic. Distribution differences do not prove model degradation or real-world drift.",
        "",
        "## Chronological windows",
        "",
        "| Window | Train period | Train rows | Test period | Test rows | Status | Accuracy | MAE |",
        "|---:|---|---:|---|---:|---|---:|---:|",
    ]
    for window in report["windows"]:
        metrics = window.get("metrics", {})
        classification = metrics.get("classification", {})
        regression = metrics.get("regression", {})
        lines.append(
            f"| {window['window']} | {window.get('train_start') or 'N/A'} – {window.get('train_end') or 'N/A'} | {window['train_rows']} | {window['test_start']} – {window['test_end']} | {window['test_rows']} | {window['status']} | {classification.get('accuracy', 'N/A')} | {regression.get('mae', 'N/A')} |"
        )
    lines.extend(
        [
            "",
            "## Feature distribution diagnostics",
            "",
            "| Window | Reference period | Reference rows | Comparison period | Comparison rows | Feature | Method | Change statistic | Missing proportion difference | Status |",
            "|---:|---|---:|---|---:|---|---|---:|---:|---|",
        ]
    )
    for diagnostic in report["feature_diagnostics"]:
        lines.append(
            f"| {diagnostic['window']} | {diagnostic['reference_start']} – {diagnostic['reference_end']} | {diagnostic['reference_rows']} | {diagnostic['comparison_start']} – {diagnostic['comparison_end']} | {diagnostic['comparison_rows']} | {diagnostic['feature']} | {diagnostic.get('method') or 'N/A'} | {diagnostic.get('statistic') if diagnostic.get('statistic') is not None else 'N/A'} | {diagnostic['missing_proportion_difference']} | {diagnostic['status']} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in report["summary"]["limitations"])
    lines.extend(f"- {limitation}" for limitation in report["summary"]["data_quality_limitations"])
    return "\n".join(lines) + "\n"
