"""Read-only comparisons of persisted synthetic model candidate evaluations."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    MODEL_COMPARISON_PATH,
    MODEL_DIAGNOSTICS_PATH,
    MODEL_PATH,
    PROJECT_VERSION,
    REGRESSION_COMPARISON_PATH,
    REGRESSION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
)

_METRICS = {
    "classifier": (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "validation_f1",
        "training_time_seconds",
    ),
    "regressor": (
        "validation_mae",
        "validation_rmse",
        "validation_r2",
        "mae",
        "rmse",
        "r2",
        "training_time_seconds",
    ),
}
_HIGHER_IS_BETTER = {"accuracy", "precision", "recall", "f1", "roc_auc", "validation_f1", "r2", "validation_r2"}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_candidates(path: Path, role: str) -> list[dict[str, Any]] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "model" not in reader.fieldnames:
                return None
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error):
        return None
    result = []
    for row in rows:
        name = row.get("model", "").strip()
        if not name:
            continue
        metrics: dict[str, float] = {}
        for key in _METRICS[role]:
            raw = row.get(key)
            if raw in (None, ""):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                metrics[key] = value
        result.append({"model": name[:80], "metrics": metrics})
    return result


def _period_signature(metadata: dict[str, Any]) -> tuple[Any, ...] | None:
    periods = []
    for name in ("train_period", "validation_period", "test_period"):
        period = metadata.get(name)
        if not isinstance(period, dict) or not isinstance(period.get("start"), str) or not isinstance(
            period.get("end"), str
        ):
            return None
        periods.append((period["start"], period["end"], period.get("rows")))
    return tuple(periods)


def _artifact_fingerprint(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _task_comparison(
    role: str,
    metadata: dict[str, Any] | None,
    rows: list[dict[str, Any]] | None,
    dataset_fingerprint: str | None,
    artifact_path: Path,
) -> dict[str, Any]:
    if rows is None:
        return {
            "role": role,
            "status": "UNAVAILABLE",
            "message": "The saved candidate evaluation table could not be read.",
            "runs": [],
        }

    metadata = metadata or {}
    selected = metadata.get("selected_model")
    period_signature = _period_signature(metadata)
    training_timestamp = metadata.get("training_timestamp_utc")
    dataset_version = metadata.get("dataset_version")
    reasons = []
    if not metadata:
        reasons.append("Training metadata is missing or malformed.")
    if not dataset_fingerprint:
        reasons.append("A validated dataset fingerprint is unavailable.")
    if period_signature is None:
        reasons.append("One or more evaluation windows are unavailable.")
    if not isinstance(training_timestamp, str):
        reasons.append("The evaluation timestamp is unavailable.")
    if not rows:
        reasons.append("No valid candidate metric rows were found.")
    status = "PARTIAL" if reasons else "VALID"
    expected_metric_keys = set().union(*(set(row["metrics"]) for row in rows)) if rows else set()
    if rows and any(set(row["metrics"]) != expected_metric_keys for row in rows):
        reasons.append("Some candidate records contain only a subset of available metrics.")
        status = "PARTIAL"
    selected_fingerprint = _artifact_fingerprint(artifact_path)
    records = []
    baseline_metrics = None
    for row in rows:
        is_selected = row["model"] == selected
        run = {
            "model": row["model"],
            "model_version": metadata.get("project_version"),
            "training_run_id": metadata.get("training_run_id"),
            "evaluation_timestamp": training_timestamp,
            "dataset_version": dataset_version,
            "dataset_fingerprint": dataset_fingerprint,
            "artifact_fingerprint": selected_fingerprint if is_selected else None,
            "artifact_persisted": is_selected and selected_fingerprint is not None,
            "selected": is_selected,
            "evaluation_windows": metadata.get("train_period") and {
                "train": metadata.get("train_period"),
                "validation": metadata.get("validation_period"),
                "test": metadata.get("test_period"),
            },
            "metrics": row["metrics"],
            "metric_deltas_vs_selected": None,
        }
        if is_selected:
            baseline_metrics = row["metrics"]
        records.append(run)

    if selected and not any(run["selected"] for run in records):
        status = "PARTIAL"
        reasons.append("Selected model is not present in the saved comparison table.")
    if status in {"VALID", "PARTIAL"} and baseline_metrics is not None and not any(
        "fingerprint" in reason.lower() or "window" in reason.lower() for reason in reasons
    ):
        for run in records:
            if run["selected"]:
                run["metric_deltas_vs_selected"] = {key: 0.0 for key in run["metrics"]}
            else:
                run["metric_deltas_vs_selected"] = {
                    key: round(value - baseline_metrics[key], 10)
                    for key, value in run["metrics"].items()
                    if key in baseline_metrics
                }
    return {
        "role": role,
        "status": status,
        "message": (
            "Candidate records share the saved task, dataset identity, and temporal split."
            if status == "VALID"
            else "Comparison is limited because required run evidence is incomplete."
        ),
        "reasons": reasons,
        "selected_model": selected,
        "dataset_fingerprint": dataset_fingerprint,
        "evaluation_windows": records[0]["evaluation_windows"] if records else None,
        "higher_is_better": sorted(_HIGHER_IS_BETTER & set(_METRICS[role])),
        "runs": records,
    }


def compare_run_records(
    records: list[dict[str, Any]], *, selected_model: str
) -> dict[str, Any]:
    """Compare synthetic candidate records, suppressing deltas for incompatible evidence.

    This public pure helper also supports focused integrity tests without touching files.
    Each input record must contain `model`, `metrics`, `dataset_fingerprint`, and
    `evaluation_windows`; an optional `role` permits task isolation.
    """
    normalized = [dict(record) for record in records if isinstance(record, dict)]
    fingerprints = {item.get("dataset_fingerprint") for item in normalized}
    windows = {json.dumps(item.get("evaluation_windows"), sort_keys=True) for item in normalized}
    incompatible = len(fingerprints) > 1 or len(windows) > 1
    baseline = next((item for item in normalized if item.get("model") == selected_model), None)
    metric_keys = [
        set(item.get("metrics", {})) if isinstance(item.get("metrics"), dict) else set()
        for item in normalized
    ]
    partial_metrics = bool(metric_keys) and any(keys != metric_keys[0] for keys in metric_keys[1:])
    incomplete = (
        baseline is None
        or any(not item.get("dataset_fingerprint") or not item.get("evaluation_windows") for item in normalized)
        or partial_metrics
    )
    status = "INCOMPATIBLE" if incompatible else "PARTIAL" if incomplete else "VALID"
    for item in normalized:
        item["metric_deltas_vs_selected"] = None
    if status in {"VALID", "PARTIAL"} and baseline is not None and not incompatible and not any(
        not item.get("dataset_fingerprint") or not item.get("evaluation_windows")
        for item in normalized
    ):
        base_metrics = baseline.get("metrics") if isinstance(baseline.get("metrics"), dict) else {}
        for item in normalized:
            metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
            item["metric_deltas_vs_selected"] = {
                key: round(value - base_metrics[key], 10)
                for key, value in metrics.items()
                if isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isinstance(base_metrics.get(key), (int, float))
                and not isinstance(base_metrics.get(key), bool)
                and math.isfinite(value)
                and math.isfinite(base_metrics[key])
            }
    return {"status": status, "runs": normalized}


def build_model_run_comparison() -> dict[str, Any]:
    """Read persisted candidate tables and compare within classifier/regressor task only."""
    classifier = _read_json(CLASSIFICATION_METADATA_PATH)
    regressor = _read_json(REGRESSION_METADATA_PATH)
    report = _read_json(MODEL_DIAGNOSTICS_PATH)
    dataset = report.get("dataset", {}) if report else {}
    dataset_fingerprint = dataset.get("sha256") if isinstance(dataset, dict) else None
    if not isinstance(dataset_fingerprint, str) or len(dataset_fingerprint) != 64:
        dataset_fingerprint = None
    groups = [
        _task_comparison(
            "classifier",
            classifier,
            _read_candidates(MODEL_COMPARISON_PATH, "classifier"),
            dataset_fingerprint,
            MODEL_PATH,
        ),
        _task_comparison(
            "regressor",
            regressor,
            _read_candidates(REGRESSION_COMPARISON_PATH, "regressor"),
            dataset_fingerprint,
            REGRESSION_MODEL_PATH,
        ),
    ]
    states = {item["status"] for item in groups}
    overall = "INCOMPATIBLE" if "INCOMPATIBLE" in states else "VALID" if states == {"VALID"} else "PARTIAL"
    return {
        "status": overall,
        "application_version": PROJECT_VERSION,
        "synthetic_data_only": True,
        "comparison_policy": "Compare candidates only within each task and shared temporal evaluation table.",
        "tasks": groups,
        "limitations": [
            "Only the most recent persisted candidate tables are retained; this is not a historical run archive.",
            "Artifact fingerprints are available only for the selected, persisted model; candidate estimators are not saved individually.",
            "All metrics are computed from synthetic data and do not validate real public WiFi demand.",
        ],
    }
