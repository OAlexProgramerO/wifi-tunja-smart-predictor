"""Read-only inventory and compatibility checks for persisted synthetic ML artifacts."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    DATASET_VERSION,
    MODEL_DIAGNOSTICS_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATASET_PATH,
    PROJECT_VERSION,
    RAW_DATASET_PATH,
    REGRESSION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
)
from wifi_tunja_smart_predictor.exceptions import ModelMetadataError
from wifi_tunja_smart_predictor.models.artifacts import (
    load_classifier_metadata,
    load_regression_metadata,
    model_bundle_metadata_matches,
)
from wifi_tunja_smart_predictor.models.diagnostics import sha256_file

_SHA256_LENGTH = 64
_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "mae",
    "rmse",
    "r2",
    "test_rows",
    "test_start",
    "test_end",
    "interval_confidence",
    "interval_radius",
    "interval_test_coverage",
    "test_interval_coverage",
)
_PERIOD_KEYS = ("start", "end", "cutoff", "rows")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError from exc
    if not isinstance(value, dict):
        raise ValueError
    return value


def _safe_fingerprint(value: Any) -> str | None:
    if (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    ):
        return value
    return None


def _safe_metric_value(value: Any) -> float | int | str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if abs(value) <= 10**12 else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:80]
    return None


def _metric_summary(value: Any) -> dict[str, float | int | str | None] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: safe
        for key in _METRIC_KEYS
        if key in value and (safe := _safe_metric_value(value[key])) is not None
    }


def _period_summary(value: Any) -> dict[str, float | int | str | None] | None:
    if not isinstance(value, dict):
        return None
    return {
        key: safe
        for key in _PERIOD_KEYS
        if key in value and (safe := _safe_metric_value(value[key])) is not None
    }


def _metadata_summary(metadata: dict[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    fields = (
        "project_version",
        "selected_model",
        "model_type",
        "training_run_id",
        "training_timestamp_utc",
        "dataset_version",
        "preprocessing_version",
        "target_column",
        "feature_count",
        "training_time_seconds",
    )
    summary = {
        key: safe
        for key in fields
        if key in metadata and (safe := _safe_metric_value(metadata[key])) is not None
    }
    for key in ("train_period", "validation_period", "test_period"):
        period = _period_summary(metadata.get(key))
        if period:
            summary[key] = period
    for key in ("validation_metrics", "test_metrics"):
        metrics = _metric_summary(metadata.get(key))
        if metrics:
            summary[key] = metrics
    return summary


def _recorded_integrity_matches(metadata: dict[str, Any], sha256: str, size: int) -> bool | None:
    expected_sha = metadata.get("artifact_sha256")
    expected_size = metadata.get("artifact_size_bytes")
    if expected_sha is None and expected_size is None:
        return None
    if _safe_fingerprint(expected_sha) is None or type(expected_size) is not int:
        return False
    return expected_sha == sha256 and expected_size == size


def _dataset_path() -> Path | None:
    return next(
        (path for path in (PROCESSED_DATASET_PATH, RAW_DATASET_PATH) if path.is_file()), None
    )


@lru_cache(maxsize=16)
def _cached_sha256(path: str, modified_ns: int, changed_ns: int, size: int) -> str:
    """Cache a read-only content hash by the same file signature used by model loading."""
    del modified_ns, changed_ns, size
    return sha256_file(Path(path))


def _file_sha256(path: Path) -> str:
    stat = path.stat()
    return _cached_sha256(str(path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size)


def _artifact_entry(
    *,
    role: str,
    name: str,
    path: Path,
    metadata_paths: tuple[Path, ...],
    metadata: dict[str, Any] | None,
    evaluation_fingerprint: str | None,
    metadata_valid: bool,
) -> dict[str, Any]:
    artifact_exists = path.is_file()
    metadata_exists = all(metadata_path.is_file() for metadata_path in metadata_paths)
    entry: dict[str, Any] = {
        "role": role,
        "artifact_name": name,
        "availability": "available" if artifact_exists else "missing",
        "metadata_availability": "available" if metadata_exists else "missing",
        "status": "READY",
        "message": "Artifact and metadata checks passed.",
        "next_action": "No artifact repair or retraining action is indicated by these checks.",
        "training_metadata": _metadata_summary(metadata),
        "fingerprints": {
            "sha256": None,
            "recorded_sha256": None,
            "evaluation_sha256": evaluation_fingerprint,
            "metadata_match": None,
            "evaluation_match": None,
        },
    }
    if not artifact_exists:
        entry.update(
            status="MISSING",
            message="The required model artifact file is missing.",
            next_action="Restore the expected persisted artifact through the approved deployment workflow.",
        )
        return entry
    try:
        stat = path.stat()
        current_sha = _file_sha256(path)
    except OSError:
        entry.update(
            availability="unknown",
            status="INCOMPLETE",
            message="The artifact file could not be read for its integrity check.",
            next_action="Check artifact availability and permissions; no files were modified.",
        )
        return entry

    evaluation_match = (
        current_sha == evaluation_fingerprint if evaluation_fingerprint is not None else None
    )
    entry["fingerprints"].update(
        sha256=current_sha,
        recorded_sha256=(
            _safe_fingerprint(metadata.get("artifact_sha256")) if metadata is not None else None
        ),
        evaluation_match=evaluation_match,
    )
    entry["size_bytes"] = stat.st_size

    if not metadata_exists:
        entry.update(
            status="INCOMPLETE",
            message="One or more required training metadata files are missing.",
            next_action="Restore or regenerate the matching training metadata and evaluation report.",
        )
        return entry
    if not metadata_valid or metadata is None:
        entry.update(
            status="INCOMPLETE",
            message="Training metadata is malformed or internally inconsistent.",
            next_action="Review and regenerate the metadata bundle from its verified training run.",
        )
        return entry

    metadata_match = _recorded_integrity_matches(metadata, current_sha, stat.st_size)
    entry["fingerprints"].update(
        recorded_sha256=_safe_fingerprint(metadata.get("artifact_sha256")),
        metadata_match=metadata_match,
    )
    if metadata_match is False:
        entry.update(
            status="RETRAINING_REQUIRED",
            message="The current artifact does not match its recorded integrity fingerprint.",
            next_action="Review the model lineage and explicitly rebuild or retrain the artifact bundle.",
        )
    return entry


def _evaluation_summary(
    current_dataset_sha: str | None,
    classifier_sha: str | None,
    regressor_sha: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "MISSING",
        "generated_at_utc": None,
        "recorded_project_version": None,
        "dataset_fingerprint": None,
        "classifier_fingerprint": None,
        "regressor_fingerprint": None,
        "message": "No saved model evaluation report is available.",
        "next_action": "Generate evaluation metadata with the project's evaluation workflow.",
    }
    if not MODEL_DIAGNOSTICS_PATH.is_file():
        return result
    try:
        report = _read_json(MODEL_DIAGNOSTICS_PATH)
    except ValueError:
        result.update(
            status="INCOMPLETE",
            message="Saved evaluation metadata is malformed.",
            next_action="Regenerate the evaluation report from the current validated assets.",
        )
        return result

    dataset = report.get("dataset")
    artifacts = report.get("artifacts")
    if (
        report.get("schema_version") != "1.0"
        or not isinstance(dataset, dict)
        or not isinstance(artifacts, dict)
    ):
        result.update(
            status="INCOMPLETE",
            message="Saved evaluation metadata is incomplete.",
            next_action="Regenerate the evaluation report from the current validated assets.",
        )
        return result

    dataset_sha = _safe_fingerprint(dataset.get("sha256"))
    classifier_fingerprint = _safe_fingerprint(artifacts.get("classifier_sha256"))
    regressor_fingerprint = _safe_fingerprint(artifacts.get("regressor_sha256"))
    result.update(
        generated_at_utc=_safe_metric_value(report.get("generated_at_utc")),
        recorded_project_version=_safe_metric_value(report.get("project_version")),
        dataset_fingerprint=dataset_sha,
        classifier_fingerprint=classifier_fingerprint,
        regressor_fingerprint=regressor_fingerprint,
    )

    if dataset_sha is None or classifier_fingerprint is None or regressor_fingerprint is None:
        result.update(
            status="INCOMPLETE",
            message="Saved evaluation metadata does not contain valid SHA-256 fingerprints.",
            next_action="Regenerate the evaluation report with complete artifact and dataset fingerprints.",
        )
    elif current_dataset_sha is None:
        result.update(
            status="INCOMPLETE",
            message="The current synthetic dataset is unavailable for comparison.",
            next_action="Restore the expected synthetic dataset before assessing compatibility.",
        )
    elif dataset_sha != current_dataset_sha:
        result.update(
            status="RETRAINING_REQUIRED",
            message="The evaluated dataset fingerprint differs from the current synthetic dataset.",
            next_action="Review dataset lineage and explicitly retrain and evaluate against the intended dataset.",
        )
    elif classifier_fingerprint != classifier_sha or regressor_fingerprint != regressor_sha:
        result.update(
            status="INCOMPLETE",
            message="Evaluation fingerprints do not identify the current model artifacts.",
            next_action="Regenerate evaluation metadata for the current model artifacts.",
        )
    elif report.get("project_version") != PROJECT_VERSION:
        result.update(
            status="INCOMPLETE",
            message="Evaluation metadata was generated by a different application package version.",
            next_action="Regenerate evaluation metadata with the current package; this alone does not imply retraining.",
        )
    else:
        result.update(
            status="READY", message="Evaluation fingerprints match current synthetic assets."
        )
        result["next_action"] = "No evaluation refresh is indicated by these checks."
    return result


def build_artifact_inventory() -> dict[str, Any]:
    """Build a sanitized read-only inventory of classifier and regressor artifacts."""
    classifier_metadata: dict[str, Any] | None = None
    classifier_metadata_valid = False
    regressor_metadata: dict[str, Any] | None = None
    regressor_metadata_valid = False
    raw_combined: dict[str, Any] | None = None
    raw_regressor: dict[str, Any] | None = None
    try:
        raw_combined = _read_json(MODEL_METADATA_PATH)
        raw_regressor = _read_json(REGRESSION_METADATA_PATH)
        classifier_metadata = load_classifier_metadata(
            MODEL_METADATA_PATH, CLASSIFICATION_METADATA_PATH
        )
        regressor_metadata = load_regression_metadata(REGRESSION_METADATA_PATH)
        classifier_metadata_valid = True
        regressor_metadata_valid = True
    except (ModelMetadataError, OSError, ValueError):
        pass

    combined_regressor = (
        classifier_metadata.get("regression") if classifier_metadata is not None else None
    )
    metadata_consistent = (
        classifier_metadata_valid
        and regressor_metadata_valid
        and isinstance(combined_regressor, dict)
        and all(
            combined_regressor.get(field) == regressor_metadata.get(field)
            for field in (
                "project_version",
                "selected_model",
                "feature_count",
                "feature_names",
                "artifact_sha256",
                "artifact_size_bytes",
            )
        )
        and model_bundle_metadata_matches(classifier_metadata or {}, regressor_metadata or {})
        and model_bundle_metadata_matches(classifier_metadata or {}, combined_regressor or {})
    )
    classifier_sha_from_evaluation = None
    regressor_sha_from_evaluation = None
    try:
        evaluation_raw = _read_json(MODEL_DIAGNOSTICS_PATH)
        evaluation_artifacts = evaluation_raw.get("artifacts", {})
        if isinstance(evaluation_artifacts, dict):
            classifier_sha_from_evaluation = _safe_fingerprint(
                evaluation_artifacts.get("classifier_sha256")
            )
            regressor_sha_from_evaluation = _safe_fingerprint(
                evaluation_artifacts.get("regressor_sha256")
            )
    except ValueError:
        pass

    classifier = _artifact_entry(
        role="classifier",
        name="WiFi demand classifier",
        path=MODEL_PATH,
        metadata_paths=(MODEL_METADATA_PATH, CLASSIFICATION_METADATA_PATH),
        metadata=classifier_metadata or raw_combined,
        evaluation_fingerprint=classifier_sha_from_evaluation,
        metadata_valid=classifier_metadata_valid and metadata_consistent,
    )
    regressor = _artifact_entry(
        role="regressor",
        name="Next-hour connection regressor",
        path=REGRESSION_MODEL_PATH,
        metadata_paths=(MODEL_METADATA_PATH, REGRESSION_METADATA_PATH),
        metadata=regressor_metadata or raw_regressor,
        evaluation_fingerprint=regressor_sha_from_evaluation,
        metadata_valid=regressor_metadata_valid and metadata_consistent,
    )

    dataset_path = _dataset_path()
    try:
        dataset_sha = _file_sha256(dataset_path) if dataset_path is not None else None
    except OSError:
        dataset_sha = None
    evaluation = _evaluation_summary(
        dataset_sha,
        classifier["fingerprints"]["sha256"],
        regressor["fingerprints"]["sha256"],
    )

    if metadata_consistent and isinstance(classifier_metadata, dict):
        for artifact, metadata in (
            (classifier, classifier_metadata),
            (regressor, regressor_metadata or {}),
        ):
            recorded_dataset_version = metadata.get("dataset_version")
            artifact["dataset_compatibility"] = {
                "recorded_dataset_version": _safe_metric_value(recorded_dataset_version),
                "current_dataset_version": DATASET_VERSION,
                "matches": recorded_dataset_version == DATASET_VERSION,
            }
            if recorded_dataset_version != DATASET_VERSION and artifact["status"] == "READY":
                artifact.update(
                    status="RETRAINING_REQUIRED",
                    message="Recorded training data version differs from the current synthetic dataset version.",
                    next_action="Review dataset lineage and explicitly retrain the model on the intended synthetic data.",
                )

    artifact_statuses = [classifier["status"], regressor["status"]]
    statuses = [*artifact_statuses, evaluation["status"]]
    if "MISSING" in artifact_statuses:
        overall_status = "MISSING"
    elif "RETRAINING_REQUIRED" in statuses:
        overall_status = "RETRAINING_REQUIRED"
    elif any(status != "READY" for status in statuses):
        overall_status = "INCOMPLETE"
    else:
        overall_status = "READY"

    overall_actions = {
        "READY": "No action is indicated by the available integrity and compatibility checks.",
        "MISSING": "Restore each missing model artifact through the approved deployment workflow.",
        "INCOMPLETE": "Restore or regenerate the missing, stale, or inconsistent metadata/evaluation evidence.",
        "RETRAINING_REQUIRED": "Review the dataset/model lineage, then explicitly retrain and evaluate the intended bundle.",
    }

    return {
        "schema_version": "1.0",
        "status": overall_status,
        "next_action": overall_actions[overall_status],
        "package_version": PROJECT_VERSION,
        "dataset": {
            "kind": "synthetic",
            "version": DATASET_VERSION,
            "available": dataset_path is not None and dataset_sha is not None,
            "sha256": dataset_sha,
        },
        "evaluation": evaluation,
        "artifacts": [classifier, regressor],
        "read_only": True,
        "disclaimer": (
            "This inventory covers synthetic ML assets only. It does not assess or include official "
            "MinTIC connectivity data, and it does not establish real-world forecast validity."
        ),
    }
