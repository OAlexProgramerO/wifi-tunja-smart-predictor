"""Compatibility assessment and pairwise metric comparison for saved ML runs."""

from __future__ import annotations

from datetime import datetime, timedelta
from math import isfinite
from typing import Any

METRIC_DIRECTIONS: dict[str, dict[str, str]] = {
    "classification": {
        "accuracy": "higher_is_better",
        "precision": "higher_is_better",
        "recall": "higher_is_better",
        "f1": "higher_is_better",
        "roc_auc": "higher_is_better",
    },
    "regression": {
        "mae": "lower_is_better",
        "rmse": "lower_is_better",
        "r2": "higher_is_better",
    },
}
TARGET_NAMES = {"demand_level", "connections_next_hour"}
FORBIDDEN_FEATURE_TOKENS = {"future", "next", "target", "label"}
TIE_TOLERANCE = 1e-9
REASON_EXPLANATIONS = {
    "EXPERIMENT_NOT_COMPLETED": "Both experiments must have completed successfully.",
    "TASK_METADATA_UNAVAILABLE": "A task type is missing or unsupported.",
    "EVALUATION_SPLIT_UNAVAILABLE": "A validation or test split is not recorded.",
    "EXPERIMENT_TIMESTAMPS_UNAVAILABLE": "UTC creation and completion timestamps are required.",
    "SYNTHETIC_DATASET_PROVENANCE_INCOMPLETE": "Synthetic dataset kind, fingerprint, or fingerprint algorithm is missing or invalid.",
    "FEATURE_CONFIGURATION_UNAVAILABLE": "The ordered model feature list is missing or malformed.",
    "TARGET_OR_FUTURE_FEATURE_LEAKAGE": "A target or future-information field appears in the recorded feature list.",
    "TARGET_CONFIGURATION_UNAVAILABLE": "The task target is missing or unsupported.",
    "TASK_TARGET_MISMATCH": "The recorded target does not match the classification or regression task.",
    "MODEL_CONFIGURATION_INCONSISTENT": "Stored model candidate, parameters, seed, and task metadata do not agree.",
    "CHRONOLOGICAL_EVALUATION_PROVENANCE_INCOMPLETE": "The selected chronological split or evaluated row count is missing.",
    "INVALID_CHRONOLOGICAL_WINDOWS": "The selected evaluation window is invalid or does not follow the training window.",
    "RUNTIME_PROVENANCE_INCOMPLETE": "Project or runtime library version metadata is missing.",
    "INCOMPATIBLE_TASK_TYPES": "Classification and regression runs cannot be compared to each other.",
    "INCOMPATIBLE_EVALUATION_SPLITS": "Both runs must evaluate the same validation or test partition.",
    "INCOMPATIBLE_PROJECT_VERSION": "The runs were produced by different project versions.",
    "INCOMPATIBLE_ENVIRONMENT": "The recorded Python or ML library environments differ.",
    "INCOMPATIBLE_RANDOM_SEED": "The runs used different random seeds.",
    "INCOMPATIBLE_FEATURE_COLUMNS": "The runs used different ordered model input features.",
    "INCOMPATIBLE_TARGET_COLUMN": "The runs used different target columns.",
    "INCOMPATIBLE_DATASET_FINGERPRINTS": "The source dataset fingerprints differ.",
    "INCOMPATIBLE_FINGERPRINT_ALGORITHMS": "The source fingerprints were produced with different algorithms.",
    "INCOMPATIBLE_CHRONOLOGICAL_WINDOWS": "The training or evaluation time windows differ.",
    "NO_SHARED_VALID_METRICS": "No shared finite task metrics are available for these runs.",
}


def _safe_record_metadata(record: dict[str, Any]) -> dict[str, Any]:
    config = record.get("configuration")
    result = record.get("result")
    config = config if isinstance(config, dict) else {}
    result = result if isinstance(result, dict) else {}
    model = result.get("model") if isinstance(result.get("model"), dict) else {}
    dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
    temporal = (
        result.get("temporal_split") if isinstance(result.get("temporal_split"), dict) else {}
    )
    evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
    return {
        "experiment_id": record.get("experiment_id"),
        "status": record.get("status"),
        "created_at_utc": record.get("created_at_utc"),
        "completed_at_utc": record.get("completed_at_utc"),
        "project_version": result.get("project_version"),
        "environment": result.get("environment"),
        "task": config.get("task"),
        "evaluation_split": config.get("evaluation_split"),
        "model_candidate": config.get("model_candidate"),
        "hyperparameters": config.get("hyperparameters"),
        "random_seed": config.get("random_seed"),
        "feature_columns": config.get("feature_columns"),
        "target_column": config.get("target_column"),
        "configured_candidate": config.get("model_candidate"),
        "configured_hyperparameters": config.get("hyperparameters"),
        "dataset": {
            "kind": dataset.get("kind"),
            "fingerprint_algorithm": dataset.get("fingerprint_algorithm"),
            "fingerprint": dataset.get("fingerprint"),
            "rows": dataset.get("rows"),
        },
        "temporal_split": temporal,
        "evaluation": evaluation,
        "artifact": {
            "persisted": model.get("artifact_persisted"),
            "fingerprint": model.get("artifact_fingerprint"),
        },
        "result_model": {
            "task": model.get("task"),
            "candidate": model.get("candidate"),
            "hyperparameters": model.get("hyperparameters"),
            "random_seed": model.get("random_seed"),
        },
    }


def _compatibility_reasons(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for item in (left, right):
        if item.get("status") != "COMPLETED":
            reasons.append("EXPERIMENT_NOT_COMPLETED")
        if item.get("task") not in METRIC_DIRECTIONS:
            reasons.append("TASK_METADATA_UNAVAILABLE")
        if item.get("evaluation_split") not in {"validation", "test"}:
            reasons.append("EVALUATION_SPLIT_UNAVAILABLE")
        try:
            datetime.fromisoformat(str(item.get("created_at_utc")).replace("Z", "+00:00"))
            datetime.fromisoformat(str(item.get("completed_at_utc")).replace("Z", "+00:00"))
        except ValueError:
            reasons.append("EXPERIMENT_TIMESTAMPS_UNAVAILABLE")
        dataset = item.get("dataset", {})
        fingerprint = dataset.get("fingerprint")
        if (
            dataset.get("kind") != "synthetic"
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in fingerprint)
            or not dataset.get("fingerprint_algorithm")
        ):
            reasons.append("SYNTHETIC_DATASET_PROVENANCE_INCOMPLETE")
        features = item.get("feature_columns")
        target = item.get("target_column")
        if (
            not isinstance(features, list)
            or not features
            or not all(isinstance(feature, str) for feature in features)
        ):
            reasons.append("FEATURE_CONFIGURATION_UNAVAILABLE")
        else:
            tokens = {
                token.casefold()
                for feature in features
                for token in feature.replace("-", "_").split("_")
            }
            if set(features) & TARGET_NAMES or tokens & FORBIDDEN_FEATURE_TOKENS:
                reasons.append("TARGET_OR_FUTURE_FEATURE_LEAKAGE")
        if not isinstance(target, str) or target not in TARGET_NAMES:
            reasons.append("TARGET_CONFIGURATION_UNAVAILABLE")
        expected_target = (
            "demand_level" if item.get("task") == "classification" else "connections_next_hour"
        )
        if item.get("task") in METRIC_DIRECTIONS and target != expected_target:
            reasons.append("TASK_TARGET_MISMATCH")
        model = item.get("result_model", {})
        if (
            not isinstance(item.get("configured_candidate"), str)
            or not item.get("configured_candidate")
            or not isinstance(item.get("configured_hyperparameters"), dict)
            or model.get("task") != item.get("task")
            or model.get("candidate") != item.get("configured_candidate")
            or model.get("hyperparameters") != item.get("configured_hyperparameters")
            or model.get("random_seed") != item.get("random_seed")
        ):
            reasons.append("MODEL_CONFIGURATION_INCONSISTENT")
        temporal = item.get("temporal_split")
        split = item.get("evaluation_split")
        selected = temporal.get(split) if isinstance(temporal, dict) else None
        train_period = temporal.get("train") if isinstance(temporal, dict) else None
        evaluation = item.get("evaluation")
        if (
            not isinstance(temporal, dict)
            or temporal.get("strategy") != "chronological"
            or temporal.get("evaluation_partition") != split
            or not isinstance(selected, dict)
            or not selected.get("start")
            or not selected.get("end")
            or not isinstance(selected.get("rows"), int)
            or not isinstance(evaluation, dict)
            or evaluation.get("rows") != selected.get("rows")
        ):
            reasons.append("CHRONOLOGICAL_EVALUATION_PROVENANCE_INCOMPLETE")
        else:
            try:
                train_end = datetime.fromisoformat(str(train_period["end"]).replace("Z", "+00:00"))
                evaluation_start = datetime.fromisoformat(
                    str(selected["start"]).replace("Z", "+00:00")
                )
                evaluation_end = datetime.fromisoformat(str(selected["end"]).replace("Z", "+00:00"))
                if (
                    evaluation_start <= train_end
                    or evaluation_end < evaluation_start
                    or selected["rows"] <= 0
                    or train_period.get("rows", 0) <= 0
                ):
                    reasons.append("INVALID_CHRONOLOGICAL_WINDOWS")
            except (TypeError, ValueError):
                reasons.append("INVALID_CHRONOLOGICAL_WINDOWS")
        try:
            created = datetime.fromisoformat(str(item.get("created_at_utc")).replace("Z", "+00:00"))
            completed = datetime.fromisoformat(
                str(item.get("completed_at_utc")).replace("Z", "+00:00")
            )
            if created.utcoffset() != timedelta(0) or completed.utcoffset() != timedelta(0):
                reasons.append("EXPERIMENT_TIMESTAMPS_UNAVAILABLE")
        except (TypeError, ValueError):
            pass
        if not item.get("project_version") or not isinstance(item.get("environment"), dict):
            reasons.append("RUNTIME_PROVENANCE_INCOMPLETE")
    if left.get("task") != right.get("task"):
        reasons.append("INCOMPATIBLE_TASK_TYPES")
    if left.get("evaluation_split") != right.get("evaluation_split"):
        reasons.append("INCOMPATIBLE_EVALUATION_SPLITS")
    for name in (
        "project_version",
        "environment",
        "random_seed",
        "feature_columns",
        "target_column",
    ):
        if left.get(name) != right.get(name):
            reasons.append(f"INCOMPATIBLE_{name.upper()}")
    left_dataset = left.get("dataset", {})
    right_dataset = right.get("dataset", {})
    if left_dataset.get("fingerprint") != right_dataset.get("fingerprint"):
        reasons.append("INCOMPATIBLE_DATASET_FINGERPRINTS")
    if left_dataset.get("fingerprint_algorithm") != right_dataset.get("fingerprint_algorithm"):
        reasons.append("INCOMPATIBLE_FINGERPRINT_ALGORITHMS")
    left_split = left.get("temporal_split", {})
    right_split = right.get("temporal_split", {})
    split_keys = ("strategy", "train", "validation", "test", "evaluation_partition")
    if any(left_split.get(key) != right_split.get(key) for key in split_keys):
        reasons.append("INCOMPATIBLE_CHRONOLOGICAL_WINDOWS")
    return sorted(set(reasons))


def _numeric_metrics(metadata: dict[str, Any]) -> dict[str, float]:
    evaluation = metadata.get("evaluation", {})
    metrics = evaluation.get("metrics") if isinstance(evaluation, dict) else None
    directions = METRIC_DIRECTIONS.get(metadata.get("task"), {})
    if not isinstance(metrics, dict):
        return {}
    return {
        name: float(value)
        for name, value in metrics.items()
        if name in directions
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(float(value))
    }


def _compare_pair(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    reasons = _compatibility_reasons(left, right)
    pair = {
        "left_experiment_id": left.get("experiment_id"),
        "right_experiment_id": right.get("experiment_id"),
        "left_model": {
            "candidate": left.get("model_candidate"),
            "hyperparameters": left.get("hyperparameters"),
        },
        "right_model": {
            "candidate": right.get("model_candidate"),
            "hyperparameters": right.get("hyperparameters"),
        },
        "status": "INCOMPATIBLE" if reasons else "COMPARABLE",
        "reasons": reasons,
        "reason_details": [
            {"code": code, "message": REASON_EXPLANATIONS.get(code, "Compatibility check failed.")}
            for code in reasons
        ],
        "metric_results": [],
        "unavailable_metrics": [],
    }
    if reasons:
        return pair
    left_metrics = _numeric_metrics(left)
    right_metrics = _numeric_metrics(right)
    directions = METRIC_DIRECTIONS[left["task"]]
    shared = sorted(set(left_metrics) & set(right_metrics) & set(directions))
    unavailable = sorted((set(left_metrics) | set(right_metrics) | set(directions)) - set(shared))
    pair["unavailable_metrics"] = [
        {
            "metric": name,
            "reason": "METRIC_MISSING_OR_NONFINITE_FOR_ONE_OR_BOTH_EXPERIMENTS",
        }
        for name in unavailable
    ]
    for name in shared:
        left_value = left_metrics[name]
        right_value = right_metrics[name]
        delta = right_value - left_value
        if abs(delta) <= TIE_TOLERANCE:
            assessment = "TIE"
        else:
            improved = delta > 0 if directions[name] == "higher_is_better" else delta < 0
            assessment = "IMPROVEMENT" if improved else "REGRESSION"
        pair["metric_results"].append(
            {
                "metric": name,
                "direction": directions[name],
                "left_value": left_value,
                "right_value": right_value,
                "delta": delta,
                "assessment": assessment,
            }
        )
    if not shared:
        pair["status"] = "PARTIAL"
        pair["reasons"].append("NO_SHARED_VALID_METRICS")
    elif unavailable:
        pair["status"] = "PARTIAL"
    pair["reason_details"] = [
        {"code": code, "message": REASON_EXPLANATIONS.get(code, "Compatibility check failed.")}
        for code in pair["reasons"]
    ]
    return pair


def compare_experiment_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare each unordered pair without ranking models across incompatible evidence."""
    metadata = [_safe_record_metadata(record) for record in records]
    comparisons = [
        _compare_pair(metadata[left], metadata[right])
        for left in range(len(metadata))
        for right in range(left + 1, len(metadata))
    ]
    statuses = {item["status"] for item in comparisons}
    if statuses == {"COMPARABLE"}:
        overall = "COMPARABLE"
    elif "COMPARABLE" in statuses or "PARTIAL" in statuses:
        overall = "PARTIAL"
    else:
        overall = "INCOMPATIBLE"
    return {
        "overall_status": overall,
        "experiments": metadata,
        "comparisons": comparisons,
        "interpretation": (
            "Pairwise metric differences describe these synthetic evaluation runs only. "
            "No universal best model is selected."
        ),
    }
