"""Reproducible synthetic feature-distribution drift analysis."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from wifi_tunja_smart_predictor.config import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    MODEL_INPUT_COLUMNS,
    PROCESSED_DATASET_PATH,
    PROJECT_VERSION,
    RAW_DATASET_PATH,
    TARGET_COLUMNS,
)

REPORT_SCHEMA_VERSION = "1.0"
MIN_SAMPLE_SIZE = 20
PSI_BIN_COUNT = 10
PSI_SMOOTHING = 0.0001
THRESHOLDS = {
    "psi_warning": 0.10,
    "psi_drift": 0.20,
    "categorical_tv_warning": 0.10,
    "categorical_tv_drift": 0.25,
    "missing_difference_warning": 0.05,
    "missing_difference_drift": 0.20,
    "unseen_category_warning_share": 0.0,
    "unseen_category_drift_share": 0.10,
    "mean_shift_warning": 1.0,
    "mean_shift_drift": 2.0,
}
_FORBIDDEN_FEATURE_NAME_PARTS = ("next", "future", "target", "label", "lead_")


def _safe_number(value: float | int | None) -> float | int | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _feature_names() -> list[str]:
    forbidden = set(TARGET_COLUMNS) | set(LEAKAGE_COLUMNS)
    return [
        name
        for name in MODEL_INPUT_COLUMNS
        if name not in forbidden
        and not any(part in name.casefold() for part in _FORBIDDEN_FEATURE_NAME_PARTS)
    ]


def _fingerprint_features(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()


def _numeric_psi(reference: np.ndarray, evaluation: np.ndarray) -> tuple[float, int]:
    internal = np.quantile(reference, np.linspace(0, 1, PSI_BIN_COUNT + 1)[1:-1])
    edges = np.concatenate(([-np.inf], np.unique(internal), [np.inf]))
    if len(edges) < 2:
        edges = np.array([-np.inf, np.inf])
    ref_counts = np.histogram(reference, bins=edges)[0].astype(float)
    eval_counts = np.histogram(evaluation, bins=edges)[0].astype(float)
    bin_count = len(edges) - 1
    ref_share = (ref_counts + PSI_SMOOTHING) / (ref_counts.sum() + PSI_SMOOTHING * bin_count)
    eval_share = (eval_counts + PSI_SMOOTHING) / (eval_counts.sum() + PSI_SMOOTHING * bin_count)
    psi = float(np.sum((eval_share - ref_share) * np.log(eval_share / ref_share)))
    return psi, bin_count


def _severity(
    *,
    psi: float | None = None,
    mean_shift: float | None = None,
    categorical_tv: float | None = None,
    missing_difference: float,
    unseen_share: float = 0.0,
) -> str:
    if (
        (psi is not None and psi >= THRESHOLDS["psi_drift"])
        or (mean_shift is not None and mean_shift >= THRESHOLDS["mean_shift_drift"])
        or (categorical_tv is not None and categorical_tv >= THRESHOLDS["categorical_tv_drift"])
        or missing_difference >= THRESHOLDS["missing_difference_drift"]
        or unseen_share >= THRESHOLDS["unseen_category_drift_share"]
    ):
        return "DRIFT_DETECTED"
    if (
        (psi is not None and psi >= THRESHOLDS["psi_warning"])
        or (mean_shift is not None and mean_shift >= THRESHOLDS["mean_shift_warning"])
        or (categorical_tv is not None and categorical_tv >= THRESHOLDS["categorical_tv_warning"])
        or missing_difference >= THRESHOLDS["missing_difference_warning"]
        or unseen_share > THRESHOLDS["unseen_category_warning_share"]
    ):
        return "WARNING"
    return "STABLE"


def _summarize_feature(
    name: str,
    reference: pd.Series,
    evaluation: pd.Series,
    *,
    min_samples: int,
) -> dict[str, Any]:
    ref_missing_count = int(reference.isna().sum())
    eval_missing_count = int(evaluation.isna().sum())
    ref_missing = ref_missing_count / len(reference) if len(reference) else 1.0
    eval_missing = eval_missing_count / len(evaluation) if len(evaluation) else 1.0
    missing_difference = abs(eval_missing - ref_missing)
    category = name in CATEGORICAL_FEATURES
    result: dict[str, Any] = {
        "feature": name,
        "kind": "categorical" if category else "numeric",
        "reference_rows": int(len(reference)),
        "evaluation_rows": int(len(evaluation)),
        "reference_missing_count": ref_missing_count,
        "evaluation_missing_count": eval_missing_count,
        "reference_missing_proportion": round(ref_missing, 6),
        "evaluation_missing_proportion": round(eval_missing, 6),
        "missing_proportion_difference": round(missing_difference, 6),
        "status": "STABLE",
    }
    if category:
        ref_values = reference.dropna().astype(str)
        eval_values = evaluation.dropna().astype(str)
        unseen = sorted(set(eval_values.unique()) - set(ref_values.unique()))
        unseen_share = (
            float(eval_values.isin(unseen).sum() / len(eval_values)) if len(eval_values) else 0.0
        )
        result.update(
            reference_non_missing_count=int(len(ref_values)),
            evaluation_non_missing_count=int(len(eval_values)),
            unseen_categories=unseen[:20],
            unseen_category_count=len(unseen),
            unseen_category_share=round(unseen_share, 6),
            reference_category_counts={
                str(key): int(value)
                for key, value in ref_values.value_counts().sort_index().items()
            },
            evaluation_category_counts={
                str(key): int(value)
                for key, value in eval_values.value_counts().sort_index().items()
            },
        )
        if len(ref_values) < min_samples or len(eval_values) < min_samples:
            result["status"] = "INSUFFICIENT_SAMPLES"
            result["method"] = "categorical_total_variation_distance"
            result["drift_score"] = None
            return result
        ref_dist = ref_values.value_counts(normalize=True)
        eval_dist = eval_values.value_counts(normalize=True)
        categories = ref_dist.index.union(eval_dist.index)
        tv_distance = float(
            0.5
            * sum(
                abs(float(ref_dist.get(key, 0.0)) - float(eval_dist.get(key, 0.0)))
                for key in categories
            )
        )
        result.update(
            method="categorical_total_variation_distance",
            drift_score=round(tv_distance, 6),
            categorical_total_variation=round(tv_distance, 6),
            status=_severity(
                categorical_tv=tv_distance,
                missing_difference=missing_difference,
                unseen_share=unseen_share,
            ),
        )
        return result

    ref_numeric = pd.to_numeric(reference, errors="coerce")
    eval_numeric = pd.to_numeric(evaluation, errors="coerce")
    ref_array = ref_numeric.to_numpy(dtype=float, na_value=np.nan)
    eval_array = eval_numeric.to_numpy(dtype=float, na_value=np.nan)
    ref_finite = np.isfinite(ref_array) & (np.abs(ref_array) <= 10**12)
    eval_finite = np.isfinite(eval_array) & (np.abs(eval_array) <= 10**12)
    invalid_ref = int((reference.notna().to_numpy() & ~ref_finite).sum())
    invalid_eval = int((evaluation.notna().to_numpy() & ~eval_finite).sum())
    ref_values = ref_array[ref_finite]
    eval_values = eval_array[eval_finite]
    result.update(
        reference_non_missing_count=int(len(ref_values)),
        evaluation_non_missing_count=int(len(eval_values)),
        reference_invalid_numeric_count=invalid_ref,
        evaluation_invalid_numeric_count=invalid_eval,
    )
    if len(ref_values) < min_samples or len(eval_values) < min_samples:
        result.update(
            status="INSUFFICIENT_SAMPLES",
            method="numeric_population_stability_index",
            drift_score=None,
        )
        return result
    psi, bin_count = _numeric_psi(ref_values, eval_values)
    ref_mean, eval_mean = float(np.mean(ref_values)), float(np.mean(eval_values))
    ref_std, eval_std = float(np.std(ref_values, ddof=1)), float(np.std(eval_values, ddof=1))
    scale = ref_std if ref_std > 0 else max(abs(ref_mean) * 1e-6, 1e-6)
    mean_shift = abs(eval_mean - ref_mean) / scale
    result.update(
        method="numeric_population_stability_index",
        drift_score=round(psi, 6),
        psi=round(psi, 6),
        bin_count=bin_count,
        normalized_mean_shift=round(mean_shift, 6),
        reference_summary={
            "mean": _safe_number(ref_mean),
            "median": _safe_number(np.median(ref_values)),
            "std": _safe_number(ref_std),
        },
        evaluation_summary={
            "mean": _safe_number(eval_mean),
            "median": _safe_number(np.median(eval_values)),
            "std": _safe_number(eval_std),
        },
        status=_severity(
            psi=psi,
            mean_shift=mean_shift,
            missing_difference=missing_difference,
        ),
    )
    return result


def analyze_feature_drift(
    reference_frame: pd.DataFrame,
    evaluation_frame: pd.DataFrame,
    *,
    evaluation_split: str,
    source_features_fingerprint: str,
    min_samples: int = MIN_SAMPLE_SIZE,
) -> dict[str, Any]:
    """Compare allowlisted model inputs; targets and unapproved columns are never inspected."""
    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test.")
    if not isinstance(reference_frame, pd.DataFrame) or not isinstance(
        evaluation_frame, pd.DataFrame
    ):
        raise ValueError("Reference and evaluation datasets must be dataframes.")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2.")
    approved_features = _feature_names()
    features = [
        name
        for name in approved_features
        if name in reference_frame.columns and name in evaluation_frame.columns
    ]
    if not approved_features:
        raise ValueError("No allowlisted model input features are configured.")
    reference = reference_frame.loc[:, features].copy()
    evaluation = evaluation_frame.loc[:, features].copy()
    feature_reports = []
    missing_reference = sorted(set(approved_features) - set(reference_frame.columns))
    missing_evaluation = sorted(set(approved_features) - set(evaluation_frame.columns))
    for name in approved_features:
        if name not in features:
            missing_from = []
            if name in missing_reference:
                missing_from.append("reference")
            if name in missing_evaluation:
                missing_from.append("evaluation")
            feature_reports.append(
                {
                    "feature": name,
                    "kind": "categorical" if name in CATEGORICAL_FEATURES else "numeric",
                    "status": "MISSING_FEATURE",
                    "missing_from": missing_from,
                    "drift_score": None,
                }
            )
            continue
        feature_reports.append(
            _summarize_feature(
                name,
                reference[name],
                evaluation[name],
                min_samples=min_samples,
            )
        )
    statuses = {item["status"] for item in feature_reports}
    overall = (
        "DRIFT_DETECTED"
        if "DRIFT_DETECTED" in statuses
        else (
            "WARNING"
            if statuses & {"WARNING", "INSUFFICIENT_SAMPLES", "MISSING_FEATURE"}
            else "STABLE"
        )
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": "synthetic_feature_drift",
        "project_version": PROJECT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": overall,
        "evaluation_split": evaluation_split,
        "synthetic_data_only": True,
        "target_exclusion": {
            "excluded_target_columns": sorted(set(TARGET_COLUMNS) | set(LEAKAGE_COLUMNS)),
            "analyzed_features": features,
            "approved_features": approved_features,
            "policy": "Only allowlisted MODEL_INPUT_COLUMNS are analyzed; targets and all other columns are ignored.",
        },
        "dataset_fingerprints": {
            "source_features_sha256": source_features_fingerprint,
            "reference_features_sha256": _fingerprint_features(reference),
            "evaluation_features_sha256": _fingerprint_features(evaluation),
        },
        "reference": {
            "partition": "chronological_training",
            "rows": int(len(reference)),
            "features": features,
        },
        "evaluation": {
            "partition": evaluation_split,
            "rows": int(len(evaluation)),
            "features": features,
        },
        "feature_coverage": {
            "missing_from_reference": missing_reference,
            "missing_from_evaluation": missing_evaluation,
        },
        "configuration": {
            "min_non_missing_samples": min_samples,
            "psi_bins": PSI_BIN_COUNT,
            "psi_smoothing": PSI_SMOOTHING,
            "thresholds": THRESHOLDS.copy(),
        },
        "feature_results": feature_reports,
        "limitations": [
            "Source datasets are synthetic and do not establish real public WiFi demand changes.",
            "Statistical feature drift is not proof of model degradation or a causal change.",
            "Fixed PSI and total-variation thresholds are descriptive and are not significance tests.",
            "Only model input features are analyzed; target columns and unapproved columns are excluded.",
        ],
    }


def build_current_drift_report(evaluation_split: str) -> dict[str, Any]:
    """Build a report from the current synthetic dataset without writing files."""
    from wifi_tunja_smart_predictor.models.train import temporal_split

    if evaluation_split not in {"validation", "test"}:
        raise ValueError("evaluation_split must be validation or test.")
    dataset_path = next(
        (path for path in (PROCESSED_DATASET_PATH, RAW_DATASET_PATH) if path.is_file()), None
    )
    if dataset_path is None:
        raise FileNotFoundError("Synthetic model input dataset is unavailable.")
    feature_columns = _feature_names()
    frame = pd.read_csv(
        dataset_path,
        usecols=lambda name: name == "timestamp" or name in feature_columns,
    )
    if "timestamp" not in frame:
        raise ValueError(
            "Synthetic dataset is missing the timestamp required for chronological splits."
        )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], format="mixed", errors="raise")
    train, validation, test = temporal_split(frame)
    evaluation = validation if evaluation_split == "validation" else test
    if evaluation.empty:
        raise ValueError(f"Synthetic {evaluation_split} partition is empty.")
    source_features = [name for name in _feature_names() if name in frame.columns]
    if not source_features:
        raise ValueError("No allowlisted model input features are present in the source dataset.")
    return analyze_feature_drift(
        train,
        evaluation,
        evaluation_split=evaluation_split,
        source_features_fingerprint=_fingerprint_features(frame.loc[:, source_features]),
    )
