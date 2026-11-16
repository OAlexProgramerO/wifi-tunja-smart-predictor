"""Validated, chronological orchestration for reproducible synthetic ML runs."""

from __future__ import annotations

import hashlib
import platform
import warnings
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning

from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    PROCESSED_DATASET_PATH,
    PROJECT_VERSION,
    RANDOM_SEED,
    RAW_DATASET_PATH,
    REGRESSION_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_COLUMNS,
    TRAIN_END,
    VALIDATION_END,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.experiments.ml_schemas import MLExperimentCreate
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.regression import (
    evaluate_regressor,
    regression_specs,
)
from wifi_tunja_smart_predictor.models.train import _baseline_specs, temporal_split
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline

_PARAMETER_BOUNDS: dict[str, dict[str, tuple[float, float] | set[str]]] = {
    "logistic_regression": {"C": (0.01, 100.0), "max_iter": (50, 2000)},
    "decision_tree": {
        "max_depth": (1, 30),
        "min_samples_leaf": (1, 100),
        "ccp_alpha": (0.0, 1.0),
    },
    "random_forest": {
        "n_estimators": (10, 300),
        "max_depth": (2, 30),
        "min_samples_leaf": (1, 100),
    },
    "knn": {"n_neighbors": (1, 51), "p": (1, 2), "weights": {"uniform", "distance"}},
    "naive_bayes": {},
    "random_forest_regressor": {
        "n_estimators": (10, 300),
        "max_depth": (2, 30),
        "min_samples_leaf": (1, 100),
    },
    "hist_gradient_boosting_regressor": {
        "max_iter": (20, 500),
        "max_leaf_nodes": (2, 63),
        "learning_rate": (0.001, 1.0),
        "l2_regularization": (0.0, 100.0),
    },
}


def _specifications(task: str) -> dict[str, dict[str, Any]]:
    if task == "classification":
        return {str(spec["name"]): spec for spec in _baseline_specs()}
    return {str(spec["name"]): spec for spec in regression_specs()}


def normalize_configuration(payload: MLExperimentCreate) -> dict[str, Any]:
    """Validate candidate/parameters and return canonical JSON-safe configuration."""
    if set(MODEL_INPUT_COLUMNS) & set(TARGET_COLUMNS):
        raise ValueError("Configured model inputs include a target column.")
    specs = _specifications(payload.task)
    if payload.model_candidate not in specs:
        raise ValueError("Model candidate is not available for this task.")
    allowed = _PARAMETER_BOUNDS[payload.model_candidate]
    if set(payload.hyperparameters) - set(allowed):
        raise ValueError("One or more hyperparameters are not allowed for this candidate.")
    for name, value in payload.hyperparameters.items():
        bounds = allowed[name]
        if isinstance(bounds, set):
            if not isinstance(value, str) or value not in bounds:
                raise ValueError(f"Hyperparameter '{name}' is outside its allowed values.")
        elif (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not bounds[0] <= value <= bounds[1]
            or (isinstance(bounds[0], int) and not float(value).is_integer())
        ):
            raise ValueError(f"Hyperparameter '{name}' is outside its allowed range.")
    estimator = specs[payload.model_candidate]["estimator"]
    current_parameters = estimator.get_params(deep=False)
    resolved_parameters = {
        name: current_parameters[name] for name in allowed if name in current_parameters
    }
    resolved_parameters.update(payload.hyperparameters)
    if "n_jobs" in current_parameters:
        resolved_parameters["n_jobs"] = 1
    return {
        "schema_version": "1.0",
        "task": payload.task,
        "model_candidate": payload.model_candidate,
        "evaluation_split": payload.evaluation_split,
        "hyperparameters": dict(sorted(resolved_parameters.items())),
        "feature_columns": list(MODEL_INPUT_COLUMNS),
        "target_column": (
            TARGET_COLUMN if payload.task == "classification" else REGRESSION_TARGET_COLUMN
        ),
        "random_seed": RANDOM_SEED,
    }


def _load_synthetic_dataset() -> pd.DataFrame:
    path = PROCESSED_DATASET_PATH if PROCESSED_DATASET_PATH.is_file() else RAW_DATASET_PATH
    frame = load_raw_dataset(path)
    if path == RAW_DATASET_PATH:
        frame = frame.drop_duplicates()
    return frame


def _period(frame: pd.DataFrame, *, cutoff: str | None = None) -> dict[str, Any]:
    return {
        "start": frame["timestamp"].min().isoformat(),
        "end": frame["timestamp"].max().isoformat(),
        "rows": int(len(frame)),
        "cutoff": cutoff,
    }


def _dataset_fingerprint(frame: pd.DataFrame, target_column: str) -> str:
    columns = ["timestamp", *MODEL_INPUT_COLUMNS, target_column]
    hashed = pd.util.hash_pandas_object(frame[columns], index=False, categorize=True)
    return hashlib.sha256(hashed.values.tobytes()).hexdigest()


def execute_experiment(configuration: dict[str, Any]) -> dict[str, Any]:
    """Fit one candidate on synthetic train rows and score one chronological holdout."""
    target_column = configuration["target_column"]
    frame = _load_synthetic_dataset()
    required = {"timestamp", target_column, *MODEL_INPUT_COLUMNS}
    if not required.issubset(frame.columns):
        raise ValueError("Synthetic training dataset does not satisfy the configured schema.")
    if set(MODEL_INPUT_COLUMNS) & set(TARGET_COLUMNS):
        raise ValueError("Configured model inputs include a target column.")
    train, validation, test = temporal_split(frame)
    evaluation = validation if configuration["evaluation_split"] == "validation" else test
    if evaluation.empty:
        raise ValueError("Selected chronological evaluation partition is empty.")

    spec = _specifications(configuration["task"])[configuration["model_candidate"]]
    estimator = clone(spec["estimator"])
    parameters = configuration["hyperparameters"]
    if parameters:
        estimator.set_params(**parameters)
    if "n_jobs" in estimator.get_params(deep=False):
        estimator.set_params(n_jobs=1)
    pipeline = build_classifier_pipeline(
        estimator,
        scale_numeric=bool(spec.get("scale_numeric", False)),
    )
    x_train = train[MODEL_INPUT_COLUMNS]
    x_evaluation = evaluation[MODEL_INPUT_COLUMNS]
    if configuration["task"] == "classification":
        y_train = train[target_column].astype(str)
        y_evaluation = evaluation[target_column].astype(str)
    else:
        y_train = pd.to_numeric(train[target_column], errors="raise")
        y_evaluation = pd.to_numeric(evaluation[target_column], errors="raise")
    pipeline.fit(x_train, y_train)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        metrics = (
            evaluate_classifier(pipeline, x_evaluation, y_evaluation)
            if configuration["task"] == "classification"
            else evaluate_regressor(pipeline, x_evaluation, y_evaluation)
        )
    if configuration["task"] == "classification" and pd.Series(y_evaluation).nunique() < 2:
        metrics["roc_auc"] = None
    metrics = {
        key: (
            (float(value) if np.isscalar(value) and np.isfinite(value) else None)
            if isinstance(value, (float, np.floating))
            else value
        )
        for key, value in metrics.items()
    }
    return {
        "report_schema_version": "1.0",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "project_version": PROJECT_VERSION,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "dataset": {
            "kind": "synthetic",
            "fingerprint_algorithm": "sha256(pandas_hash_pandas_object_v1)",
            "fingerprint": _dataset_fingerprint(frame, target_column),
            "rows": int(len(frame)),
        },
        "temporal_split": {
            "strategy": "chronological",
            "train": _period(train, cutoff=TRAIN_END),
            "validation": (
                _period(validation, cutoff=VALIDATION_END)
                if not validation.empty
                else {"start": None, "end": None, "rows": 0, "cutoff": VALIDATION_END}
            ),
            "test": _period(test),
            "evaluation_partition": configuration["evaluation_split"],
        },
        "model": {
            "task": configuration["task"],
            "candidate": configuration["model_candidate"],
            "hyperparameters": configuration["hyperparameters"],
            "random_seed": configuration["random_seed"],
            "artifact_persisted": False,
            "artifact_fingerprint": None,
        },
        "evaluation": {"metrics": metrics, "rows": int(len(evaluation))},
        "limitations": [
            "All data and metrics in this report are synthetic.",
            "The fitted candidate is not persisted; artifact fingerprint is unavailable.",
            "Results do not establish real-world forecast validity for Tunja.",
            "Official MinTIC data is excluded from training and evaluation.",
        ],
    }
