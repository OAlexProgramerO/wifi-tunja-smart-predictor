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
    RAW_DATASET_PATH,
    TARGET_COLUMNS,
    TRAIN_END,
    VALIDATION_END,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.experiments.ml_config import candidate_specs
from wifi_tunja_smart_predictor.models.evaluate import evaluate_classifier
from wifi_tunja_smart_predictor.models.regression import (
    evaluate_regressor,
)
from wifi_tunja_smart_predictor.models.train import temporal_split
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline


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

    spec = candidate_specs(configuration["task"])[configuration["model_candidate"]]
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
        "reproducibility": {
            "status": "TRACEABLE_NOT_SELF_CONTAINED",
            "exact_replay_verified": False,
            "requirements": [
                "The same synthetic dataset bytes must remain available.",
                "The same source code and recorded library versions are required.",
            ],
        },
        "evaluation": {"metrics": metrics, "rows": int(len(evaluation))},
        "limitations": [
            "All data and metrics in this report are synthetic.",
            "The fitted candidate is not persisted; artifact fingerprint is unavailable.",
            "Results do not establish real-world forecast validity for Tunja.",
            "Official MinTIC data is excluded from training and evaluation.",
        ],
    }
