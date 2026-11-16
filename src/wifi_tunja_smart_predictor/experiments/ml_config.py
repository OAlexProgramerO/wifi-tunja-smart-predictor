"""Task contracts, candidate allowlists, and bounded experiment configuration."""

from __future__ import annotations

from typing import Any

from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    RANDOM_SEED,
    REGRESSION_TARGET_COLUMN,
    TARGET_COLUMN,
    TARGET_COLUMNS,
)
from wifi_tunja_smart_predictor.experiments.ml_schemas import MLExperimentCreate
from wifi_tunja_smart_predictor.models.regression import regression_specs
from wifi_tunja_smart_predictor.models.train import _baseline_specs

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


def candidate_specs(task: str) -> dict[str, dict[str, Any]]:
    """Return the existing, task-specific candidate specifications."""
    if task == "classification":
        return {str(spec["name"]): spec for spec in _baseline_specs()}
    if task == "regression":
        return {str(spec["name"]): spec for spec in regression_specs()}
    raise ValueError("Experiment task is unsupported.")


def normalize_configuration(payload: MLExperimentCreate) -> dict[str, Any]:
    """Validate candidate/parameters and return canonical JSON-safe configuration."""
    if set(MODEL_INPUT_COLUMNS) & set(TARGET_COLUMNS):
        raise ValueError("Configured model inputs include a target column.")
    specs = candidate_specs(payload.task)
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
