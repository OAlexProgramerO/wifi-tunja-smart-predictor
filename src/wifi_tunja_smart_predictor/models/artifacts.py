"""Validated model metadata and loadability checks for runtime readiness."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    REGRESSION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
)
from wifi_tunja_smart_predictor.exceptions import (
    ModelArtifactError,
    ModelMetadataError,
    ModelNotFoundError,
)
from wifi_tunja_smart_predictor.models.predict import load_model

logger = logging.getLogger(__name__)


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ModelMetadataError("Required model metadata is unavailable.")
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ModelMetadataError("Required model metadata is invalid.") from exc
    if not isinstance(metadata, dict):
        raise ModelMetadataError("Required model metadata is invalid.")
    return metadata


def _validate_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    version = metadata.get("project_version")
    selected_model = metadata.get("selected_model")
    feature_count = metadata.get("feature_count")
    feature_names = metadata.get("feature_names")
    if (
        not isinstance(version, str)
        or not version.strip()
        or not isinstance(selected_model, str)
        or not selected_model.strip()
        or type(feature_count) is not int
        or feature_count != len(MODEL_INPUT_COLUMNS)
        or feature_names != MODEL_INPUT_COLUMNS
    ):
        raise ModelMetadataError("Required model metadata is incomplete or inconsistent.")
    return metadata


def load_classifier_metadata(
    path: Path = MODEL_METADATA_PATH,
    classification_path: Path = CLASSIFICATION_METADATA_PATH,
) -> dict[str, Any]:
    """Read and validate the combined and classifier-specific metadata files."""
    combined = _validate_metadata(_read_metadata(path))
    classifier = _validate_metadata(_read_metadata(classification_path))
    if (
        combined["project_version"] != classifier["project_version"]
        or combined["selected_model"] != classifier["selected_model"]
    ):
        raise ModelMetadataError("Classifier metadata files are inconsistent.")
    return combined


def load_regression_metadata(path: Path = REGRESSION_METADATA_PATH) -> dict[str, Any]:
    """Read and validate the regressor metadata used by scenario predictions."""
    return _validate_metadata(_read_metadata(path))


@lru_cache(maxsize=4)
def _deserialize_regression(path: str, modified_ns: int, size: int) -> Any:
    del modified_ns, size  # File signature invalidates the cache when an artifact changes.
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelArtifactError("The regression model artifact could not be loaded.") from exc


def load_regression_model(path: Path | None = None) -> Any:
    """Load a regressor once per file signature and normalize deserialization errors."""
    model_path = path or REGRESSION_MODEL_PATH
    if not model_path.is_file():
        raise ModelNotFoundError("The regression model artifact is unavailable.")
    stat = model_path.stat()
    return _deserialize_regression(str(model_path), stat.st_mtime_ns, stat.st_size)


def model_artifacts_available() -> bool:
    """Return whether both models and their versioned metadata can be loaded."""
    try:
        classifier_metadata = load_classifier_metadata()
        regression_metadata = load_regression_metadata()
        combined_regression = classifier_metadata.get("regression")
        if not isinstance(combined_regression, dict):
            return False
        combined_regression = _validate_metadata(combined_regression)
        if (
            regression_metadata["project_version"] != classifier_metadata["project_version"]
            or combined_regression["project_version"] != regression_metadata["project_version"]
            or combined_regression["selected_model"] != regression_metadata["selected_model"]
        ):
            return False

        classifier = load_model()
        regressor = load_regression_model()
        return (
            callable(getattr(classifier, "predict", None))
            and callable(getattr(classifier, "predict_proba", None))
            and hasattr(classifier, "classes_")
            and callable(getattr(regressor, "predict", None))
        )
    except (ModelArtifactError, ModelMetadataError, ModelNotFoundError, OSError, ValueError):
        logger.warning("Model artifacts failed the readiness check.")
        return False
