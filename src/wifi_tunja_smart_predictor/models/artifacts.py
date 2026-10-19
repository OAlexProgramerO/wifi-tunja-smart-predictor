"""Validated model metadata and loadability checks for runtime readiness."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

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
from wifi_tunja_smart_predictor.models.loaded_artifact import (
    LoadedArtifact,
    deserialize_artifact,
    metadata_matches_artifact,
)
from wifi_tunja_smart_predictor.models.predict import (
    classifier_supports_demand_labels,
    load_model,
)

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
        or combined.get("artifact_sha256") != classifier.get("artifact_sha256")
        or combined.get("artifact_size_bytes") != classifier.get("artifact_size_bytes")
        or not model_bundle_metadata_matches(combined, classifier)
    ):
        raise ModelMetadataError("Classifier metadata files are inconsistent.")
    return combined


def load_regression_metadata(path: Path = REGRESSION_METADATA_PATH) -> dict[str, Any]:
    """Read and validate the regressor metadata used by scenario predictions."""
    return _validate_metadata(_read_metadata(path))


_BUNDLE_IDENTITY_FIELDS = (
    "training_run_id",
    "training_timestamp_utc",
    "dataset_version",
    "preprocessing_version",
    "train_period",
    "validation_period",
    "test_period",
)


def model_bundle_metadata_matches(classifier: dict[str, Any], regressor: dict[str, Any]) -> bool:
    """Require classifier/regressor metadata to identify the same training run."""
    # Legacy bundles have a shared training timestamp; new bundles also carry
    # a UUID. When either side records an identity field, both must agree.
    for field in _BUNDLE_IDENTITY_FIELDS:
        left = classifier.get(field)
        right = regressor.get(field)
        if (left is not None or right is not None) and left != right:
            return False
    return True


def artifact_matches_metadata(metadata: dict[str, Any], artifact: LoadedArtifact) -> bool:
    """Validate optional persisted artifact identity against loaded bytes."""
    return metadata_matches_artifact(metadata, artifact)


@lru_cache(maxsize=4)
def _deserialize_regression(
    path: str, modified_ns: int, changed_ns: int, size: int
) -> LoadedArtifact:
    del modified_ns, changed_ns, size  # File signature invalidates cache after replacement.
    try:
        return deserialize_artifact(Path(path))
    except Exception as exc:
        raise ModelArtifactError("The regression model artifact could not be loaded.") from exc


def load_regression_artifact(path: Path | None = None) -> LoadedArtifact:
    """Load a regressor and identify the exact serialized bytes used."""
    model_path = path or REGRESSION_MODEL_PATH
    if not model_path.is_file():
        raise ModelNotFoundError("The regression model artifact is unavailable.")
    try:
        stat = model_path.stat()
    except OSError as exc:
        raise ModelArtifactError("The regression model artifact is unavailable.") from exc
    return _deserialize_regression(
        str(model_path), stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size
    )


def load_regression_model(path: Path | None = None) -> Any:
    """Load a regressor once per file signature and normalize errors."""
    return load_regression_artifact(path).estimator


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
        if not model_bundle_metadata_matches(classifier_metadata, regression_metadata):
            return False
        if not model_bundle_metadata_matches(classifier_metadata, combined_regression):
            return False

        classifier_identity = any(
            classifier_metadata.get(field) is not None
            for field in ("artifact_sha256", "artifact_size_bytes")
        )
        regressor_identity = any(
            metadata.get(field) is not None
            for metadata in (regression_metadata, combined_regression)
            for field in ("artifact_sha256", "artifact_size_bytes")
        )
        classifier_artifact = None
        regressor_artifact = None
        if classifier_identity:
            from wifi_tunja_smart_predictor.models.predict import load_model_artifact

            classifier_artifact = load_model_artifact()
            if not artifact_matches_metadata(classifier_metadata, classifier_artifact):
                return False
            classifier = classifier_artifact.estimator
        else:
            classifier = load_model()
        if regressor_identity:
            regressor_artifact = load_regression_artifact()
            if not artifact_matches_metadata(regression_metadata, regressor_artifact):
                return False
            regressor = regressor_artifact.estimator
        else:
            regressor = load_regression_model()
        if classifier_artifact is not None and not artifact_matches_metadata(
            combined_regression, classifier_artifact
        ):
            return False
        if regressor_artifact is not None and not artifact_matches_metadata(
            combined_regression, regressor_artifact
        ):
            return False
        return (
            callable(getattr(classifier, "predict", None))
            and callable(getattr(classifier, "predict_proba", None))
            and classifier_supports_demand_labels(classifier)
            and callable(getattr(regressor, "predict", None))
        )
    except (ModelArtifactError, ModelMetadataError, ModelNotFoundError, OSError, ValueError):
        logger.warning("Model artifacts failed the readiness check.")
        return False
