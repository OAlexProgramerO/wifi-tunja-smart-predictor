"""Safe, response-ready model and prediction trace metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.models.loaded_artifact import LoadedArtifact


def _model_reference(
    metadata: dict[str, Any], artifact: LoadedArtifact | None
) -> dict[str, Any]:
    return {
        "model_version": metadata.get("project_version"),
        "selected_model": metadata.get("selected_model"),
        "artifact_sha256": artifact.sha256 if artifact is not None else None,
        "artifact_size_bytes": artifact.size_bytes if artifact is not None else None,
    }


def build_prediction_traceability(
    *,
    classifier_metadata: dict[str, Any],
    classifier_artifact: LoadedArtifact | None,
    regressor_metadata: dict[str, Any] | None = None,
    regressor_artifact: LoadedArtifact | None = None,
) -> dict[str, Any]:
    """Create a fresh trace record without disclosing local artifact paths."""
    trace = {
        "prediction_id": str(uuid4()),
        "predicted_at_utc": datetime.now(timezone.utc).isoformat(),
        "application_version": PROJECT_VERSION,
        "classifier": _model_reference(classifier_metadata, classifier_artifact),
    }
    trace["regressor"] = (
        _model_reference(regressor_metadata, regressor_artifact)
        if regressor_metadata is not None
        else None
    )
    return trace
