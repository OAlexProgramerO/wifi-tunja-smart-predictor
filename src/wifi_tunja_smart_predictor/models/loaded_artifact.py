"""Identity for the exact persisted bytes used to load a model."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib


@dataclass(frozen=True)
class LoadedArtifact:
    """A deserialized estimator and a safe identity for its source bytes."""

    estimator: Any
    sha256: str
    size_bytes: int


def deserialize_artifact(path: Path) -> LoadedArtifact:
    """Hash and deserialize one byte snapshot so the identity matches inference."""
    artifact_bytes = path.read_bytes()
    estimator = joblib.load(io.BytesIO(artifact_bytes))
    return LoadedArtifact(
        estimator=estimator,
        sha256=hashlib.sha256(artifact_bytes).hexdigest(),
        size_bytes=len(artifact_bytes),
    )
