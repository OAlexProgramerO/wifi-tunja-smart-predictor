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


def metadata_matches_artifact(metadata: dict[str, Any], artifact: LoadedArtifact) -> bool:
    """Check optional persisted integrity fields against the bytes actually loaded.

    Metadata written before artifact fingerprints were introduced remains readable,
    while partially populated or mismatched integrity records fail closed.
    """
    expected_sha256 = metadata.get("artifact_sha256")
    expected_size = metadata.get("artifact_size_bytes")
    if expected_sha256 is None and expected_size is None:
        return True
    return (
        isinstance(expected_sha256, str)
        and len(expected_sha256) == 64
        and all(character in "0123456789abcdef" for character in expected_sha256)
        and type(expected_size) is int
        and expected_size > 0
        and expected_sha256 == artifact.sha256
        and expected_size == artifact.size_bytes
    )
