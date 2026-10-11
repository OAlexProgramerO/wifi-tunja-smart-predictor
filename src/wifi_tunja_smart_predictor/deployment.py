"""Small deployment checks shared by APIs and the one-time resource initializer."""

from __future__ import annotations

from pathlib import Path

from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    DATA_DIR,
    MODEL_COMPARISON_PATH,
    MODEL_METADATA_PATH,
    MODEL_PATH,
    PROCESSED_DATASET_PATH,
    RAW_DATASET_PATH,
    REGRESSION_COMPARISON_PATH,
    REGRESSION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
)

RUNTIME_READY_MARKER = DATA_DIR / ".runtime-ready"

REQUIRED_DEMO_ASSETS: tuple[Path, ...] = (
    RUNTIME_READY_MARKER,
    RAW_DATASET_PATH,
    PROCESSED_DATASET_PATH,
    MODEL_PATH,
    MODEL_METADATA_PATH,
    CLASSIFICATION_METADATA_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_METADATA_PATH,
    MODEL_COMPARISON_PATH,
    REGRESSION_COMPARISON_PATH,
)


def missing_demo_assets(paths: tuple[Path, ...] | None = None) -> tuple[Path, ...]:
    """Return required demonstration resources that are not regular files."""
    paths = REQUIRED_DEMO_ASSETS if paths is None else paths
    return tuple(path for path in paths if not path.is_file())


def runtime_assets_available() -> bool:
    """Return whether all resources used by the APIs and dashboard are present."""
    return not missing_demo_assets()
