"""CSV loading helpers. Paths are repository-relative, never machine-specific."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from wifi_tunja_smart_predictor.config import RAW_DATASET_PATH
from wifi_tunja_smart_predictor.exceptions import DatasetNotFoundError

logger = logging.getLogger(__name__)


def resolve_dataset_path(path: Path | str | None = None) -> Path:
    """Return an existing dataset path or raise DatasetNotFoundError."""
    candidate = Path(path) if path is not None else RAW_DATASET_PATH
    if not candidate.is_file():
        raise DatasetNotFoundError(
            f"Dataset not found at '{candidate}'. Generate it from the "
            "repository root with: python scripts/generate_synthetic_dataset.py"
        )
    return candidate


def load_raw_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """Load the synthetic WiFi CSV and parse ``timestamp``.

    The file is not assumed to contain real municipal measurements.
    """
    dataset_path = resolve_dataset_path(path)
    logger.info("Loading dataset from %s", dataset_path)
    frame = pd.read_csv(dataset_path)
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    logger.info("Loaded %s rows and %s columns", f"{len(frame):,}", frame.shape[1])
    return frame
