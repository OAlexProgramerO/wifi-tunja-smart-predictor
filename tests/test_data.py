"""Data loading and validation tests (in-memory fixtures only)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from wifi_tunja_smart_predictor.config import REQUIRED_COLUMNS, TARGET_COLUMN, TARGET_LABELS
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset, resolve_dataset_path
from wifi_tunja_smart_predictor.data.validator import validate_dataset
from wifi_tunja_smart_predictor.exceptions import DatasetNotFoundError, DatasetValidationError


def test_resolve_dataset_path_missing(tmp_path: Path) -> None:
    with pytest.raises(DatasetNotFoundError):
        resolve_dataset_path(tmp_path / "nope.csv")


def test_load_raw_dataset_parses_timestamp(tmp_path: Path, mini_frame: pd.DataFrame) -> None:
    path = tmp_path / "sample.csv"
    mini_frame.to_csv(path, index=False)
    loaded = load_raw_dataset(path)
    assert set(REQUIRED_COLUMNS).issubset(loaded.columns)
    assert pd.api.types.is_datetime64_any_dtype(loaded["timestamp"])
    assert loaded["timestamp"].notna().all()


def test_expected_columns_exist(mini_frame: pd.DataFrame) -> None:
    for column in REQUIRED_COLUMNS:
        assert column in mini_frame.columns


def test_target_labels_are_valid(mini_frame: pd.DataFrame) -> None:
    assert set(mini_frame[TARGET_COLUMN].unique()) <= set(TARGET_LABELS)


def test_validator_accepts_expected_missingness(mini_frame: pd.DataFrame) -> None:
    frame = mini_frame.copy()
    frame.loc[frame.index[0], "temperature_c"] = pd.NA
    report = validate_dataset(frame, raise_on_critical=True)
    assert report.is_valid
    assert report.expected_issues


def test_validator_rejects_bad_target(mini_frame: pd.DataFrame) -> None:
    frame = mini_frame.copy()
    frame.loc[0, TARGET_COLUMN] = "MEDIUM"
    with pytest.raises(DatasetValidationError):
        validate_dataset(frame)
