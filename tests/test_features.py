"""Feature engineering tests — leakage and prediction-time constraints."""

from __future__ import annotations

import pandas as pd

from wifi_tunja_smart_predictor.config import ENGINEERED_COLUMNS, LEAKAGE_COLUMNS
from wifi_tunja_smart_predictor.features.engineering import FeatureEngineer, add_engineered_features


def test_engineered_features_are_created(mini_frame: pd.DataFrame) -> None:
    out = add_engineered_features(mini_frame)
    for column in ENGINEERED_COLUMNS:
        assert column in out.columns
        assert out[column].notna().any()


def test_targets_are_not_used_for_engineering(mini_frame: pd.DataFrame) -> None:
    baseline = add_engineered_features(mini_frame.drop(columns=list(LEAKAGE_COLUMNS)))
    mutated = mini_frame.copy()
    mutated["demand_level"] = "HIGH"
    mutated["connections_next_hour"] = 9999
    compared = add_engineered_features(mutated.drop(columns=list(LEAKAGE_COLUMNS)))
    for column in ENGINEERED_COLUMNS:
        pd.testing.assert_series_equal(
            baseline[column].reset_index(drop=True),
            compared[column].reset_index(drop=True),
            check_names=False,
        )


def test_feature_engineer_drops_targets(mini_frame: pd.DataFrame) -> None:
    transformed = FeatureEngineer().fit_transform(mini_frame)
    for column in LEAKAGE_COLUMNS:
        assert column not in transformed.columns


def test_no_future_timestamp_features(mini_frame: pd.DataFrame) -> None:
    out = add_engineered_features(mini_frame)
    added = set(out.columns) - set(mini_frame.columns)
    assert not any("next" in name.lower() or "future" in name.lower() for name in added)
    assert "demand_level" not in added
