"""Prediction-time feature engineering.

Every derived column uses information that is available at ``timestamp``
(the prediction time). Targets ``demand_level`` and ``connections_next_hour``
are never read.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from wifi_tunja_smart_predictor.config import (
    COLUMNS_DROPPED_AFTER_ENGINEERING,
    ENGINEERED_COLUMNS,
    LEAKAGE_COLUMNS,
    MODEL_INPUT_COLUMNS,
)

logger = logging.getLogger(__name__)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise ratio; missing or non-positive denominators stay missing."""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    den = den.mask(den <= 0)
    return num / den


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with documented engineered columns appended.

    Rationale
    ---------
    hour_sin / hour_cos, day_of_week_sin / day_of_week_cos, month_sin / month_cos
        Cyclical encodings so hour 23 is close to hour 0 (and December to January).
    device_capacity_ratio
        Instantaneous occupancy relative to AP capacity at prediction time.
    session_device_ratio
        Share of connected devices that currently hold an active session.
    bandwidth_per_session
        Rough per-session load from the hour that just ended.
    network_stress_indicator
        Combined channel utilization and packet loss from the last completed hour.
    recent_to_daily_ratio / recent_to_weekly_ratio
        Short-term historical demand relative to longer historical baselines.
        Both baselines are lag features (hours strictly before ``timestamp``).
    """
    if any(col in frame.columns for col in LEAKAGE_COLUMNS if col in frame):
        # Presence is allowed on a full analytical frame; values are never used here.
        pass

    out = frame.copy()
    hour = pd.to_numeric(out["hour"], errors="coerce")
    dow = pd.to_numeric(out["day_of_week"], errors="coerce")
    month = pd.to_numeric(out["month"], errors="coerce")

    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["day_of_week_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["day_of_week_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    out["device_capacity_ratio"] = _safe_ratio(
        out["connected_devices"], out["access_point_capacity"]
    )
    out["session_device_ratio"] = _safe_ratio(out["active_sessions"], out["connected_devices"])
    out["bandwidth_per_session"] = _safe_ratio(out["bandwidth_usage_mbps"], out["active_sessions"])
    util = pd.to_numeric(out["channel_utilization_percent"], errors="coerce") / 100.0
    loss = pd.to_numeric(out["packet_loss_percent"], errors="coerce") / 100.0
    out["network_stress_indicator"] = util * loss
    out["recent_to_daily_ratio"] = _safe_ratio(
        out["connections_previous_hour"], out["average_connections_last_24_hours"]
    )
    out["recent_to_weekly_ratio"] = _safe_ratio(
        out["connections_previous_hour"], out["average_connections_last_7_days"]
    )
    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible wrapper so engineering is persisted with the pipeline."""

    def fit(self, X: pd.DataFrame, y: object | None = None) -> FeatureEngineer:
        # Stateless transformer: no statistics are learned from X, which avoids
        # leaking evaluation-period distributions into feature construction.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        missing = [col for col in MODEL_INPUT_COLUMNS if col not in X.columns]
        if missing:
            raise ValueError(f"FeatureEngineer missing required input columns: {missing}")
        leaked = [col for col in LEAKAGE_COLUMNS if col in X.columns]
        if leaked:
            logger.debug("Dropping target columns before engineering: %s", leaked)
            X = X.drop(columns=leaked, errors="ignore")
        engineered = add_engineered_features(X)
        keep = [
            col
            for col in engineered.columns
            if col not in COLUMNS_DROPPED_AFTER_ENGINEERING and col not in LEAKAGE_COLUMNS
        ]
        # Restrict to model inputs + engineered fields; drop identifiers if present.
        allowed = set(MODEL_INPUT_COLUMNS) | set(ENGINEERED_COLUMNS)
        keep = [col for col in keep if col in allowed]
        return engineered[keep]
