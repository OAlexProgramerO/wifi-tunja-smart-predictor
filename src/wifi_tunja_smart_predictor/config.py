"""Central configuration for WiFi Tunja Smart Predictor (VERSION 0.3.4).

Paths are resolved relative to the repository root so the project works on any
machine as long as it is executed from a clone of this repository.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

try:
    PROJECT_VERSION = version("wifi-tunja-smart-predictor")
except PackageNotFoundError:
    PROJECT_VERSION = "0.3.4"
PROJECT_NAME = "WiFi Tunja Smart Predictor"
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Repository root: src/wifi_tunja_smart_predictor/config.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "sample"

RAW_DATASET_FILENAME = "wifi_tunja_public.csv"
RAW_DATASET_PATH = RAW_DATA_DIR / RAW_DATASET_FILENAME
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "wifi_tunja_public_processed.csv"
SAMPLE_DATASET_PATH = SAMPLE_DATA_DIR / "wifi_tunja_public_sample.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "wifi_demand_classifier.joblib"
MODEL_PATH = MODELS_DIR / MODEL_FILENAME
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
CLASSIFICATION_METADATA_PATH = MODELS_DIR / "classification_metadata.json"
REGRESSION_MODEL_FILENAME = "wifi_demand_regressor.joblib"
REGRESSION_MODEL_PATH = MODELS_DIR / REGRESSION_MODEL_FILENAME
REGRESSION_METADATA_PATH = MODELS_DIR / "regression_metadata.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODEL_COMPARISON_PATH = METRICS_DIR / "model_comparison.csv"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"
SELECTED_MODEL_METRICS_PATH = METRICS_DIR / "selected_model_test_metrics.json"
REGRESSION_COMPARISON_PATH = METRICS_DIR / "regression_model_comparison.csv"
SELECTED_REGRESSION_METRICS_PATH = METRICS_DIR / "selected_regression_test_metrics.json"
REGRESSION_RESIDUALS_FIGURE_PATH = FIGURES_DIR / "regression_residuals.png"
REGRESSION_PREDICTIONS_FIGURE_PATH = FIGURES_DIR / "regression_predictions.png"

TARGET_COLUMN = "demand_level"
SECONDARY_TARGET_COLUMN = "connections_next_hour"
REGRESSION_TARGET_COLUMN = SECONDARY_TARGET_COLUMN
TARGET_LABELS = ("LOW", "HIGH")
POSITIVE_LABEL = "HIGH"

# Calendar holdout. Future rows must never appear in training.
# The synthetic series covers 2024-01-01 through 2025-12-31.
TRAIN_END = "2025-06-30 23:59:59"
VALIDATION_END = "2025-09-30 23:59:59"
# Test period: 2025-10-01 00:00:00 through 2025-12-31 23:00:00

IDENTIFIER_COLUMNS = [
    "timestamp",
    "wifi_id",
    "zone_id",
    "zone_name",
    "zone_type",
    "latitude",
    "longitude",
    "altitude_m",
]
TEMPORAL_COLUMNS = [
    "date",
    "year",
    "month",
    "day",
    "day_of_week",
    "day_name",
    "week_of_year",
    "hour",
    "minute",
    "is_weekend",
    "is_holiday",
    "is_working_day",
    "is_school_day",
    "time_period",
]
WEATHER_COLUMNS = [
    "temperature_c",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_kmh",
    "weather_condition",
]
CONTEXT_COLUMNS = [
    "estimated_people_nearby",
    "traffic_level",
    "public_transport_activity",
    "nearby_business_activity",
    "nearby_student_population",
    "nearby_worker_population",
]
EVENT_COLUMNS = [
    "event_nearby",
    "event_type",
    "estimated_event_attendance",
]
NETWORK_COLUMNS = [
    "access_point_capacity",
    "connected_devices",
    "active_sessions",
    "average_session_duration_min",
    "bandwidth_usage_mbps",
    "packet_loss_percent",
    "latency_ms",
    "signal_strength_dbm",
    "channel_utilization_percent",
    "network_uptime_percent",
]
HISTORICAL_COLUMNS = [
    "connections_previous_hour",
    "connections_previous_day",
    "connections_same_hour_previous_day",
    "connections_same_hour_previous_week",
    "average_connections_last_3_hours",
    "average_connections_last_24_hours",
    "average_connections_last_7_days",
]
TARGET_COLUMNS = [SECONDARY_TARGET_COLUMN, TARGET_COLUMN]

REQUIRED_COLUMNS = (
    IDENTIFIER_COLUMNS
    + TEMPORAL_COLUMNS
    + WEATHER_COLUMNS
    + CONTEXT_COLUMNS
    + EVENT_COLUMNS
    + NETWORK_COLUMNS
    + HISTORICAL_COLUMNS
    + TARGET_COLUMNS
)

# Columns that must never be used as model inputs (targets + identifiers that
# would act as unique-location dummies without a documented reason).
LEAKAGE_COLUMNS = [TARGET_COLUMN, SECONDARY_TARGET_COLUMN]

# Raw columns consumed by feature engineering before sklearn preprocessing.
# zone_type is used instead of zone_name / wifi_id / zone_id: it describes land
# use and can generalise across synthetic access points. zone_name is a 1:1
# proxy for zone_id and would behave like a location dummy.
MODEL_INPUT_COLUMNS = [
    "zone_type",
    "month",
    "day_of_week",
    "hour",
    "is_weekend",
    "is_holiday",
    "is_working_day",
    "is_school_day",
    "time_period",
    *WEATHER_COLUMNS,
    *CONTEXT_COLUMNS,
    *EVENT_COLUMNS,
    *NETWORK_COLUMNS,
    *HISTORICAL_COLUMNS,
]

ENGINEERED_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "device_capacity_ratio",
    "session_device_ratio",
    "bandwidth_per_session",
    "network_stress_indicator",
    "recent_to_daily_ratio",
    "recent_to_weekly_ratio",
]

# Dropped after cyclical encodings are created (raw cyclic sources).
COLUMNS_DROPPED_AFTER_ENGINEERING = ["hour", "day_of_week", "month"]

CATEGORICAL_FEATURES = [
    "zone_type",
    "time_period",
    "weather_condition",
    "traffic_level",
    "event_type",
]

NUMERIC_FEATURES = [
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_weekend",
    "is_holiday",
    "is_working_day",
    "is_school_day",
    "temperature_c",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_kmh",
    "estimated_people_nearby",
    "public_transport_activity",
    "nearby_business_activity",
    "nearby_student_population",
    "nearby_worker_population",
    "event_nearby",
    "estimated_event_attendance",
    *NETWORK_COLUMNS,
    *HISTORICAL_COLUMNS,
    "device_capacity_ratio",
    "session_device_ratio",
    "bandwidth_per_session",
    "network_stress_indicator",
    "recent_to_daily_ratio",
    "recent_to_weekly_ratio",
]

CATEGORICAL_ALLOWED_VALUES = {
    "zone_type": {
        "DOWNTOWN",
        "UNIVERSITY",
        "COMMERCIAL",
        "PARK",
        "TRANSPORT",
        "RESIDENTIAL",
        "PUBLIC_SERVICE",
        "HEALTHCARE",
        "EDUCATION",
        "TOURISM",
    },
    "time_period": {"NIGHT", "MORNING", "AFTERNOON", "EVENING"},
    "weather_condition": {"CLEAR", "CLOUDY", "RAIN", "HEAVY_RAIN", "FOG"},
    "traffic_level": {"LOW", "MEDIUM", "HIGH"},
    "event_type": {
        "NONE",
        "ACADEMIC",
        "SPORT",
        "CULTURAL",
        "MUSIC",
        "COMMUNITY",
        "GOVERNMENT",
    },
    TARGET_COLUMN: {"LOW", "HIGH"},
}

# Columns where the generator injects missingness on purpose (not a contract failure).
EXPECTED_MISSING_COLUMNS = {
    "temperature_c",
    "humidity_percent",
    "precipitation_mm",
    "wind_speed_kmh",
    "weather_condition",
    "estimated_people_nearby",
    "traffic_level",
    "public_transport_activity",
    "nearby_business_activity",
    "nearby_student_population",
    "nearby_worker_population",
    "average_session_duration_min",
    "bandwidth_usage_mbps",
    "packet_loss_percent",
    "latency_ms",
    "signal_strength_dbm",
    "channel_utilization_percent",
    "network_uptime_percent",
    "connections_previous_hour",
    "connections_same_hour_previous_week",
    "average_connections_last_3_hours",
}

RANGE_CHECKS: dict[str, tuple[float, float]] = {
    "latitude": (5.45, 5.62),
    "longitude": (-73.45, -73.28),
    "altitude_m": (2600, 3000),
    "year": (2024, 2025),
    "month": (1, 12),
    "day": (1, 31),
    "day_of_week": (0, 6),
    "week_of_year": (1, 53),
    "hour": (0, 23),
    "minute": (0, 59),
    "temperature_c": (-5, 30),
    "humidity_percent": (0, 100),
    "precipitation_mm": (0, 80),
    "wind_speed_kmh": (0, 120),
    "access_point_capacity": (10, 500),
    "connected_devices": (0, 500),
    "active_sessions": (0, 500),
}

SYNTHETIC_DATA_DISCLAIMER = (
    "The dataset is synthetic and was generated for software development, "
    "machine learning experimentation, demonstration, and portfolio purposes."
)

DATASET_VERSION = "synthetic_seed_42_v1"
PREPROCESSING_VERSION = "scenario-features-v1"
INTERVAL_CONFIDENCE = 0.90
CLASSIFICATION_THRESHOLD = 0.50

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
