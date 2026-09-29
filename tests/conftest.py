"""Shared synthetic fixtures. Tests never require the 60k-row CSV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from wifi_tunja_smart_predictor.config import REQUIRED_COLUMNS, TARGET_COLUMN


def make_mini_frame(n_rows: int = 48) -> pd.DataFrame:
    """Tiny but schema-complete synthetic frame for unit tests."""
    rng = np.random.default_rng(42)
    start = pd.Timestamp("2024-06-01 00:00:00")
    timestamps = pd.date_range(start, periods=n_rows, freq="h")
    hours = timestamps.hour
    dow = timestamps.dayofweek
    data = {
        "timestamp": timestamps,
        "wifi_id": np.where(np.arange(n_rows) % 2 == 0, "WIFI_01", "WIFI_02"),
        "zone_id": "ZONE_01",
        "zone_name": "Synthetic Downtown Core",
        "zone_type": "DOWNTOWN",
        "latitude": 5.5353,
        "longitude": -73.3678,
        "altitude_m": 2782.0,
        "date": timestamps.normalize(),
        "year": timestamps.year,
        "month": timestamps.month,
        "day": timestamps.day,
        "day_of_week": dow,
        "day_name": timestamps.day_name().str.upper(),
        "week_of_year": timestamps.isocalendar().week.astype(int),
        "hour": hours,
        "minute": 0,
        "is_weekend": (dow >= 5).astype(int),
        "is_holiday": 0,
        "is_working_day": ((dow < 5)).astype(int),
        "is_school_day": ((dow < 5)).astype(int),
        "time_period": np.select(
            [hours < 6, hours < 12, hours < 18],
            ["NIGHT", "MORNING", "AFTERNOON"],
            default="EVENING",
        ),
        "temperature_c": rng.normal(14, 2, n_rows),
        "humidity_percent": rng.uniform(50, 90, n_rows),
        "precipitation_mm": rng.uniform(0, 2, n_rows),
        "wind_speed_kmh": rng.uniform(2, 15, n_rows),
        "weather_condition": rng.choice(["CLEAR", "CLOUDY", "RAIN"], n_rows),
        "estimated_people_nearby": rng.integers(50, 800, n_rows),
        "traffic_level": rng.choice(["LOW", "MEDIUM", "HIGH"], n_rows),
        "public_transport_activity": rng.uniform(10, 80, n_rows),
        "nearby_business_activity": rng.uniform(10, 80, n_rows),
        "nearby_student_population": rng.integers(20, 400, n_rows),
        "nearby_worker_population": rng.integers(20, 400, n_rows),
        "event_nearby": rng.integers(0, 2, n_rows),
        "event_type": rng.choice(["NONE", "CULTURAL", "MUSIC"], n_rows),
        "estimated_event_attendance": rng.integers(0, 200, n_rows),
        "access_point_capacity": 150,
        "connected_devices": rng.integers(10, 120, n_rows),
        "active_sessions": rng.integers(5, 80, n_rows),
        "average_session_duration_min": rng.uniform(8, 40, n_rows),
        "bandwidth_usage_mbps": rng.uniform(20, 200, n_rows),
        "packet_loss_percent": rng.uniform(0, 5, n_rows),
        "latency_ms": rng.uniform(10, 80, n_rows),
        "signal_strength_dbm": rng.uniform(-75, -45, n_rows),
        "channel_utilization_percent": rng.uniform(10, 80, n_rows),
        "network_uptime_percent": rng.uniform(95, 100, n_rows),
        "connections_previous_hour": rng.integers(20, 200, n_rows),
        "connections_previous_day": rng.uniform(20, 200, n_rows),
        "connections_same_hour_previous_day": rng.integers(20, 200, n_rows),
        "connections_same_hour_previous_week": rng.integers(20, 200, n_rows),
        "average_connections_last_3_hours": rng.uniform(20, 200, n_rows),
        "average_connections_last_24_hours": rng.uniform(20, 200, n_rows),
        "average_connections_last_7_days": rng.uniform(20, 200, n_rows),
        "connections_next_hour": rng.integers(20, 200, n_rows),
        TARGET_COLUMN: np.where(np.arange(n_rows) < n_rows // 2, "LOW", "HIGH"),
    }
    frame = pd.DataFrame(data)
    # Keep sessions <= devices for a realistic snapshot.
    frame["active_sessions"] = np.minimum(frame["active_sessions"], frame["connected_devices"])
    assert list(frame.columns) == REQUIRED_COLUMNS or set(REQUIRED_COLUMNS) <= set(frame.columns)
    return frame[REQUIRED_COLUMNS]


@pytest.fixture
def mini_frame() -> pd.DataFrame:
    return make_mini_frame()
