#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WiFi Tunja Smart Predictor - Synthetic Dataset Generator
=========================================================

Synthetic dataset generated for the WiFi Tunja Smart Predictor project.

IMPORTANT DISCLAIMER
--------------------
Everything produced by this script is SIMULATED. The WiFi access points, their
coordinates, the zones, the connection counts, the weather, the events and the
network metrics are NOT real measurements and do NOT describe any real public
or commercial WiFi infrastructure in Tunja, Colombia. The holiday calendar is
a documented approximation of Colombian public holidays, not an official
historical record.

Usage (from the repository root)::

    python scripts/generate_synthetic_dataset.py

Output::

    data/raw/wifi_tunja_public.csv

Generation strategy (short version)
-----------------------------------
1. A full HOURLY simulation is run for every synthetic access point (AP) over
   the export period plus a short warm-up period (the warm-up is never exported;
   it only exists so that lag features are valid on the first exported day).
2. Hourly connection counts come from a latent demand process: zone-type daily
   profiles x calendar effects x weather x events x persistent AR(1) shocks,
   drawn from a Negative Binomial (over-dispersed counts).
3. The exported CSV is a reproducible random sample of AP-hour observations
   taken from that full hourly simulation. Because history features are
   computed on the FULL hourly series *before* sampling, they stay exact.
4. Timestamp semantics: a row's ``timestamp`` is the PREDICTION TIME, i.e. the
   start of the hour being forecast. ``connections_next_hour`` is the number of
   connections in [timestamp, timestamp + 1h). Every feature is computed from
   information available at or before ``timestamp`` (see "Temporal leakage").
5. Some realistic data-quality issues (missing values, a few duplicates) are
   injected at the very end, only into non-target, non-identifier columns.

Temporal leakage
----------------
* History features only use hours strictly before ``timestamp``.
* Network snapshot metrics (connected devices, latency, ...) are derived from
  the hour that just ended, never from the hour being predicted.
* The latent demand score is never exported; only observable variables are.
* ``validate_dataset`` verifies lag features against the target values of
  earlier rows, and rejects any known "leaky" column name.

Do not use ``connections_next_hour`` or ``demand_level`` as predictors.
"""

# ---------------------------------------------------------------------------
# 1. IMPORTS
# ---------------------------------------------------------------------------
from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------------------------
DISCLAIMER = "Synthetic dataset generated for the WiFi Tunja Smart Predictor project."

RANDOM_SEED = 42

START_DATE = "2024-01-01"  # first exported day
END_DATE = "2025-12-31"  # last exported day (inclusive)
WARMUP_DAYS = 14  # simulated but NOT exported (valid lag history)
N_ROWS = 60_000  # final number of rows in the CSV (duplicates included)
N_DUPLICATE_ROWS = 60  # exact duplicate rows injected at the end
MIN_ROWS = 50_000
MAX_ROWS = 100_000

HIGH_DEMAND_SHARE = 0.38  # target share of HIGH demand (calibrates the threshold)
LOW_SHARE_RANGE = (0.55, 0.70)  # accepted LOW share after generation
HIGH_SHARE_RANGE = (0.30, 0.45)  # accepted HIGH share after generation

ANNUAL_GROWTH = 0.08  # slow usage growth per year
DISPERSION = 14.0  # Negative Binomial dispersion (higher = less noisy)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "wifi_tunja_public.csv"

# Rough bounding box of the simulated urban area (approximately Tunja).
LAT_BOUNDS = (5.500, 5.575)
LON_BOUNDS = (-73.400, -73.330)
CITY_CENTER_ALTITUDE_M = 2782.0

EVENT_TYPES = ["NONE", "ACADEMIC", "SPORT", "CULTURAL", "MUSIC", "COMMUNITY", "GOVERNMENT"]
WEATHER_CONDITIONS = ["CLEAR", "CLOUDY", "RAIN", "HEAVY_RAIN", "FOG"]
TRAFFIC_LEVELS = ["LOW", "MEDIUM", "HIGH"]
# Effective "wetness" per weather code (CLEAR, CLOUDY, RAIN, HEAVY_RAIN, FOG)
RAIN_FACTOR = np.array([0.0, 0.0, 1.0, 2.0, 0.3])
HEAVY_RAIN_MM = 3.0  # mm/hour threshold for HEAVY_RAIN
RAIN_MM = 0.2  # mm/hour threshold for RAIN


class ZoneSpec(NamedTuple):
    """Definition of one synthetic zone."""

    zone_id: str
    zone_name: str
    zone_type: str
    latitude: float
    longitude: float
    n_aps: int


# 12 synthetic zones, 30 synthetic access points in total.
# Names are deliberately generic: they do NOT refer to real facilities.
ZONES: list[ZoneSpec] = [
    ZoneSpec("ZONE_01", "Synthetic Downtown Core", "DOWNTOWN", 5.5353, -73.3678, 3),
    ZoneSpec("ZONE_02", "Synthetic Campus North", "UNIVERSITY", 5.5590, -73.3555, 3),
    ZoneSpec("ZONE_03", "Synthetic Campus South", "UNIVERSITY", 5.5205, -73.3805, 2),
    ZoneSpec("ZONE_04", "Synthetic Retail Avenue", "COMMERCIAL", 5.5440, -73.3610, 3),
    ZoneSpec("ZONE_05", "Synthetic Shopping Plaza", "COMMERCIAL", 5.5300, -73.3540, 2),
    ZoneSpec("ZONE_06", "Synthetic Central Green Area", "PARK", 5.5480, -73.3720, 3),
    ZoneSpec("ZONE_07", "Synthetic Bus Terminal Area", "TRANSPORT", 5.5400, -73.3455, 3),
    ZoneSpec("ZONE_08", "Synthetic Residential West", "RESIDENTIAL", 5.5250, -73.3830, 2),
    ZoneSpec("ZONE_09", "Synthetic Civic Center", "PUBLIC_SERVICE", 5.5340, -73.3640, 2),
    ZoneSpec("ZONE_10", "Synthetic Medical District", "HEALTHCARE", 5.5190, -73.3610, 2),
    ZoneSpec("ZONE_11", "Synthetic School Cluster", "EDUCATION", 5.5510, -73.3690, 2),
    ZoneSpec("ZONE_12", "Synthetic Historic Quarter", "TOURISM", 5.5375, -73.3705, 3),
]

# Static parameters per zone type.
#   capacity : possible AP capacities (max concurrent devices)
#   outdoor  : exposure to weather (1 = fully outdoor)
#   people   : typical people nearby at peak
#   students / workers : typical nearby populations at peak
#   business / transit : 0-1 intensity of business / public-transport activity
#   holiday  : demand multiplier on public holidays
#   vacation : demand multiplier on working days outside the academic terms
#   traffic  : baseline traffic pressure (0-1)
#   session  : typical session duration in minutes
ZONE_TYPE_PARAMS: dict[str, dict] = {
    "UNIVERSITY": dict(
        capacity=(150, 200, 250),
        outdoor=0.30,
        people=1500,
        students=2500,
        workers=250,
        business=0.35,
        transit=0.40,
        holiday=0.35,
        vacation=0.30,
        traffic=0.50,
        session=28,
    ),
    "DOWNTOWN": dict(
        capacity=(150, 200),
        outdoor=0.40,
        people=1200,
        students=200,
        workers=900,
        business=0.85,
        transit=0.55,
        holiday=0.75,
        vacation=1.0,
        traffic=0.80,
        session=18,
    ),
    "COMMERCIAL": dict(
        capacity=(100, 150, 200),
        outdoor=0.20,
        people=1000,
        students=100,
        workers=600,
        business=0.95,
        transit=0.40,
        holiday=1.05,
        vacation=1.0,
        traffic=0.70,
        session=20,
    ),
    "PARK": dict(
        capacity=(50, 75, 100),
        outdoor=1.00,
        people=600,
        students=60,
        workers=40,
        business=0.15,
        transit=0.15,
        holiday=1.35,
        vacation=1.0,
        traffic=0.25,
        session=22,
    ),
    "TRANSPORT": dict(
        capacity=(150, 200),
        outdoor=0.50,
        people=1500,
        students=300,
        workers=300,
        business=0.45,
        transit=0.95,
        holiday=0.70,
        vacation=1.0,
        traffic=0.90,
        session=10,
    ),
    "RESIDENTIAL": dict(
        capacity=(50, 75),
        outdoor=0.10,
        people=500,
        students=100,
        workers=100,
        business=0.10,
        transit=0.20,
        holiday=1.05,
        vacation=1.0,
        traffic=0.30,
        session=35,
    ),
    "PUBLIC_SERVICE": dict(
        capacity=(50, 75, 100),
        outdoor=0.10,
        people=300,
        students=30,
        workers=700,
        business=0.30,
        transit=0.25,
        holiday=0.15,
        vacation=1.0,
        traffic=0.50,
        session=15,
    ),
    "HEALTHCARE": dict(
        capacity=(50, 75, 100),
        outdoor=0.05,
        people=400,
        students=40,
        workers=500,
        business=0.30,
        transit=0.30,
        holiday=0.85,
        vacation=1.0,
        traffic=0.50,
        session=30,
    ),
    "EDUCATION": dict(
        capacity=(75, 100),
        outdoor=0.20,
        people=700,
        students=900,
        workers=120,
        business=0.15,
        transit=0.25,
        holiday=0.20,
        vacation=0.15,
        traffic=0.55,
        session=25,
    ),
    "TOURISM": dict(
        capacity=(75, 100, 150),
        outdoor=0.70,
        people=700,
        students=30,
        workers=250,
        business=0.50,
        transit=0.25,
        holiday=1.50,
        vacation=1.0,
        traffic=0.45,
        session=20,
    ),
}

# Hourly demand profiles per zone type: (baseline, [(center_hour, width_hours, amplitude), ...]).
HOURLY_PROFILES: dict[str, dict[str, tuple[float, list[tuple[float, float, float]]]]] = {
    "UNIVERSITY": {
        "weekday": (0.05, [(10.0, 2.2, 1.00), (15.0, 2.5, 0.90), (20.0, 2.0, 0.25)]),
        "weekend": (0.06, [(14.0, 4.0, 0.25)]),
    },
    "DOWNTOWN": {
        "weekday": (0.08, [(10.5, 2.5, 0.80), (13.0, 2.0, 0.60), (17.0, 2.5, 0.70)]),
        "weekend": (0.08, [(12.0, 3.5, 0.70), (16.0, 3.0, 0.60)]),
    },
    "COMMERCIAL": {
        "weekday": (0.06, [(12.0, 3.0, 0.60), (16.5, 3.0, 0.70), (19.0, 2.0, 0.30)]),
        "weekend": (0.06, [(13.0, 3.5, 0.90), (17.0, 3.0, 1.00)]),
    },
    "PARK": {
        "weekday": (0.04, [(16.5, 2.5, 0.40), (7.0, 1.5, 0.15)]),
        "weekend": (0.05, [(15.0, 3.5, 1.00), (11.0, 2.0, 0.40)]),
    },
    "TRANSPORT": {
        "weekday": (0.06, [(6.8, 1.2, 1.00), (12.5, 2.0, 0.35), (17.8, 1.5, 1.00)]),
        "weekend": (0.05, [(9.0, 2.5, 0.35), (17.0, 2.5, 0.45)]),
    },
    "RESIDENTIAL": {
        "weekday": (0.08, [(20.0, 2.5, 0.90), (7.0, 1.5, 0.15), (13.0, 2.0, 0.20)]),
        "weekend": (0.10, [(15.0, 3.5, 0.40), (20.5, 2.5, 0.80)]),
    },
    "PUBLIC_SERVICE": {
        "weekday": (0.03, [(9.5, 2.0, 0.90), (14.0, 1.5, 0.50)]),
        "weekend": (0.02, [(10.0, 2.0, 0.10)]),
    },
    "HEALTHCARE": {
        "weekday": (0.10, [(9.0, 2.5, 0.70), (15.0, 3.0, 0.50)]),
        "weekend": (0.10, [(11.0, 3.0, 0.40)]),
    },
    "EDUCATION": {
        "weekday": (0.03, [(7.5, 1.0, 0.70), (12.5, 1.5, 0.60), (15.5, 1.5, 0.50)]),
        "weekend": (0.02, [(10.0, 3.0, 0.08)]),
    },
    "TOURISM": {
        "weekday": (0.05, [(12.0, 3.5, 0.60), (16.0, 3.0, 0.60)]),
        "weekend": (0.06, [(12.5, 3.5, 1.00), (16.5, 3.0, 1.00)]),
    },
}

# Monthly seasonal multipliers (January .. December). Default is flat.
DEFAULT_MONTH_FACTOR = np.ones(12)
MONTH_FACTORS: dict[str, np.ndarray] = {
    "TOURISM": np.array([1.25, 1.00, 1.10, 1.15, 0.90, 1.15, 1.30, 1.10, 0.90, 1.00, 0.95, 1.30]),
    "COMMERCIAL": np.array(
        [1.00, 0.95, 0.98, 1.00, 1.02, 1.00, 1.03, 1.00, 0.98, 1.00, 1.10, 1.28]
    ),
    "PARK": np.array([1.05, 1.05, 1.00, 0.92, 0.92, 1.00, 1.08, 1.08, 1.00, 0.92, 0.92, 1.02]),
}
DOW_FACTOR = np.array([0.97, 1.00, 1.00, 1.02, 1.05, 1.00, 0.94])  # Monday..Sunday

# Approximation of Colombian public holidays (Ley Emiliani style Monday moves).
# This is a documented APPROXIMATION for simulation purposes, NOT an official record.
HOLIDAYS_ISO = [
    # warm-up period (December 2023)
    "2023-12-08",
    "2023-12-25",
    # 2024
    "2024-01-01",
    "2024-01-08",
    "2024-03-25",
    "2024-03-28",
    "2024-03-29",
    "2024-05-01",
    "2024-05-13",
    "2024-06-03",
    "2024-06-10",
    "2024-07-01",
    "2024-07-20",
    "2024-08-07",
    "2024-08-19",
    "2024-10-14",
    "2024-11-04",
    "2024-11-11",
    "2024-12-08",
    "2024-12-25",
    # 2025
    "2025-01-01",
    "2025-01-06",
    "2025-03-24",
    "2025-04-17",
    "2025-04-18",
    "2025-05-01",
    "2025-06-02",
    "2025-06-23",
    "2025-06-30",
    "2025-07-20",
    "2025-08-07",
    "2025-08-18",
    "2025-10-13",
    "2025-11-03",
    "2025-11-17",
    "2025-12-08",
    "2025-12-25",
]
HOLIDAY_DATES = pd.DatetimeIndex(pd.to_datetime(HOLIDAYS_ISO))

# Synthetic academic terms (school/university in session).
ACADEMIC_TERMS = [
    ("2024-01-22", "2024-06-07"),
    ("2024-08-05", "2024-11-29"),
    ("2025-01-20", "2025-06-06"),
    ("2025-08-04", "2025-11-28"),
]

# Weather climatology (Andean highland city: bimodal rainy seasons).
RAINY_DAY_PROB = np.array([0.25, 0.30, 0.45, 0.65, 0.65, 0.50, 0.40, 0.40, 0.50, 0.70, 0.70, 0.35])
WIND_MONTH_FACTOR = np.array([1.0, 1.0, 0.9, 0.9, 0.9, 1.2, 1.5, 1.5, 1.2, 0.9, 0.9, 1.0])

# Event configuration.
EVENT_CODE = {name: idx for idx, name in enumerate(EVENT_TYPES)}
EVENT_RATES: dict[str, tuple[float, float]] = {  # (weekday, weekend/holiday) daily probability
    "UNIVERSITY": (0.10, 0.04),
    "EDUCATION": (0.08, 0.02),
    "DOWNTOWN": (0.08, 0.15),
    "COMMERCIAL": (0.05, 0.12),
    "PARK": (0.05, 0.20),
    "TRANSPORT": (0.02, 0.03),
    "RESIDENTIAL": (0.02, 0.06),
    "PUBLIC_SERVICE": (0.06, 0.01),
    "HEALTHCARE": (0.01, 0.005),
    "TOURISM": (0.06, 0.20),
}
EVENT_TYPE_WEIGHTS: dict[str, dict[str, float]] = {
    "UNIVERSITY": {"ACADEMIC": 0.60, "CULTURAL": 0.20, "SPORT": 0.15, "COMMUNITY": 0.05},
    "EDUCATION": {"ACADEMIC": 0.70, "SPORT": 0.20, "CULTURAL": 0.10},
    "DOWNTOWN": {
        "GOVERNMENT": 0.25,
        "CULTURAL": 0.30,
        "MUSIC": 0.20,
        "COMMUNITY": 0.20,
        "SPORT": 0.05,
    },
    "COMMERCIAL": {
        "MUSIC": 0.25,
        "COMMUNITY": 0.35,
        "CULTURAL": 0.25,
        "SPORT": 0.05,
        "GOVERNMENT": 0.10,
    },
    "PARK": {"SPORT": 0.35, "MUSIC": 0.25, "COMMUNITY": 0.25, "CULTURAL": 0.15},
    "TRANSPORT": {"COMMUNITY": 0.40, "GOVERNMENT": 0.30, "CULTURAL": 0.30},
    "RESIDENTIAL": {"COMMUNITY": 0.60, "SPORT": 0.20, "MUSIC": 0.20},
    "PUBLIC_SERVICE": {"GOVERNMENT": 0.70, "COMMUNITY": 0.30},
    "HEALTHCARE": {"COMMUNITY": 0.60, "GOVERNMENT": 0.40},
    "TOURISM": {"CULTURAL": 0.40, "MUSIC": 0.30, "COMMUNITY": 0.20, "GOVERNMENT": 0.10},
}
EVENT_START_HOURS = {
    "ACADEMIC": (8, 15),
    "SPORT": (9, 18),
    "CULTURAL": (15, 19),
    "MUSIC": (18, 20),
    "COMMUNITY": (10, 16),
    "GOVERNMENT": (9, 13),
}
EVENT_DURATION_HOURS = {
    "ACADEMIC": (2, 5),
    "SPORT": (2, 3),
    "CULTURAL": (2, 4),
    "MUSIC": (3, 5),
    "COMMUNITY": (2, 5),
    "GOVERNMENT": (2, 4),
}
EVENT_MEAN_ATTENDANCE = {
    "ACADEMIC": 250,
    "SPORT": 600,
    "CULTURAL": 500,
    "MUSIC": 900,
    "COMMUNITY": 300,
    "GOVERNMENT": 350,
}
EVENT_RATE_SCALE = 2.5  # global multiplier so events appear in a few % of observations
EVENT_MONTH_BOOST = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.4, 1.8, 1.4, 1.0, 1.5])
FESTIVE_ZONE_TYPES = {"TOURISM", "PARK", "DOWNTOWN"}
MAX_EVENT_ATTENDANCE = 8_000

# Final CSV column order (55 columns).
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
EVENT_COLUMNS = ["event_nearby", "event_type", "estimated_event_attendance"]
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
TARGET_COLUMNS = ["connections_next_hour", "demand_level"]
FINAL_COLUMNS = (
    IDENTIFIER_COLUMNS
    + TEMPORAL_COLUMNS
    + WEATHER_COLUMNS
    + CONTEXT_COLUMNS
    + EVENT_COLUMNS
    + NETWORK_COLUMNS
    + HISTORICAL_COLUMNS
    + TARGET_COLUMNS
)

# Column names that would indicate target leakage. None of them may ever exist.
FORBIDDEN_COLUMNS = [
    "target_copy",
    "target_encoded",
    "demand_score",
    "future_connections",
    "future_demand",
    "target_probability",
    "target_numeric",
    "target_as_feature",
    "latent_demand",
    "latent_score",
]

# Missing-value injection: column -> probability (kept within 1%-3%).
MISSING_RATES: dict[str, float] = {
    "temperature_c": 0.015,
    "humidity_percent": 0.020,
    "precipitation_mm": 0.015,
    "wind_speed_kmh": 0.025,
    "weather_condition": 0.010,
    "estimated_people_nearby": 0.020,
    "traffic_level": 0.015,
    "public_transport_activity": 0.030,
    "nearby_business_activity": 0.020,
    "nearby_student_population": 0.010,
    "nearby_worker_population": 0.010,
    "average_session_duration_min": 0.020,
    "bandwidth_usage_mbps": 0.015,
    "packet_loss_percent": 0.030,
    "latency_ms": 0.020,
    "signal_strength_dbm": 0.025,
    "channel_utilization_percent": 0.015,
    "network_uptime_percent": 0.010,
    "connections_previous_hour": 0.010,
    "connections_same_hour_previous_week": 0.020,
    "average_connections_last_3_hours": 0.010,
}

# Sensible ranges used by the validation step (inclusive).
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
    "is_weekend": (0, 1),
    "is_holiday": (0, 1),
    "is_working_day": (0, 1),
    "is_school_day": (0, 1),
    "temperature_c": (-5, 30),
    "humidity_percent": (0, 100),
    "precipitation_mm": (0, 80),
    "wind_speed_kmh": (0, 120),
    "estimated_people_nearby": (0, 10_000),
    "public_transport_activity": (0, 100),
    "nearby_business_activity": (0, 100),
    "nearby_student_population": (0, 10_000),
    "nearby_worker_population": (0, 10_000),
    "event_nearby": (0, 1),
    "estimated_event_attendance": (0, 10_000),
    "access_point_capacity": (10, 500),
    "connected_devices": (0, 500),
    "active_sessions": (0, 500),
    "average_session_duration_min": (0, 240),
    "bandwidth_usage_mbps": (0, 1000),
    "packet_loss_percent": (0, 100),
    "latency_ms": (0, 5000),
    "signal_strength_dbm": (-100, -20),
    "channel_utilization_percent": (0, 100),
    "network_uptime_percent": (0, 100),
    "connections_previous_hour": (0, 1500),
    "connections_previous_day": (0, 1500),
    "connections_same_hour_previous_day": (0, 1500),
    "connections_same_hour_previous_week": (0, 1500),
    "average_connections_last_3_hours": (0, 1500),
    "average_connections_last_24_hours": (0, 1500),
    "average_connections_last_7_days": (0, 1500),
    "connections_next_hour": (0, 1500),
}


class DatasetValidationError(RuntimeError):
    """Raised when the generated dataset fails a critical validation check."""


# ---------------------------------------------------------------------------
# 3. RANDOM GENERATOR
# ---------------------------------------------------------------------------
def make_rng(seed: int = RANDOM_SEED) -> np.random.Generator:
    """Return the single NumPy Generator used for the whole simulation.

    A single, sequentially consumed generator makes the output fully
    reproducible for a given Python/NumPy/pandas environment.
    """
    return np.random.default_rng(seed)


def _bump(hours: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    """Circular (24h) Gaussian bump used to build daily demand profiles."""
    delta = (hours - center + 12.0) % 24.0 - 12.0
    return amplitude * np.exp(-0.5 * (delta / width) ** 2)


def ar1_process(
    n_steps: int, n_series: int, phi: float, stationary_sd: float, rng: np.random.Generator
) -> np.ndarray:
    """Simulate independent zero-mean AR(1) series with a given stationary std."""
    innovation_sd = stationary_sd * np.sqrt(1.0 - phi**2)
    out = np.empty((n_steps, n_series))
    out[0] = rng.normal(0.0, stationary_sd, n_series)
    noise = rng.normal(0.0, innovation_sd, (n_steps, n_series))
    for step in range(1, n_steps):
        out[step] = phi * out[step - 1] + noise[step]
    return out


# ---------------------------------------------------------------------------
# 4. SYNTHETIC WIFI LOCATION GENERATION
# ---------------------------------------------------------------------------
def generate_access_points(rng: np.random.Generator) -> pd.DataFrame:
    """Create ~30 SYNTHETIC access points spread over the synthetic zones.

    Coordinates are simulated around each zone centre inside a small bounding
    box around Tunja. They do NOT represent any real access point.
    """
    rows: list[dict] = []
    counter = 1
    for zone_idx, zone in enumerate(ZONES):
        params = ZONE_TYPE_PARAMS[zone.zone_type]
        for _ in range(zone.n_aps):
            lat = float(np.clip(zone.latitude + rng.normal(0, 0.0012), *LAT_BOUNDS))
            lon = float(np.clip(zone.longitude + rng.normal(0, 0.0012), *LON_BOUNDS))
            # Rough altitude model: mildly rising to the east/north + local noise.
            altitude = (
                CITY_CENTER_ALTITUDE_M
                - 2.0
                + 1500.0 * (lon + 73.3678)
                + 400.0 * (lat - 5.5353)
                + rng.normal(0, 10)
            )
            rows.append(
                {
                    "wifi_id": f"WIFI_{counter:03d}",
                    "zone_id": zone.zone_id,
                    "zone_name": zone.zone_name,
                    "zone_type": zone.zone_type,
                    "zone_idx": zone_idx,
                    "latitude": round(lat, 6),
                    "longitude": round(lon, 6),
                    "altitude_m": round(float(np.clip(altitude, 2700, 2900)), 1),
                    "access_point_capacity": int(rng.choice(params["capacity"])),
                    "site_factor": float(rng.lognormal(0.0, 0.20)),  # site attractiveness
                    "event_proximity": float(rng.uniform(0.35, 1.0)),  # closeness to event venues
                    "people_base": params["people"] * float(rng.lognormal(0.0, 0.25)),
                    "students_base": params["students"] * float(rng.lognormal(0.0, 0.30)),
                    "workers_base": params["workers"] * float(rng.lognormal(0.0, 0.30)),
                    "business_level": float(
                        np.clip(params["business"] * rng.lognormal(0.0, 0.15), 0, 1)
                    ),
                    "transit_level": float(
                        np.clip(params["transit"] * rng.lognormal(0.0, 0.15), 0, 1)
                    ),
                    "outdoor": params["outdoor"],
                    "traffic_base": params["traffic"],
                    "session_base": params["session"],
                    "signal_offset": float(rng.normal(-58.0, 4.0)),
                }
            )
            counter += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. TIMESTAMP GENERATION
# ---------------------------------------------------------------------------
def make_hourly_index() -> pd.DatetimeIndex:
    """Contiguous hourly index: warm-up days + the exported period."""
    sim_start = pd.Timestamp(START_DATE) - pd.Timedelta(days=WARMUP_DAYS)
    n_days = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days + 1 + WARMUP_DAYS
    return pd.date_range(start=sim_start, periods=n_days * 24, freq=pd.Timedelta(hours=1))


# ---------------------------------------------------------------------------
# 6. TEMPORAL FEATURE GENERATION
# ---------------------------------------------------------------------------
def build_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Derive EVERY temporal field from the timestamp (never generated independently)."""
    cal = pd.DataFrame({"timestamp": index})
    ts = cal["timestamp"].dt
    day_start = ts.normalize()

    cal["date"] = ts.strftime("%Y-%m-%d")
    cal["year"] = ts.year
    cal["month"] = ts.month
    cal["day"] = ts.day
    cal["day_of_week"] = ts.dayofweek  # Monday = 0 ... Sunday = 6
    cal["day_name"] = ts.day_name()
    cal["week_of_year"] = ts.isocalendar().week.astype(int)
    cal["hour"] = ts.hour
    cal["minute"] = ts.minute
    cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(int)
    cal["is_holiday"] = day_start.isin(HOLIDAY_DATES).astype(int)
    cal["is_working_day"] = ((cal["is_weekend"] == 0) & (cal["is_holiday"] == 0)).astype(int)

    in_term = np.zeros(len(cal), dtype=bool)
    for term_start, term_end in ACADEMIC_TERMS:
        in_term |= (
            (day_start >= pd.Timestamp(term_start)) & (day_start <= pd.Timestamp(term_end))
        ).to_numpy()
    cal["is_school_day"] = ((cal["is_working_day"] == 1) & in_term).astype(int)

    # NIGHT 00-05, MORNING 06-11, AFTERNOON 12-17, EVENING 18-23
    cal["time_period"] = np.select(
        [cal["hour"] < 6, cal["hour"] < 12, cal["hour"] < 18],
        ["NIGHT", "MORNING", "AFTERNOON"],
        default="EVENING",
    )
    return cal


def build_profile_table() -> dict[str, np.ndarray]:
    """Return, for each zone type, a (2, 24) array: [weekday, weekend/holiday] x hour."""
    hours = np.arange(24, dtype=float)
    table: dict[str, np.ndarray] = {}
    for zone_type, spec in HOURLY_PROFILES.items():
        rows = []
        for day_kind in ("weekday", "weekend"):
            base, bumps = spec[day_kind]
            rows.append(base + sum(_bump(hours, c, w, a) for c, w, a in bumps))
        table[zone_type] = np.vstack(rows)
    return table


def compute_activity(cal: pd.DataFrame, aps: pd.DataFrame) -> np.ndarray:
    """Latent 'activity' level per hour and AP: zone profile x calendar/season effects.

    Shape: (n_hours, n_aps). This is an INTERNAL quantity; it is never exported.
    """
    profile_table = build_profile_table()
    hour = cal["hour"].to_numpy()
    off_day = ((cal["is_weekend"] | cal["is_holiday"]).to_numpy()).astype(int)
    holiday = cal["is_holiday"].to_numpy().astype(bool)
    vacation = (cal["is_working_day"].to_numpy() == 1) & (cal["is_school_day"].to_numpy() == 0)
    month_idx = cal["month"].to_numpy() - 1
    dow = cal["day_of_week"].to_numpy()
    days_since_start = (cal["timestamp"] - pd.Timestamp(START_DATE)).dt.days.to_numpy()
    trend = 1.0 + ANNUAL_GROWTH * np.clip(days_since_start, 0, None) / 365.0

    activity = np.empty((len(cal), len(aps)))
    for j, ap in enumerate(aps.itertuples()):
        params = ZONE_TYPE_PARAMS[ap.zone_type]
        curve = profile_table[ap.zone_type][off_day, hour]
        multiplier = np.where(holiday, params["holiday"], 1.0)
        if params["vacation"] < 1.0:  # universities / schools outside academic terms
            multiplier = multiplier * np.where(vacation, params["vacation"], 1.0)
        multiplier = (
            multiplier
            * MONTH_FACTORS.get(ap.zone_type, DEFAULT_MONTH_FACTOR)[month_idx]
            * DOW_FACTOR[dow]
            * trend
        )
        activity[:, j] = curve * multiplier
    return activity


# ---------------------------------------------------------------------------
# 8. WEATHER GENERATION  (defined before context: context depends on it)
# ---------------------------------------------------------------------------
def generate_weather(
    cal: pd.DataFrame, aps: pd.DataFrame, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Simulate weather with plausible relationships (city-wide regime + local noise)."""
    n_hours, n_aps = len(cal), len(aps)
    n_days = n_hours // 24
    hour = cal["hour"].to_numpy()
    month_idx = cal["month"].to_numpy() - 1
    day_idx = np.arange(n_hours) // 24
    day_of_year = cal["timestamp"].dt.dayofyear.to_numpy()
    altitude = aps["altitude_m"].to_numpy()

    # Day-level regime: persistent rainy / dry spells.
    p_day = RAINY_DAY_PROB[month_idx[::24]]
    logit = np.log(p_day / (1 - p_day)) + 0.9 * ar1_process(n_days, 1, 0.6, 1.0, rng)[:, 0]
    rainy_day = rng.random(n_days) < 1.0 / (1.0 + np.exp(-logit))

    # Hourly rain: afternoon-heavy, gamma intensity, occasional heavy bursts, light drizzle.
    p_hour = 0.05 + 0.22 * _bump(hour, 16.0, 3.5, 1.0) + 0.07 * _bump(hour, 3.0, 3.0, 1.0)
    wet_city = rainy_day[day_idx] & (rng.random(n_hours) < p_hour)
    intensity = rng.gamma(1.1, 0.9, n_hours) * np.where(rng.random(n_hours) < 0.15, 3.5, 1.0)
    drizzle = (~rainy_day[day_idx]) & (rng.random(n_hours) < 0.04)
    precip_city = np.where(wet_city, intensity, 0.0) + np.where(
        drizzle, rng.uniform(0.1, 0.5, n_hours), 0.0
    )

    # Local precipitation: patchy (a few APs stay dry) and locally scaled.
    patchy = rng.random((n_hours, n_aps)) > 0.10
    precipitation = np.where(
        patchy, precip_city[:, None] * rng.lognormal(-0.05, 0.35, (n_hours, n_aps)), 0.0
    )
    precipitation = np.round(np.clip(precipitation, 0, 80), 2)

    # Temperature: diurnal + seasonal + day anomaly + rain cooling + lapse rate with altitude.
    diurnal = 4.0 * np.cos(2 * np.pi * (hour - 15) / 24)
    seasonal = -0.7 * np.cos(2 * np.pi * (day_of_year - 200) / 365)
    anomaly = ar1_process(n_days, 1, 0.7, 1.2, rng)[:, 0][day_idx]
    temp_city = 13.0 + diurnal + seasonal + anomaly - 1.5 * rainy_day[day_idx]
    temperature = (
        temp_city[:, None]
        - 6.5 * (altitude[None, :] - CITY_CENTER_ALTITUDE_M) / 1000.0
        + rng.normal(0, 0.5, (n_hours, n_aps))
    )
    temperature = np.clip(temperature, 2.0, 26.0)

    # Humidity: colder and rainier -> more humid.
    humidity = (
        80.0
        - 2.3 * (temperature - 13.0)
        + 7.0 * (precipitation > 0.1)
        + 3.0 * rainy_day[day_idx][:, None]
        + rng.normal(0, 4.0, (n_hours, n_aps))
    )
    humidity = np.clip(humidity, 35.0, 100.0)

    # Wind: gamma-distributed, windier in the afternoon, in Jul-Aug and during rain.
    wind_city = (
        rng.gamma(2.0, 4.0, n_hours)
        * (1 + 0.5 * _bump(hour, 15.0, 4.0, 1.0))
        * WIND_MONTH_FACTOR[month_idx]
        * (1 + 0.2 * (precip_city > 0))
    )
    wind = np.clip(wind_city[:, None] * rng.lognormal(0, 0.2, (n_hours, n_aps)), 0, 60)

    # Condition codes: 0 CLEAR, 1 CLOUDY, 2 RAIN, 3 HEAVY_RAIN, 4 FOG
    cloudy_city = rng.random(n_hours) < (0.30 + 0.40 * rainy_day[day_idx])
    codes = np.repeat(cloudy_city[:, None].astype(np.int8), n_aps, axis=1)
    foggy = (
        ((hour >= 4) & (hour <= 9))[:, None]
        & (humidity > 88)
        & (precipitation < 0.1)
        & (rng.random((n_hours, n_aps)) < 0.55)
    )
    codes[foggy] = 4
    codes[precipitation >= RAIN_MM] = 2
    codes[precipitation >= HEAVY_RAIN_MM] = 3

    return {
        "temperature_c": np.round(temperature, 1),
        "humidity_percent": np.round(humidity, 1),
        "precipitation_mm": precipitation,
        "wind_speed_kmh": np.round(wind, 1),
        "condition_code": codes,
    }


# ---------------------------------------------------------------------------
# 9. EVENT GENERATION
# ---------------------------------------------------------------------------
def generate_events(
    cal: pd.DataFrame, aps: pd.DataFrame, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Generate zone-level events (type, timing, attendance) and map them to APs.

    Event frequency and type depend on the zone type (e.g. academic events near
    education zones, sport/music in parks), weekends, holidays and season.
    """
    n_hours = len(cal)
    n_days = n_hours // 24
    day_cal = cal.iloc[::24].reset_index(drop=True)
    off_day = (day_cal["is_weekend"] | day_cal["is_holiday"]).to_numpy().astype(bool)
    holiday_day = day_cal["is_holiday"].to_numpy().astype(bool)
    month_boost = EVENT_MONTH_BOOST[day_cal["month"].to_numpy() - 1]

    zone_code = np.zeros((n_hours, len(ZONES)), dtype=np.int8)
    zone_attendance = np.zeros((n_hours, len(ZONES)))

    for zone_idx, zone in enumerate(ZONES):
        weekday_rate, weekend_rate = EVENT_RATES[zone.zone_type]
        p_day = np.where(off_day, weekend_rate, weekday_rate) * month_boost * EVENT_RATE_SCALE
        if zone.zone_type in FESTIVE_ZONE_TYPES:
            p_day = np.where(holiday_day, p_day * 1.6, p_day)  # festivities on holidays
        p_day = np.clip(p_day, 0.0, 0.9)

        weights = EVENT_TYPE_WEIGHTS[zone.zone_type]
        names = list(weights)
        probs = np.array([weights[name] for name in names], dtype=float)
        probs /= probs.sum()

        for day in np.flatnonzero(rng.random(n_days) < p_day):
            name = names[int(rng.choice(len(names), p=probs))]
            start = day * 24 + int(
                rng.integers(EVENT_START_HOURS[name][0], EVENT_START_HOURS[name][1] + 1)
            )
            duration = int(
                rng.integers(EVENT_DURATION_HOURS[name][0], EVENT_DURATION_HOURS[name][1] + 1)
            )
            attendance = float(rng.lognormal(np.log(EVENT_MEAN_ATTENDANCE[name]), 0.5))
            window = slice(start, min(start + duration, n_hours))
            stronger = (
                zone_attendance[window, zone_idx] < attendance
            )  # keep the biggest overlapping event
            zone_code[window, zone_idx][stronger] = EVENT_CODE[name]
            zone_attendance[window, zone_idx][stronger] = attendance

    zone_of_ap = aps["zone_idx"].to_numpy()
    code_ap = zone_code[:, zone_of_ap]
    attendance_ap = zone_attendance[:, zone_of_ap] * aps["event_proximity"].to_numpy()[None, :]
    attendance_ap = np.where(
        code_ap > 0, attendance_ap * rng.lognormal(0, 0.12, attendance_ap.shape), 0.0
    )
    attendance_ap = np.minimum(np.rint(attendance_ap), MAX_EVENT_ATTENDANCE)
    return {"event_code": code_ap, "attendance": attendance_ap}


# ---------------------------------------------------------------------------
# 7. GEOGRAPHIC / CONTEXT GENERATION
# ---------------------------------------------------------------------------
def generate_context_features(
    cal: pd.DataFrame,
    aps: pd.DataFrame,
    activity: np.ndarray,
    weather: dict[str, np.ndarray],
    events: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Mobility/context proxies (people, traffic, transit, business, students, workers).

    Both a 'true' version (used by the latent demand process) and a noisy
    'observed' version (exported) are returned.
    """
    n_hours, n_aps = len(cal), len(aps)
    hour = cal["hour"].to_numpy()
    dow = cal["day_of_week"].to_numpy()
    working = cal["is_working_day"].to_numpy().astype(bool)
    school = cal["is_school_day"].to_numpy().astype(bool)
    holiday = cal["is_holiday"].to_numpy().astype(bool)
    rain_flag = weather["precipitation_mm"] >= RAIN_MM
    rain_factor = RAIN_FACTOR[weather["condition_code"]]
    attendance = events["attendance"]

    # Shared hourly curves.
    commute = (
        _bump(hour, 7.5, 1.2, 1.0) + _bump(hour, 12.5, 1.3, 0.45) + _bump(hour, 18.0, 1.5, 1.0)
    )
    leisure = 0.35 * (_bump(hour, 10.0, 3.0, 0.6) + _bump(hour, 17.0, 3.0, 0.6))
    transit_curve = np.where(working, commute, leisure) + 0.05
    day_scale = np.select([holiday, dow == 5, dow == 6], [0.40, 0.75, 0.35], default=1.0)
    business_curve = (
        0.05 + _bump(hour, 10.5, 3.0, 0.75) + _bump(hour, 16.0, 3.0, 0.80)
    ) * day_scale
    student_curve = np.where(
        school, 0.05 + _bump(hour, 9.5, 2.5, 0.8) + _bump(hour, 15.0, 2.5, 0.7), 0.03
    )
    worker_curve = np.where(
        working, 0.03 + _bump(hour, 10.5, 2.5, 0.85) + _bump(hour, 15.5, 2.5, 0.85), 0.02
    )

    shape = (n_hours, n_aps)
    students = (
        aps["students_base"].to_numpy()[None, :]
        * student_curve[:, None]
        * rng.lognormal(0, 0.18, shape)
    )
    workers = (
        aps["workers_base"].to_numpy()[None, :]
        * worker_curve[:, None]
        * rng.lognormal(0, 0.18, shape)
    )
    transit = np.clip(
        100
        * aps["transit_level"].to_numpy()[None, :]
        * transit_curve[:, None]
        * (1 + 0.12 * rain_flag)
        * rng.lognormal(0, 0.15, shape),
        0,
        100,
    )
    business = np.clip(
        100
        * aps["business_level"].to_numpy()[None, :]
        * business_curve[:, None]
        * rng.lognormal(0, 0.15, shape),
        0,
        100,
    )

    # People nearby: activity + persistent crowd shock + events, damped by rain outdoors.
    outdoor = aps["outdoor"].to_numpy()[None, :]
    crowd_shock = ar1_process(n_hours, n_aps, 0.80, 0.15, rng)
    people = (
        aps["people_base"].to_numpy()[None, :]
        * (0.08 + 0.92 * np.minimum(activity, 1.8))
        * np.exp(crowd_shock)
        * (1 - 0.12 * outdoor * rain_factor)
        + 0.6 * attendance
    )

    # Traffic: rush hours (working days), zone pressure, events, rain, noise.
    rush = _bump(hour, 7.5, 1.2, 1.0) + _bump(hour, 12.5, 1.2, 0.5) + _bump(hour, 18.0, 1.5, 1.0)
    traffic_index = (
        0.15
        + 0.50 * (rush * np.where(working, 1.0, 0.4))[:, None]
        + 0.25 * aps["traffic_base"].to_numpy()[None, :]
        + 0.0003 * attendance
        + 0.08 * rain_flag
        + 0.04 * (cal["month"].to_numpy() == 12)[:, None]
        + rng.normal(0, 0.07, shape)
    )
    traffic_code = np.select(
        [traffic_index < 0.45, traffic_index < 0.70], [0, 1], default=2
    ).astype(np.int8)

    return {
        "crowd_shock": crowd_shock,
        "students_true": students,
        "workers_true": workers,
        "transit_true": transit,
        "business_true": business,
        "people_true": people,
        "students_obs": np.rint(students * rng.lognormal(0, 0.08, shape)),
        "workers_obs": np.rint(workers * rng.lognormal(0, 0.08, shape)),
        "transit_obs": np.round(np.clip(transit * rng.lognormal(0, 0.08, shape), 0, 100), 1),
        "business_obs": np.round(np.clip(business * rng.lognormal(0, 0.08, shape), 0, 100), 1),
        "people_obs": np.rint(people * rng.lognormal(0, 0.10, shape)),
        "traffic_code": traffic_code,
    }


# ---------------------------------------------------------------------------
# 12. LATENT DEMAND GENERATION
# ---------------------------------------------------------------------------
def simulate_connections(
    cal: pd.DataFrame,
    aps: pd.DataFrame,
    activity: np.ndarray,
    weather: dict[str, np.ndarray],
    events: dict[str, np.ndarray],
    context: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate hourly connection counts from a latent (never exported) demand process.

    log(mean) = AP scale + log(zone/calendar activity) + weather + events
                + context + crowd shock + AP AR(1) shock + zone-day shock
    Counts are then drawn from a Negative Binomial (Gamma-Poisson mixture).
    """
    n_hours, n_aps = len(cal), len(aps)
    n_days = n_hours // 24
    outdoor = aps["outdoor"].to_numpy()[None, :]
    capacity = aps["access_point_capacity"].to_numpy()

    # Weather: rain hurts outdoor zones and (slightly) helps indoor ones; cold/wind hurt outdoor.
    rain_factor = RAIN_FACTOR[weather["condition_code"]]
    weather_effect = (
        rain_factor * (0.12 - 0.47 * outdoor)
        - 0.02 * np.maximum(12.0 - weather["temperature_c"], 0.0) * outdoor
        - 0.004 * weather["wind_speed_kmh"] * outdoor
    )

    # Events create demand spikes (diminishing returns with attendance).
    event_effect = 0.32 * np.log1p(events["attendance"] / 150.0)

    # Nearby populations / transit / business activity.
    context_effect = (
        0.10 * np.log1p(context["students_true"] / 250.0)
        + 0.10 * np.log1p(context["workers_true"] / 250.0)
        + 0.12 * context["transit_true"] / 100.0
        + 0.12 * context["business_true"] / 100.0
    )

    # Persistent shocks: this is what makes recent history so informative.
    ap_shock = ar1_process(n_hours, n_aps, 0.90, 0.18, rng)
    zone_day_shock = np.repeat(ar1_process(n_days, len(ZONES), 0.40, 0.10, rng), 24, axis=0)
    zone_day_shock = zone_day_shock[:, aps["zone_idx"].to_numpy()]

    scale = capacity * 1.3 * aps["site_factor"].to_numpy()
    log_mu = (
        np.log(scale)[None, :]
        + np.log(np.maximum(activity, 1e-3))
        + weather_effect
        + event_effect
        + context_effect
        + 0.35 * context["crowd_shock"]
        + ap_shock
        + zone_day_shock
    )

    mean = np.exp(log_mu)
    gamma_draw = rng.gamma(shape=DISPERSION, scale=mean / DISPERSION)
    connections = rng.poisson(gamma_draw)
    return np.minimum(connections, 4 * capacity[None, :]).astype(np.int64)


# ---------------------------------------------------------------------------
# 10. HISTORICAL FEATURE GENERATION
# ---------------------------------------------------------------------------
def generate_historical_features(connections: np.ndarray) -> dict[str, np.ndarray]:
    """Lag / rolling features computed from the FULL hourly series.

    Convention: at row time ``t`` (start of the predicted hour) only hours
    ``<= t-1`` are used. Index ``t`` itself (the target hour) is never touched.
    """
    n_hours, n_aps = connections.shape
    counts = connections.astype(float)
    nan = np.nan

    def lagged(lag: int) -> np.ndarray:
        out = np.full((n_hours, n_aps), nan)
        out[lag:] = counts[:-lag]
        return out

    cumulative = np.vstack(
        [np.zeros((1, n_aps)), np.cumsum(counts, axis=0)]
    )  # cumulative[k] = sum(c[0..k-1])

    def trailing_mean(window: int) -> np.ndarray:
        """mean(c[t-window .. t-1]) for every t >= window."""
        out = np.full((n_hours, n_aps), nan)
        out[window:] = (cumulative[window:n_hours] - cumulative[: n_hours - window]) / window
        return out

    # Mean hourly connections of the previous COMPLETE calendar day.
    daily_mean = counts.reshape(n_hours // 24, 24, n_aps).mean(axis=1)
    previous_day = np.full((n_hours, n_aps), nan)
    previous_day[24:] = np.repeat(daily_mean[:-1], 24, axis=0)

    return {
        "previous_hour": lagged(1),
        "previous_day_mean": previous_day,
        "same_hour_previous_day": lagged(24),
        "same_hour_previous_week": lagged(168),
        "avg_3h": trailing_mean(3),
        "avg_24h": trailing_mean(24),
        "avg_7d": trailing_mean(168),
    }


# ---------------------------------------------------------------------------
# 11. NETWORK FEATURE GENERATION
# ---------------------------------------------------------------------------
def generate_network_features(
    connections: np.ndarray,
    aps: pd.DataFrame,
    weather: dict[str, np.ndarray],
    cal: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Network snapshot metrics measured at prediction time.

    They summarise the hour that JUST ENDED (connections at t-1), so they never
    contain information about the hour being predicted.
    """
    n_hours, n_aps = connections.shape
    shape = (n_hours, n_aps)
    capacity = aps["access_point_capacity"].to_numpy()[None, :]
    previous = np.vstack([connections[:1], connections[:-1]]).astype(
        float
    )  # row 0 is warm-up, never exported
    evening = ((cal["hour"].to_numpy() >= 18) & (cal["hour"].to_numpy() <= 22))[:, None]

    # Little's law: concurrent devices ~ arrival rate x session duration.
    session = aps["session_base"].to_numpy()[None, :] * rng.lognormal(0, 0.18, shape)
    session = np.clip(session, 2.0, 120.0)
    devices = np.clip(
        np.rint(previous * session / 60.0 * rng.lognormal(0, 0.18, shape)), 0, capacity
    )
    sessions = np.minimum(np.rint(devices * rng.beta(8, 3, shape)), devices)

    per_session_mbps = rng.lognormal(np.log(0.55), 0.45, shape) * (1 + 0.25 * evening)
    bandwidth = np.clip(sessions * per_session_mbps, 0, 1.1 * capacity)

    load = devices / capacity
    utilization = np.clip(6 + 78 * load**0.85 + rng.normal(0, 4.5, shape), 1, 100)
    latency = (
        12 + 0.30 * utilization + 0.012 * np.maximum(utilization - 55, 0) ** 2
    ) * rng.lognormal(0, 0.25, shape)
    packet_loss = (
        0.15
        + 0.012 * utilization
        + 0.02 * np.maximum(utilization - 80, 0)
        + rng.gamma(0.8, 0.25, shape)
        + (rng.random(shape) < 0.01) * rng.uniform(2, 6, shape)
    )
    signal = (
        aps["signal_offset"].to_numpy()[None, :]
        - 0.06 * utilization
        - 0.5 * np.minimum(weather["precipitation_mm"], 8)
        + rng.normal(0, 3.5, shape)
    )
    outage = (rng.random(shape) < 0.006) * rng.uniform(1, 12, shape)
    uptime = 100 - rng.exponential(0.12, shape) - outage

    return {
        "connected_devices": devices.astype(np.int64),
        "active_sessions": sessions.astype(np.int64),
        "average_session_duration_min": np.round(session, 1),
        "bandwidth_usage_mbps": np.round(bandwidth, 2),
        "packet_loss_percent": np.round(np.clip(packet_loss, 0, 25), 2),
        "latency_ms": np.round(np.clip(latency, 4, 800), 1),
        "signal_strength_dbm": np.round(np.clip(signal, -95, -35), 1),
        "channel_utilization_percent": np.round(utilization, 1),
        "network_uptime_percent": np.round(np.clip(uptime, 80, 100), 2),
    }


# ---------------------------------------------------------------------------
# 13. TARGET GENERATION  (+ sampling and dataset assembly)
# ---------------------------------------------------------------------------
def sample_observations(
    n_hours: int, n_aps: int, n_rows: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Reproducibly sample AP-hour observations from the exported period (no warm-up)."""
    warmup_hours = WARMUP_DAYS * 24
    n_eligible = (n_hours - warmup_hours) * n_aps
    flat = np.sort(rng.choice(n_eligible, size=n_rows, replace=False))  # time-major order
    return warmup_hours + flat // n_aps, flat % n_aps


def assemble_dataset(
    cal: pd.DataFrame,
    aps: pd.DataFrame,
    t_idx: np.ndarray,
    a_idx: np.ndarray,
    weather: dict,
    context: dict,
    events: dict,
    network: dict,
    history: dict,
    connections: np.ndarray,
    high_load_threshold: float,
) -> pd.DataFrame:
    """Build the exported table. The latent demand process is NOT included."""
    cal_rows = cal.iloc[t_idx].reset_index(drop=True)
    ap_rows = aps.iloc[a_idx].reset_index(drop=True)

    def pick(array: np.ndarray) -> np.ndarray:
        return array[t_idx, a_idx]

    def as_count(array: np.ndarray) -> np.ndarray:
        return pick(array).astype(np.int64)

    target = as_count(connections)
    capacity = ap_rows["access_point_capacity"].to_numpy()
    demand_level = np.where(target / capacity >= high_load_threshold, "HIGH", "LOW")
    event_code = pick(events["event_code"])

    data = {c: cal_rows[c] for c in ["timestamp"]}
    data.update(
        {
            c: ap_rows[c]
            for c in [
                "wifi_id",
                "zone_id",
                "zone_name",
                "zone_type",
                "latitude",
                "longitude",
                "altitude_m",
            ]
        }
    )
    data.update({c: cal_rows[c] for c in TEMPORAL_COLUMNS})
    data.update(
        {
            "temperature_c": pick(weather["temperature_c"]),
            "humidity_percent": pick(weather["humidity_percent"]),
            "precipitation_mm": pick(weather["precipitation_mm"]),
            "wind_speed_kmh": pick(weather["wind_speed_kmh"]),
            "weather_condition": np.array(WEATHER_CONDITIONS)[pick(weather["condition_code"])],
            "estimated_people_nearby": as_count(context["people_obs"]),
            "traffic_level": np.array(TRAFFIC_LEVELS)[pick(context["traffic_code"])],
            "public_transport_activity": pick(context["transit_obs"]),
            "nearby_business_activity": pick(context["business_obs"]),
            "nearby_student_population": as_count(context["students_obs"]),
            "nearby_worker_population": as_count(context["workers_obs"]),
            "event_nearby": (event_code > 0).astype(np.int64),
            "event_type": np.array(EVENT_TYPES)[event_code],
            "estimated_event_attendance": as_count(events["attendance"]),
            "access_point_capacity": capacity,
            "connected_devices": pick(network["connected_devices"]),
            "active_sessions": pick(network["active_sessions"]),
            "average_session_duration_min": pick(network["average_session_duration_min"]),
            "bandwidth_usage_mbps": pick(network["bandwidth_usage_mbps"]),
            "packet_loss_percent": pick(network["packet_loss_percent"]),
            "latency_ms": pick(network["latency_ms"]),
            "signal_strength_dbm": pick(network["signal_strength_dbm"]),
            "channel_utilization_percent": pick(network["channel_utilization_percent"]),
            "network_uptime_percent": pick(network["network_uptime_percent"]),
            "connections_previous_hour": as_count(history["previous_hour"]),
            "connections_previous_day": np.round(pick(history["previous_day_mean"]), 2),
            "connections_same_hour_previous_day": as_count(history["same_hour_previous_day"]),
            "connections_same_hour_previous_week": as_count(history["same_hour_previous_week"]),
            "average_connections_last_3_hours": np.round(pick(history["avg_3h"]), 2),
            "average_connections_last_24_hours": np.round(pick(history["avg_24h"]), 2),
            "average_connections_last_7_days": np.round(pick(history["avg_7d"]), 2),
            "connections_next_hour": target,
            "demand_level": demand_level,
        }
    )
    return pd.DataFrame(data)[FINAL_COLUMNS]


# ---------------------------------------------------------------------------
# 14. DATA QUALITY CORRUPTION
# ---------------------------------------------------------------------------
def apply_data_quality_issues(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject realistic missing values and a handful of exact duplicate rows.

    Identifiers, temporal fields and targets are never corrupted.
    """
    df = df.copy()
    for column, rate in MISSING_RATES.items():
        if pd.api.types.is_integer_dtype(df[column]):
            df[column] = df[column].astype("Int64")  # nullable ints keep clean CSV output
        df[column] = df[column].mask(rng.random(len(df)) < rate)

    duplicate_positions = rng.choice(len(df), size=N_DUPLICATE_ROWS, replace=False)
    df = pd.concat([df, df.iloc[duplicate_positions]], ignore_index=True)
    return df.sort_values(["timestamp", "wifi_id"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 15. VALIDATION
# ---------------------------------------------------------------------------
def _as_float(series: pd.Series) -> np.ndarray:
    """Convert any numeric Series (incl. nullable Int64) to float64 with NaN for missing."""
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64", na_value=np.nan)


def _check_lag_alignment(df: pd.DataFrame, lag_hours: int, feature: str) -> tuple[int, int]:
    """Compare a lag feature with the target of the row observed `lag_hours` earlier.

    Returns (n_compared, n_mismatched). The row at t-lag has
    connections_next_hour == c[t-lag], which is exactly what the lag feature
    at row t must contain. A mismatch means misaligned (possibly leaky) features.
    """
    unique = df.drop_duplicates(["wifi_id", "timestamp"])
    earlier = unique[["wifi_id", "timestamp", "connections_next_hour"]].copy()
    earlier["timestamp"] = earlier["timestamp"] + pd.Timedelta(hours=lag_hours)
    earlier = earlier.rename(columns={"connections_next_hour": "earlier_target"})
    merged = unique[["wifi_id", "timestamp", feature]].merge(earlier, on=["wifi_id", "timestamp"])
    merged = merged.dropna(subset=[feature])
    mismatches = int((_as_float(merged[feature]) != _as_float(merged["earlier_target"])).sum())
    return len(merged), mismatches


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate the dataset. Raises DatasetValidationError on any critical failure."""
    problems: list[str] = []

    # 1. Minimum / maximum row count
    n_rows = len(df)
    if n_rows < MIN_ROWS:
        problems.append(f"Row count {n_rows:,} is below the minimum {MIN_ROWS:,}.")
    if n_rows > MAX_ROWS:
        problems.append(f"Row count {n_rows:,} exceeds the maximum {MAX_ROWS:,}.")

    # 2. Required columns (and exact order); no leaky columns
    missing_columns = [c for c in FINAL_COLUMNS if c not in df.columns]
    if missing_columns:
        problems.append(f"Missing required columns: {missing_columns}")
        raise DatasetValidationError("Dataset validation failed:\n - " + "\n - ".join(problems))
    if list(df.columns) != FINAL_COLUMNS:
        problems.append("Column order differs from FINAL_COLUMNS.")
    leaky = [c for c in FORBIDDEN_COLUMNS if c in df.columns]
    if leaky:
        problems.append(f"Forbidden (leaky) columns present: {leaky}")

    # 3-4. Targets: no missing values, valid labels
    for target in TARGET_COLUMNS:
        if df[target].isna().any():
            problems.append(f"Target column '{target}' contains missing values.")
    invalid_labels = set(df["demand_level"].dropna().unique()) - {"LOW", "HIGH"}
    if invalid_labels:
        problems.append(f"demand_level has invalid labels: {sorted(invalid_labels)}")
    for column in IDENTIFIER_COLUMNS + TEMPORAL_COLUMNS:
        if df[column].isna().any():
            problems.append(f"Identifier/temporal column '{column}' contains missing values.")

    # 5-7. Coordinates and numeric ranges
    if not df["latitude"].between(-90, 90).all() or not df["longitude"].between(-180, 180).all():
        problems.append("Latitude/longitude outside valid geographic limits.")
    for column, (low, high) in RANGE_CHECKS.items():
        values = _as_float(df[column])
        values = values[~np.isnan(values)]
        if len(values) and (values.min() < low or values.max() > high):
            problems.append(
                f"'{column}' out of range [{low}, {high}]: observed [{values.min()}, {values.max()}]."
            )
    devices, sessions, capacity = (
        _as_float(df[c]) for c in ("connected_devices", "active_sessions", "access_point_capacity")
    )
    if np.nanmax(sessions - devices) > 0 or np.nanmax(devices - capacity) > 0:
        problems.append(
            "Network consistency broken (active_sessions > connected_devices or devices > capacity)."
        )

    # 8. Timestamps parse and temporal fields are consistent with them
    timestamps = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamps.isna().any():
        problems.append("Some timestamps cannot be parsed.")
    else:
        if (
            not (df["hour"] == timestamps.dt.hour).all()
            or not (df["day_of_week"] == timestamps.dt.dayofweek).all()
        ):
            problems.append("hour/day_of_week are inconsistent with timestamp.")
        if not (df["is_weekend"] == (timestamps.dt.dayofweek >= 5).astype(int)).all():
            problems.append("is_weekend is inconsistent with day_of_week.")
        expected_period = np.select(
            [timestamps.dt.hour < 6, timestamps.dt.hour < 12, timestamps.dt.hour < 18],
            ["NIGHT", "MORNING", "AFTERNOON"],
            default="EVENING",
        )
        if not (df["time_period"].to_numpy() == expected_period).all():
            problems.append("time_period is inconsistent with hour.")

        # 9. Expected date range
        first_day, last_day = timestamps.min().normalize(), timestamps.max().normalize()
        if first_day != pd.Timestamp(START_DATE) or last_day != pd.Timestamp(END_DATE):
            problems.append(
                f"Date range {first_day.date()} -> {last_day.date()} differs from "
                f"{START_DATE} -> {END_DATE}."
            )

    # 10-12. Duplicates, missing values, class distribution
    duplicate_count = int(df.duplicated().sum())
    if duplicate_count > 0.01 * n_rows:
        problems.append(f"Too many duplicate rows: {duplicate_count}.")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    for column, count in missing.items():
        if count / n_rows > 0.05:
            problems.append(f"Column '{column}' has {count / n_rows:.1%} missing values (max 5%).")
    distribution = df["demand_level"].value_counts(normalize=True)
    low_share, high_share = float(distribution.get("LOW", 0.0)), float(
        distribution.get("HIGH", 0.0)
    )
    if not (
        LOW_SHARE_RANGE[0] <= low_share <= LOW_SHARE_RANGE[1]
        and HIGH_SHARE_RANGE[0] <= high_share <= HIGH_SHARE_RANGE[1]
    ):
        problems.append(
            f"Target distribution out of range: LOW={low_share:.1%}, HIGH={high_share:.1%}."
        )

    # Temporal-leakage guard: lag features must equal targets observed earlier.
    lag_report: dict[str, tuple[int, int]] = {}
    for lag_hours, feature in [
        (1, "connections_previous_hour"),
        (24, "connections_same_hour_previous_day"),
        (168, "connections_same_hour_previous_week"),
    ]:
        compared, mismatched = _check_lag_alignment(df, lag_hours, feature)
        lag_report[feature] = (compared, mismatched)
        if compared == 0 or mismatched > 0:
            problems.append(
                f"Lag alignment check failed for '{feature}' ({mismatched} mismatches / {compared})."
            )

    if problems:
        raise DatasetValidationError("Dataset validation failed:\n - " + "\n - ".join(problems))

    return {
        "n_rows": n_rows,
        "n_columns": df.shape[1],
        "first_day": first_day.date(),
        "last_day": last_day.date(),
        "n_access_points": int(df["wifi_id"].nunique()),
        "n_zones": int(df["zone_id"].nunique()),
        "low_share": low_share,
        "high_share": high_share,
        "missing": missing,
        "duplicates": duplicate_count,
        "lag_checks": lag_report,
    }


# ---------------------------------------------------------------------------
# 16. CSV EXPORT
# ---------------------------------------------------------------------------
def export_csv(df: pd.DataFrame, path: Path = OUTPUT_PATH) -> Path:
    """Write the dataset as UTF-8, comma-separated CSV (creates directories if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", sep=",", date_format="%Y-%m-%d %H:%M:%S")
    return path


# ---------------------------------------------------------------------------
# 17. SUMMARY OUTPUT
# ---------------------------------------------------------------------------
def print_summary(summary: dict, threshold: float, output_path: Path) -> None:
    """Print a human-readable generation summary."""
    line = "=" * 60
    try:
        shown_path = output_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        shown_path = str(output_path)
    print(line)
    print("WiFi Tunja Smart Predictor")
    print("Synthetic Dataset Generator")
    print(line)
    print()
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Rows generated: {summary['n_rows']:,}")
    print(f"Columns generated: {summary['n_columns']}")
    print()
    print("Date range:")
    print(f"{summary['first_day']} -> {summary['last_day']}")
    print()
    print(f"WiFi access points: {summary['n_access_points']}")
    print(f"Synthetic zones: {summary['n_zones']}")
    print()
    print("Target distribution:")
    print(f"LOW: {summary['low_share']:.1%}")
    print(f"HIGH: {summary['high_share']:.1%}")
    print(f"(HIGH = connections_next_hour / access_point_capacity >= {threshold:.3f})")
    print()
    print("Missing values:")
    if len(summary["missing"]) == 0:
        print("none")
    for column, count in summary["missing"].items():
        print(f"  {column}: {count:,} ({count / summary['n_rows']:.1%})")
    print()
    print(f"Duplicate rows: {summary['duplicates']:,}")
    print()
    print("Temporal-leakage checks (lag feature vs. earlier target; compared / mismatches):")
    for feature, (compared, mismatched) in summary["lag_checks"].items():
        print(f"  {feature}: {compared:,} / {mismatched}")
    print()
    print("Output:")
    print(shown_path)
    print()
    print(DISCLAIMER)
    print("Dataset generation completed successfully.")
    print(line)


# ---------------------------------------------------------------------------
# 18. MAIN ENTRY POINT
# ---------------------------------------------------------------------------
def build_dataset(rng: np.random.Generator) -> tuple[pd.DataFrame, float]:
    """Run the whole simulation and return (dataset before quality issues, HIGH threshold)."""
    aps = generate_access_points(rng)
    cal = build_calendar(make_hourly_index())

    # Order follows data dependencies, not the section numbering above.
    weather = generate_weather(cal, aps, rng)
    events = generate_events(cal, aps, rng)
    activity = compute_activity(cal, aps)
    context = generate_context_features(cal, aps, activity, weather, events, rng)
    connections = simulate_connections(
        cal, aps, activity, weather, events, context, rng
    )  # latent -> counts
    history = generate_historical_features(connections)
    network = generate_network_features(connections, aps, weather, cal, rng)

    # HIGH/LOW: load ratio (connections / capacity) above a calibrated threshold.
    warmup_hours = WARMUP_DAYS * 24
    load_ratio = connections[warmup_hours:] / aps["access_point_capacity"].to_numpy()[None, :]
    high_load_threshold = float(np.quantile(load_ratio, 1.0 - HIGH_DEMAND_SHARE))

    n_base_rows = N_ROWS - N_DUPLICATE_ROWS
    t_idx, a_idx = sample_observations(len(cal), len(aps), n_base_rows, rng)
    dataset = assemble_dataset(
        cal,
        aps,
        t_idx,
        a_idx,
        weather,
        context,
        events,
        network,
        history,
        connections,
        high_load_threshold,
    )
    return dataset, high_load_threshold


def main() -> int:
    """Generate, corrupt, validate, export and summarise the synthetic dataset."""
    rng = make_rng(RANDOM_SEED)
    dataset, threshold = build_dataset(rng)
    dataset = apply_data_quality_issues(dataset, rng)
    try:
        summary = validate_dataset(dataset)
    except DatasetValidationError as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    output_path = export_csv(dataset, OUTPUT_PATH)
    print_summary(summary, threshold, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
