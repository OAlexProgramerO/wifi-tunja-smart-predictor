"""Geospatial demand summaries distinguish observed targets from predictions."""

from __future__ import annotations

import pandas as pd
import pytest

from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.analytics.geospatial import analyze_geospatial_demand


def test_geospatial_aggregations_use_observation_counts_means_and_high_rates():
    frame = make_mini_frame(8).copy()
    result = analyze_geospatial_demand(frame)

    assert result.observations.shape[0] == len(frame)
    assert result.by_hour["observations"].sum() == len(frame)
    assert result.by_weekday["observations"].sum() == len(frame)
    assert result.by_zone["observations"].sum() == len(frame)
    assert result.by_zone["high_demand_rate"].between(0, 1).all()
    assert result.by_zone["mean_observed_next_hour_connections"].notna().all()
    assert {"observations", "mean_observed_next_hour_connections"}.issubset(
        result.access_points.columns
    )
    assert result.peak_hour in range(24)
    assert result.peak_weekday in range(7)


def test_geospatial_filters_have_inclusive_date_and_hour_weekday_boundaries():
    frame = make_mini_frame(8).copy()
    timestamps = pd.to_datetime(frame["timestamp"])
    target_date = timestamps.iloc[0].date()
    expected = frame.loc[timestamps.dt.date == target_date]
    zone_id = str(expected["zone_id"].iloc[0])
    target_hour = int(timestamps.iloc[0].hour)
    target_weekday = int(timestamps.iloc[0].dayofweek)

    result = analyze_geospatial_demand(
        frame,
        start_date=target_date,
        end_date=target_date,
        zone_ids=[zone_id],
        hour=target_hour,
        weekday=target_weekday,
        demand_level="high",
    )
    expected_filtered = expected.loc[
        expected["zone_id"].eq(zone_id)
        & timestamps.loc[expected.index].dt.hour.eq(target_hour)
        & timestamps.loc[expected.index].dt.dayofweek.eq(target_weekday)
        & expected["demand_level"].eq("HIGH")
    ]
    assert len(result.observations) == len(expected_filtered)
    assert result.observations["demand_level"].eq("HIGH").all()
    assert result.observations["zone_id"].astype(str).eq(zone_id).all()


def test_geospatial_empty_filter_and_invalid_coordinates_are_safe():
    frame = make_mini_frame(8).copy()
    frame["latitude"] = float("nan")
    frame["longitude"] = 200
    result = analyze_geospatial_demand(frame, zone_ids=["NO_SUCH_SYNTHETIC_ZONE"])
    assert result.observations.empty
    assert result.by_hour.empty
    assert result.access_points.empty

    invalid_points = analyze_geospatial_demand(frame)
    assert not invalid_points.observations.empty
    assert invalid_points.access_points.empty


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"hour": 24}, "hour must be between"),
        ({"weekday": 7}, "weekday must be between"),
        ({"demand_level": "MEDIUM"}, "demand_level must be LOW or HIGH"),
        ({"start_date": "2025-02-02", "end_date": "2025-02-01"}, "on or before"),
    ],
)
def test_geospatial_rejects_invalid_filters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        analyze_geospatial_demand(make_mini_frame(8), **kwargs)


def test_geospatial_requires_observation_fields():
    with pytest.raises(ValueError, match="Missing required demand columns"):
        analyze_geospatial_demand(pd.DataFrame({"timestamp": []}))
