"""Filter and summarize historical synthetic demand without model inference."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import pandas as pd

_REQUIRED_COLUMNS = {
    "timestamp",
    "zone_id",
    "connections_next_hour",
    "demand_level",
}


@dataclass(frozen=True)
class GeospatialDemandAnalysis:
    """Descriptive tables derived only from filtered synthetic observations."""

    observations: pd.DataFrame
    by_hour: pd.DataFrame
    by_weekday: pd.DataFrame
    by_zone: pd.DataFrame
    zone_by_hour: pd.DataFrame
    access_points: pd.DataFrame
    peak_hour: int | None
    peak_weekday: int | None


def _validate_date(value: date | str | None, field: str) -> pd.Timestamp | None:
    if value is None:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a valid date") from exc
    if pd.isna(stamp):
        raise ValueError(f"{field} must be a valid date")
    return stamp.normalize()


def _group_summary(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                *keys,
                "observations",
                "mean_observed_next_hour_connections",
                "peak_observed_next_hour_connections",
                "high_demand_rate",
            ]
        )
    return (
        frame.groupby(keys, as_index=False, dropna=False)
        .agg(
            observations=("connections_next_hour", "count"),
            mean_observed_next_hour_connections=("connections_next_hour", "mean"),
            peak_observed_next_hour_connections=("connections_next_hour", "max"),
            high_demand_rate=(
                "demand_level",
                lambda values: (
                    values.dropna().eq("HIGH").mean() if values.notna().any() else float("nan")
                ),
            ),
        )
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )


def analyze_geospatial_demand(
    frame: pd.DataFrame,
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
    zone_ids: Sequence[str] | None = None,
    hour: int | None = None,
    weekday: int | None = None,
    demand_level: str | None = None,
) -> GeospatialDemandAnalysis:
    """Return explicitly descriptive summaries of historical synthetic rows.

    ``connections_next_hour`` is an observed synthetic target in the source data,
    not a model prediction and not a value available to a live prediction.
    Date bounds are inclusive calendar dates; hour uses 0–23 and weekday uses
    Monday=0 through Sunday=6.
    """
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required demand columns: {', '.join(missing)}")
    if hour is not None and (isinstance(hour, bool) or not 0 <= hour <= 23):
        raise ValueError("hour must be between 0 and 23")
    if weekday is not None and (isinstance(weekday, bool) or not 0 <= weekday <= 6):
        raise ValueError("weekday must be between 0 (Monday) and 6 (Sunday)")
    if demand_level is not None:
        demand_level = str(demand_level).upper()
        if demand_level not in {"LOW", "HIGH"}:
            raise ValueError("demand_level must be LOW or HIGH")

    start = _validate_date(start_date, "start_date")
    end = _validate_date(end_date, "end_date")
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be on or before end_date")

    columns = list(frame.columns)
    working = frame.copy()
    working["_timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    working = working.loc[working["_timestamp"].notna()].copy()
    calendar_dates = working["_timestamp"].dt.date
    if start is not None:
        working = working.loc[calendar_dates >= start.date()]
    if end is not None:
        working = working.loc[calendar_dates <= end.date()]
    if zone_ids is not None:
        selected_zones = {str(zone_id) for zone_id in zone_ids}
        working = working.loc[working["zone_id"].astype(str).isin(selected_zones)]
    if hour is not None:
        working = working.loc[working["_timestamp"].dt.hour == hour]
    if weekday is not None:
        working = working.loc[working["_timestamp"].dt.dayofweek == weekday]
    if demand_level is not None:
        working = working.loc[working["demand_level"].astype("string").str.upper() == demand_level]

    working["hour"] = working["_timestamp"].dt.hour
    working["weekday"] = working["_timestamp"].dt.dayofweek
    by_hour = _group_summary(working, ["hour"])
    by_weekday = _group_summary(working, ["weekday"])

    zone_keys = ["zone_id"]
    if "zone_name" in working.columns:
        zone_keys.append("zone_name")
    by_zone = _group_summary(working, zone_keys)
    if "wifi_id" in working.columns and not by_zone.empty:
        access_point_counts = working.groupby(zone_keys, dropna=False)["wifi_id"].nunique()
        by_zone["synthetic_access_points"] = [
            int(access_point_counts.loc[tuple(row) if len(zone_keys) > 1 else row[0]])
            for row in by_zone[zone_keys].itertuples(index=False, name=None)
        ]
    elif "wifi_id" in working.columns:
        by_zone["synthetic_access_points"] = pd.Series(dtype="int64")
    zone_by_hour = (
        working.pivot_table(
            index=zone_keys,
            columns="hour",
            values="connections_next_hour",
            aggfunc="mean",
            observed=False,
        ).sort_index()
        if not working.empty
        else pd.DataFrame()
    )
    zone_by_hour.columns.name = "hour"

    ap_keys = [column for column in ("wifi_id", "zone_id", "zone_name") if column in working]
    point_columns = [column for column in ("latitude", "longitude") if column in working]
    if "wifi_id" in ap_keys and point_columns:
        access_points = (
            working.groupby(ap_keys, as_index=False, dropna=False)
            .agg(
                observations=("connections_next_hour", "count"),
                mean_observed_next_hour_connections=("connections_next_hour", "mean"),
                high_demand_rate=(
                    "demand_level",
                    lambda values: (
                        values.dropna().eq("HIGH").mean() if values.notna().any() else float("nan")
                    ),
                ),
                **{column: (column, "first") for column in point_columns},
            )
            .sort_values("wifi_id", kind="stable")
            .reset_index(drop=True)
        )
        access_points["latitude"] = pd.to_numeric(access_points["latitude"], errors="coerce")
        access_points["longitude"] = pd.to_numeric(access_points["longitude"], errors="coerce")
        access_points = access_points.loc[
            access_points["latitude"].between(-90, 90)
            & access_points["longitude"].between(-180, 180)
        ].reset_index(drop=True)
    else:
        access_points = pd.DataFrame(
            columns=[
                *ap_keys,
                *point_columns,
                "observations",
                "mean_observed_next_hour_connections",
                "high_demand_rate",
            ]
        )

    peak_hour = None
    if not by_hour.empty:
        peak_hour = int(
            by_hour.sort_values(
                ["mean_observed_next_hour_connections", "hour"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]["hour"]
        )
    peak_weekday = None
    if not by_weekday.empty:
        peak_weekday = int(
            by_weekday.sort_values(
                ["mean_observed_next_hour_connections", "weekday"],
                ascending=[False, True],
                kind="stable",
            ).iloc[0]["weekday"]
        )
    working = working.drop(columns="_timestamp")
    working = working.reindex(columns=[*columns, "hour", "weekday"])
    return GeospatialDemandAnalysis(
        observations=working,
        by_hour=by_hour,
        by_weekday=by_weekday,
        by_zone=by_zone,
        zone_by_hour=zone_by_hour,
        access_points=access_points,
        peak_hour=peak_hour,
        peak_weekday=peak_weekday,
    )
