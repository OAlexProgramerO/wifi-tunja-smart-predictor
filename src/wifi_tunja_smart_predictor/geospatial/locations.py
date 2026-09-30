"""Deterministic lookup of simulated zones, access points, and coordinates."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from wifi_tunja_smart_predictor.exceptions import LocationResolutionError

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class ResolvedLocation:
    """Selected synthetic access point and its simulated zone metadata."""

    access_point_id: str
    zone_id: str
    zone_name: str
    zone_type: str
    latitude: float
    longitude: float
    access_point_capacity: float
    distance_to_ap_km: float | None
    selection_method: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible location metadata."""
        return asdict(self)


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compute great-circle distance from one coordinate to arrays of points."""
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    delta_lat = lat2_rad - lat1_rad
    delta_lon = np.radians(lon2 - lon1)
    value = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0, 1)))


class LocationResolver:
    """Resolve friendly synthetic location names or coordinates to an AP."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._catalog = (
            frame.sort_values("wifi_id")
            .groupby("wifi_id", as_index=False)
            .agg(
                zone_id=("zone_id", "first"),
                zone_name=("zone_name", "first"),
                zone_type=("zone_type", "first"),
                latitude=("latitude", "first"),
                longitude=("longitude", "first"),
                access_point_capacity=("access_point_capacity", "median"),
            )
        )
        self._catalog["_zone_label"] = self._catalog["zone_name"].map(_normalize)

    @property
    def catalog(self) -> pd.DataFrame:
        """Return a copy of unique simulated access-point metadata."""
        return self._catalog.drop(columns="_zone_label").copy()

    def resolve(
        self,
        *,
        zone: str | None = None,
        zone_id: str | None = None,
        access_point_id: str | None = None,
        latitude: float | None = None,
        longitude: float | None = None,
    ) -> ResolvedLocation:
        """Resolve an AP, zone alias, or coordinate pair; reject ambiguous absence."""
        if (latitude is None) != (longitude is None):
            raise LocationResolutionError("Provide both latitude and longitude, or neither.")
        candidates = self._catalog
        distance: float | None = None
        method = "zone_alias"

        if access_point_id:
            candidates = candidates.loc[candidates["wifi_id"] == access_point_id]
            method = "access_point_id"
        elif latitude is not None and longitude is not None:
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise LocationResolutionError(
                    "Coordinates must be valid latitude/longitude values."
                )
            distances = haversine_km(
                float(latitude),
                float(longitude),
                self._catalog["latitude"].to_numpy(dtype=float),
                self._catalog["longitude"].to_numpy(dtype=float),
            )
            index = int(np.argmin(distances))
            distance = float(distances[index])
            candidates = self._catalog.iloc[[index]]
            method = "nearest_coordinates"
        elif zone_id:
            candidates = candidates.loc[candidates["zone_id"].str.casefold() == zone_id.casefold()]
            method = "zone_id"
        elif zone:
            term = _normalize(zone)
            exact = candidates.loc[
                (candidates["_zone_label"] == term)
                | (candidates["zone_type"].str.casefold() == term.replace(" ", "_"))
                | (candidates["zone_id"].str.casefold() == term.replace(" ", "_"))
                | (candidates["wifi_id"].str.casefold() == term.replace(" ", "_"))
            ]
            candidates = (
                exact
                if not exact.empty
                else candidates.loc[
                    candidates["_zone_label"].str.contains(re.escape(term), na=False)
                ]
            )
            if candidates.empty and term in {"north", "south", "university"}:
                candidates = self._catalog.loc[
                    self._catalog["zone_type"].eq("UNIVERSITY")
                    & self._catalog["_zone_label"].str.contains(term, na=False)
                ]
            method = "zone_alias"
        else:
            raise LocationResolutionError("Choose a synthetic zone, access point, or coordinates.")

        if candidates.empty:
            raise LocationResolutionError(
                "No synthetic access point matched the requested location."
            )
        chosen = candidates.sort_values("wifi_id").iloc[0]
        if latitude is not None and longitude is not None and distance is None:
            distance = float(
                haversine_km(
                    float(latitude),
                    float(longitude),
                    np.array([float(chosen["latitude"])]),
                    np.array([float(chosen["longitude"])]),
                )[0]
            )
        return ResolvedLocation(
            access_point_id=str(chosen["wifi_id"]),
            zone_id=str(chosen["zone_id"]),
            zone_name=str(chosen["zone_name"]),
            zone_type=str(chosen["zone_type"]),
            latitude=float(chosen["latitude"]),
            longitude=float(chosen["longitude"]),
            access_point_capacity=float(chosen["access_point_capacity"]),
            distance_to_ap_km=distance,
            selection_method=method,
        )
