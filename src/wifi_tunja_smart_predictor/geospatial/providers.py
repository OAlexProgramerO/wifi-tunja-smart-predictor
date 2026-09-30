"""Map provider contracts and a no-paid-key Plotly/OpenStreetMap adapter."""

from __future__ import annotations

from typing import Protocol

import pandas as pd
import plotly.express as px


class MapProvider(Protocol):
    """Interface for creating a map from synthetic access-point metadata."""

    def build_map(self, points: pd.DataFrame): ...


class PlotlyOpenStreetMapProvider:
    """Create an interactive map with open street tiles and selectable AP markers."""

    def build_map(self, points: pd.DataFrame):
        """Build the map; tile availability depends on the user's internet connection."""
        figure = px.scatter_map(
            points,
            lat="latitude",
            lon="longitude",
            color="zone_type",
            hover_name="wifi_id",
            hover_data=["zone_id", "zone_name", "access_point_capacity"],
            custom_data=["wifi_id"],
            map_style="open-street-map",
            zoom=12,
            height=560,
        )
        figure.update_layout(margin={"l": 0, "r": 0, "t": 20, "b": 0})
        return figure
