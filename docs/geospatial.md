# Geospatial resolution

The geospatial catalog is derived from simulated AP IDs, zone labels, capacities, and coordinates in the project dataset. Friendly zone names and known aliases resolve deterministically. Coordinates resolve to the nearest synthetic AP with the Haversine distance, returned in kilometers.

The V6 Geospatial Intelligence page uses Plotly geographic rendering without external tiles or paid map keys and filters the synthetic observation timeline by inclusive dates, zone, hour, weekday, and demand class. Reusable analytics report observation counts, mean and peak **observed synthetic** next-hour connections, HIGH-class share, per-zone/access-point summaries, and zone-by-hour means. An AP point is omitted when its coordinates are missing or outside latitude/longitude bounds. These are descriptive summaries of generated rows, not forecasts; historical target values may not have been available at prediction time. The map shows simulated coordinates, not actual municipal WiFi infrastructure. No external geocoding or live basemap request is made. See the [scenario workbench guide](scenario-workbench.md) for formulas and limitations.

The main API `GET /locations` exposes the compact AP catalog, not observation-level records. Coordinate requests must include both latitude and longitude and pass geographic bounds.
