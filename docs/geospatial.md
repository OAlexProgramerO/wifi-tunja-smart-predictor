# Geospatial resolution

The geospatial catalog is derived from simulated AP IDs, zone labels, capacities, and coordinates in the project dataset. Friendly zone names and known aliases resolve deterministically. Coordinates resolve to the nearest synthetic AP with the Haversine distance, returned in kilometers.

The Streamlit map uses Plotly with OpenStreetMap tiles and can select AP markers. The Geographic Analysis page also filters the synthetic observation timeline and aggregates row count, mean estimated next-hour connections, HIGH-demand share, and synthetic AP count by zone. These are descriptive summaries of generated rows, not forecasts. Map tiles require internet access from the browser, but no paid map key is configured. The map is a visualization of synthetic points; it does not show actual municipal WiFi infrastructure. No external geocoding request is made.

The main API `GET /locations` exposes the compact AP catalog, not observation-level records. Coordinate requests must include both latitude and longitude and pass geographic bounds.
