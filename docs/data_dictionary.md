# Data dictionary

**The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.** Examples are illustrative of the schema, not real observations.

Role values: `identifier` | `feature` | `historical_feature` | `target`.

`historical_feature` columns are available at prediction time (strictly before `timestamp`, or from the hour that just ended for network snapshots listed under Network).

## Source columns (`data/raw/wifi_tunja_public.csv`)

| column | data_type | description | example | role |
| --- | --- | --- | --- | --- |
| timestamp | datetime | Prediction time (start of the hour being classified) | 2024-03-12 14:00:00 | identifier |
| wifi_id | string | Synthetic access-point id | WIFI_07 | identifier |
| zone_id | string | Synthetic zone id | ZONE_02 | identifier |
| zone_name | string | Synthetic zone label (not a real venue) | Synthetic Campus North | identifier |
| zone_type | string | Synthetic land-use type | UNIVERSITY | feature |
| latitude | float | Simulated latitude | 5.5353 | identifier |
| longitude | float | Simulated longitude | -73.3678 | identifier |
| altitude_m | float | Simulated altitude (m) | 2782 | identifier |
| date | date | Calendar date of timestamp | 2024-03-12 | identifier |
| year | int | Year | 2024 | identifier |
| month | int | Month 1–12 (also used to build cyclical features) | 3 | feature |
| day | int | Day of month | 12 | identifier |
| day_of_week | int | Monday=0 … Sunday=6 | 1 | feature |
| day_name | string | Weekday name | TUESDAY | identifier |
| week_of_year | int | ISO week | 11 | identifier |
| hour | int | Hour 0–23 | 14 | feature |
| minute | int | Minute (hourly simulation, typically 0) | 0 | identifier |
| is_weekend | int | 1 if Saturday/Sunday | 0 | feature |
| is_holiday | int | 1 if simulated public holiday | 0 | feature |
| is_working_day | int | 1 if simulated working day | 1 | feature |
| is_school_day | int | 1 if simulated school/university term day | 1 | feature |
| time_period | string | NIGHT / MORNING / AFTERNOON / EVENING | AFTERNOON | feature |
| temperature_c | float | Simulated temperature | 13.4 | feature |
| humidity_percent | float | Simulated humidity | 72.1 | feature |
| precipitation_mm | float | Simulated precipitation | 0.2 | feature |
| wind_speed_kmh | float | Simulated wind speed | 8.5 | feature |
| weather_condition | string | CLEAR, CLOUDY, RAIN, HEAVY_RAIN, FOG | CLOUDY | feature |
| estimated_people_nearby | float | Simulated nearby population proxy | 640 | feature |
| traffic_level | string | LOW / MEDIUM / HIGH | MEDIUM | feature |
| public_transport_activity | float | Simulated transit activity 0–100 | 41.2 | feature |
| nearby_business_activity | float | Simulated business activity 0–100 | 55.0 | feature |
| nearby_student_population | float | Simulated nearby students | 210 | feature |
| nearby_worker_population | float | Simulated nearby workers | 180 | feature |
| event_nearby | int | 1 if a simulated event overlaps the hour | 0 | feature |
| event_type | string | NONE or event category | NONE | feature |
| estimated_event_attendance | float | Simulated attendance | 0 | feature |
| access_point_capacity | int | Simulated device capacity | 150 | feature |
| connected_devices | float | Snapshot: devices after the previous hour | 62 | feature |
| active_sessions | float | Snapshot: sessions after the previous hour | 48 | feature |
| average_session_duration_min | float | Snapshot | 18.4 | feature |
| bandwidth_usage_mbps | float | Snapshot | 92.0 | feature |
| packet_loss_percent | float | Snapshot | 1.1 | feature |
| latency_ms | float | Snapshot | 24.0 | feature |
| signal_strength_dbm | float | Snapshot | -58 | feature |
| channel_utilization_percent | float | Snapshot | 44.0 | feature |
| network_uptime_percent | float | Snapshot | 99.2 | feature |
| connections_previous_hour | float | Connections in hour t−1 | 71 | historical_feature |
| connections_previous_day | float | Mean connections previous calendar day | 68.4 | historical_feature |
| connections_same_hour_previous_day | float | Connections at t−24h | 70 | historical_feature |
| connections_same_hour_previous_week | float | Connections at t−168h | 65 | historical_feature |
| average_connections_last_3_hours | float | Mean of hours t−1, t−2, t−3 | 69.1 | historical_feature |
| average_connections_last_24_hours | float | Mean of the previous 24 hours | 66.0 | historical_feature |
| average_connections_last_7_days | float | Mean of the previous 168 hours | 64.2 | historical_feature |
| connections_next_hour | int | **Future** connection count in [timestamp, timestamp+1h) | 80 | target |
| demand_level | string | **Primary** class: LOW or HIGH | HIGH | target |

## Engineered columns (VERSION 0.2)

| column | data_type | description | example | role |
| --- | --- | --- | --- | --- |
| hour_sin | float | Sine encoding of the hour of day | 0.5 | feature |
| hour_cos | float | Cosine encoding of the hour of day | -0.87 | feature |
| day_of_week_sin | float | Sine encoding of the day of week | 0.78 | feature |
| day_of_week_cos | float | Cosine encoding of the day of week | 0.62 | feature |
| month_sin | float | Sine encoding of the month | 1.0 | feature |
| month_cos | float | Cosine encoding of the month | 0.0 | feature |
| device_capacity_ratio | float | connected_devices / capacity | 0.41 | feature |
| session_device_ratio | float | active_sessions / connected_devices | 0.77 | feature |
| bandwidth_per_session | float | bandwidth / active_sessions | 1.9 | feature |
| network_stress_indicator | float | utilisation × packet-loss (fractions) | 0.005 | feature |
| recent_to_daily_ratio | float | previous hour / last-24h mean | 1.08 | feature |
| recent_to_weekly_ratio | float | previous hour / last-7d mean | 1.11 | feature |
