# API

Base URL (local): `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs`

**The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.** API outputs are classifications from a model trained on that simulation.

Machine-learning logic lives in `src/wifi_tunja_smart_predictor/models/predict.py`. Routes only validate HTTP payloads and return JSON.

## GET `/`

Service name, version, and disclaimer.

## GET `/health`

Liveness probe.

Example response:

```json
{
  "status": "ok",
  "project": "WiFi Tunja Smart Predictor",
  "version": "0.2.0"
}
```

## GET `/model-info`

Whether the joblib artefact is present, feature-contract size, optional training-period metadata from `models/model_metadata.json`.

Example response (shape):

```json
{
  "model_loaded": true,
  "project_version": "0.2.0",
  "selected_model": "random_forest",
  "feature_count": 40,
  "disclaimer": "The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.",
  "train_period": {"start": "...", "end": "...", "rows": 0, "cutoff": "2025-06-30 23:59:59"},
  "validation_period": {},
  "test_period": {}
}
```

`feature_count` is the number of **raw model input fields** (before cyclical encodings), matching `MODEL_INPUT_COLUMNS`.

## POST `/predict`

JSON body must include the feature contract. **Do not send** `demand_level` or `connections_next_hour`.

Pydantic rejects out-of-range `hour`, unknown `zone_type`, etc. (HTTP 422).

If the model file is missing: HTTP 503.

Example request (illustrative values):

```json
{
  "zone_type": "DOWNTOWN",
  "month": 3,
  "day_of_week": 2,
  "hour": 14,
  "is_weekend": 0,
  "is_holiday": 0,
  "is_working_day": 1,
  "is_school_day": 1,
  "time_period": "AFTERNOON",
  "temperature_c": 14.0,
  "humidity_percent": 70.0,
  "precipitation_mm": 0.0,
  "wind_speed_kmh": 8.0,
  "weather_condition": "CLOUDY",
  "estimated_people_nearby": 400,
  "traffic_level": "MEDIUM",
  "public_transport_activity": 40,
  "nearby_business_activity": 50,
  "nearby_student_population": 80,
  "nearby_worker_population": 200,
  "event_nearby": 0,
  "event_type": "NONE",
  "estimated_event_attendance": 0,
  "access_point_capacity": 150,
  "connected_devices": 60,
  "active_sessions": 45,
  "average_session_duration_min": 18,
  "bandwidth_usage_mbps": 90,
  "packet_loss_percent": 1.0,
  "latency_ms": 25,
  "signal_strength_dbm": -58,
  "channel_utilization_percent": 45,
  "network_uptime_percent": 99,
  "connections_previous_hour": 70,
  "connections_previous_day": 65,
  "connections_same_hour_previous_day": 68,
  "connections_same_hour_previous_week": 64,
  "average_connections_last_3_hours": 66,
  "average_connections_last_24_hours": 62,
  "average_connections_last_7_days": 60
}
```

Example response:

```json
{
  "prediction": "LOW",
  "prediction_probability": {"LOW": 0.72, "HIGH": 0.28},
  "model_version": "0.2.0",
  "selected_model": "random_forest",
  "disclaimer": "The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes."
}
```

`prediction` is `LOW` or `HIGH`. `prediction_probability` is omitted/null only if the estimator cannot produce probabilities. Probabilities are **not** claimed to be real-world confidence.
