"""Pydantic contracts for the prediction API.

Targets ``demand_level`` and ``connections_next_hour`` are intentionally absent.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ZoneType = Literal[
    "DOWNTOWN",
    "UNIVERSITY",
    "COMMERCIAL",
    "PARK",
    "TRANSPORT",
    "RESIDENTIAL",
    "PUBLIC_SERVICE",
    "HEALTHCARE",
    "EDUCATION",
    "TOURISM",
]
TimePeriod = Literal["NIGHT", "MORNING", "AFTERNOON", "EVENING"]
WeatherCondition = Literal["CLEAR", "CLOUDY", "RAIN", "HEAVY_RAIN", "FOG"]
TrafficLevel = Literal["LOW", "MEDIUM", "HIGH"]
EventType = Literal[
    "NONE",
    "ACADEMIC",
    "SPORT",
    "CULTURAL",
    "MUSIC",
    "COMMUNITY",
    "GOVERNMENT",
]
DemandLevel = Literal["LOW", "HIGH"]


class HealthResponse(BaseModel):
    status: str = "ok"
    project: str
    version: str


class ModelInfoResponse(BaseModel):
    model_loaded: bool
    project_version: str
    model_version: str | None = Field(
        default=None,
        description="Version recorded in the loaded classifier artifact metadata.",
    )
    selected_model: str | None = None
    feature_count: int
    disclaimer: str
    train_period: dict | None = None
    validation_period: dict | None = None
    test_period: dict | None = None
    regression_model_loaded: bool = False
    regression_model: str | None = None


class ModelDiagnosticsResponse(BaseModel):
    """Versioned, provenance-checked diagnostics computed from synthetic data."""

    schema_version: str
    generated_at_utc: datetime
    project_version: str
    model_version: str | None
    dataset: dict[str, Any]
    data_quality: dict[str, Any]
    artifacts: dict[str, Any]
    method: dict[str, Any]
    test_period: dict[str, Any]
    test: dict[str, Any]
    rolling_windows: list[dict[str, Any]]
    warnings: list[str]


class TemporalEvaluationRequest(BaseModel):
    """Bounded settings for additional synthetic chronological evaluation."""

    model_config = ConfigDict(extra="forbid")
    window_count: int = Field(default=3, ge=2, le=5)
    min_train_rows: int = Field(default=100, ge=2, le=50_000)
    min_test_rows: int = Field(default=20, ge=2, le=10_000)
    features: list[str] | None = None

    @field_validator("features")
    @classmethod
    def validate_features(cls, value: list[str] | None) -> list[str] | None:
        if value is not None and (not 1 <= len(value) <= 12 or len(set(value)) != len(value)):
            raise ValueError("features must contain 1 to 12 unique values")
        return value


class TemporalEvaluationResponse(BaseModel):
    """Versioned synthetic temporal evaluation report."""

    schema_version: str
    project_version: str
    feature_version: str | None
    generated_at_utc: datetime
    evaluation_type: str
    parameters: dict[str, Any]
    dataset: dict[str, Any]
    artifacts: dict[str, Any]
    windows: list[dict[str, Any]]
    feature_diagnostics: list[dict[str, Any]]
    summary: dict[str, Any]


class ModelArtifactTrace(BaseModel):
    model_version: str | None = None
    selected_model: str | None = None
    artifact_sha256: str | None = None
    artifact_size_bytes: int | None = None


class PredictionTraceability(BaseModel):
    prediction_id: str
    predicted_at_utc: datetime
    application_version: str
    request_id: str | None = None
    classifier: ModelArtifactTrace
    regressor: ModelArtifactTrace | None = None


class PredictRequest(BaseModel):
    """Fields required by the persisted training pipeline (pre-engineering)."""

    model_config = ConfigDict(extra="forbid")

    zone_type: ZoneType = Field(description="Synthetic land-use type of the access point.")
    month: int = Field(ge=1, le=12)
    day_of_week: int = Field(ge=0, le=6, description="Monday=0 ... Sunday=6.")
    hour: int = Field(ge=0, le=23, description="Prediction hour (start of the forecast hour).")
    is_weekend: int = Field(ge=0, le=1)
    is_holiday: int = Field(ge=0, le=1)
    is_working_day: int = Field(ge=0, le=1)
    is_school_day: int = Field(ge=0, le=1)
    time_period: TimePeriod
    temperature_c: float = Field(ge=-10, le=40)
    humidity_percent: float = Field(ge=0, le=100)
    precipitation_mm: float = Field(ge=0, le=100)
    wind_speed_kmh: float = Field(ge=0, le=150)
    weather_condition: WeatherCondition
    estimated_people_nearby: float = Field(ge=0, le=20_000)
    traffic_level: TrafficLevel
    public_transport_activity: float = Field(ge=0, le=100)
    nearby_business_activity: float = Field(ge=0, le=100)
    nearby_student_population: float = Field(ge=0, le=20_000)
    nearby_worker_population: float = Field(ge=0, le=20_000)
    event_nearby: int = Field(ge=0, le=1)
    event_type: EventType
    estimated_event_attendance: float = Field(ge=0, le=20_000)
    access_point_capacity: float = Field(gt=0, le=500)
    connected_devices: float = Field(ge=0, le=500)
    active_sessions: float = Field(ge=0, le=500)
    average_session_duration_min: float = Field(ge=0, le=240)
    bandwidth_usage_mbps: float = Field(ge=0, le=2000)
    packet_loss_percent: float = Field(ge=0, le=100)
    latency_ms: float = Field(ge=0, le=5000)
    signal_strength_dbm: float = Field(ge=-120, le=-10)
    channel_utilization_percent: float = Field(ge=0, le=100)
    network_uptime_percent: float = Field(ge=0, le=100)
    connections_previous_hour: float = Field(ge=0, le=2000)
    connections_previous_day: float = Field(ge=0, le=2000)
    connections_same_hour_previous_day: float = Field(ge=0, le=2000)
    connections_same_hour_previous_week: float = Field(ge=0, le=2000)
    average_connections_last_3_hours: float = Field(ge=0, le=2000)
    average_connections_last_24_hours: float = Field(ge=0, le=2000)
    average_connections_last_7_days: float = Field(ge=0, le=2000)


class PredictResponse(BaseModel):
    prediction: DemandLevel
    prediction_probability: dict[str, float] | None = Field(
        default=None,
        description="Class probabilities when the estimator implements predict_proba.",
    )
    model_version: str
    selected_model: str | None = None
    disclaimer: str
    traceability: PredictionTraceability | None = None


class ScenarioLocationRequest(BaseModel):
    """Synthetic area, AP, or coordinate input; coordinates must be paired."""

    model_config = ConfigDict(extra="forbid")

    zone: str | None = Field(default=None, min_length=1, max_length=100)
    zone_id: str | None = Field(default=None, min_length=1, max_length=40)
    access_point_id: str | None = Field(default=None, min_length=1, max_length=40)
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @model_validator(mode="after")
    def validate_location(self) -> ScenarioLocationRequest:
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError("latitude and longitude must be provided together")
        selectors = [
            self.zone is not None,
            self.zone_id is not None,
            self.access_point_id is not None,
            self.latitude is not None,
        ]
        if sum(selectors) == 0:
            raise ValueError("provide a zone, zone_id, access_point_id, or coordinate pair")
        if sum(selectors) > 1:
            raise ValueError("choose one location selector to avoid ambiguous resolution")
        return self


class ScenarioAdvancedContext(BaseModel):
    """Small optional context override set; all other fields come from analogs."""

    model_config = ConfigDict(extra="forbid")

    temperature_c: float | None = Field(default=None, ge=-10, le=40)
    humidity_percent: float | None = Field(default=None, ge=0, le=100)
    precipitation_mm: float | None = Field(default=None, ge=0, le=100)
    wind_speed_kmh: float | None = Field(default=None, ge=0, le=150)
    weather_condition: WeatherCondition | None = None
    traffic_level: TrafficLevel | None = None
    event_nearby: int | None = Field(default=None, ge=0, le=1)
    event_type: EventType | None = None
    estimated_event_attendance: float | None = Field(default=None, ge=0, le=20_000)
    estimated_people_nearby: float | None = Field(default=None, ge=0, le=20_000)


class ScenarioPredictRequest(BaseModel):
    """Simple user scenario; no target or technical feature vector is accepted."""

    model_config = ConfigDict(extra="forbid")

    location: ScenarioLocationRequest
    datetime: datetime
    advanced_context: ScenarioAdvancedContext | None = None


class ScenarioPredictionResponse(BaseModel):
    """Combined outputs with model-derived estimates and synthetic-data notice."""

    scenario: dict[str, Any]
    classification: dict[str, Any]
    regression: dict[str, Any]
    capacity: dict[str, Any]
    explanation: dict[str, Any]
    traceability: PredictionTraceability | None = None
    disclaimer: str


class PlanningZoneRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zone_id: str = Field(min_length=1, max_length=40)
    capacity_connections: float = Field(ge=0, le=100_000)


class PlanningScenarioRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    datetime: datetime
    zones: list[PlanningZoneRequest] = Field(min_length=1, max_length=12)
    resource_budget_connections: float = Field(default=0, ge=0, le=100_000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("name must contain a non-space character")
        return value.strip()

    @model_validator(mode="after")
    def unique_zones(self) -> PlanningScenarioRequest:
        ids = [zone.zone_id for zone in self.zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone_id values must be unique")
        return self


class ExperimentComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_ids: list[str] = Field(min_length=2, max_length=10)

    @model_validator(mode="after")
    def unique_ids(self) -> ExperimentComparisonRequest:
        if len(self.experiment_ids) != len(set(self.experiment_ids)):
            raise ValueError("experiment_ids must be unique")
        return self


class ExperimentIntegrity(BaseModel):
    status: Literal["verified", "incomplete", "invalid"]
    assessment_status: Literal["assessable", "not_assessable"]
    reason_codes: list[str]


class ExperimentIntegrityWarning(BaseModel):
    experiment_id: str
    reason_code: str
    message: str


class ExperimentRecord(BaseModel):
    id: str
    created_at_utc: datetime
    updated_at_utc: datetime
    status: Literal["COMPLETED"]
    experiment_type: Literal["capacity_scenario"]
    name: str
    schema_version: str
    configuration: dict[str, Any]
    result: dict[str, Any]
    integrity: ExperimentIntegrity


class ExperimentPage(BaseModel):
    items: list[ExperimentRecord]
    limit: int
    offset: int
    integrity_warnings: list[ExperimentIntegrityWarning] = Field(default_factory=list)
