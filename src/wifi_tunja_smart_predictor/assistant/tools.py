"""Structured project tools used by the deterministic assistant."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from wifi_tunja_smart_predictor.assistant.queries import DatasetQuery, DatasetQueryEngine
from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    MODEL_METADATA_PATH,
    PROJECT_VERSION,
    REGRESSION_METADATA_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.geospatial.locations import LocationResolver
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService


def prediction_tool(service: ScenarioPredictionService, request: ScenarioRequest) -> dict[str, Any]:
    """Invoke the shared scenario prediction engine and return its structured result."""
    return service.predict(request).to_dict()


def location_tool(resolver: LocationResolver, zone: str) -> dict[str, Any]:
    """Resolve a user-provided synthetic zone label to its representative AP."""
    return resolver.resolve(zone=zone).to_dict()


def dataset_query_tool(engine: DatasetQueryEngine, query: DatasetQuery) -> Any:
    """Execute a Pydantic-validated dataset aggregate."""
    return engine.execute(query)


def dashboard_summary_tool(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize the data behind the dashboard's main overview."""
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "access_points": int(frame["wifi_id"].nunique()),
        "synthetic_zones": int(frame["zone_id"].nunique()),
        "date_start": str(pd.to_datetime(frame["timestamp"]).min()),
        "date_end": str(pd.to_datetime(frame["timestamp"]).max()),
        "high_share": float(frame["demand_level"].eq("HIGH").mean()),
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
    }


def model_info_tool() -> dict[str, Any]:
    """Read generated model metadata; missing metrics are explicitly unavailable."""
    combined = _read_json(MODEL_METADATA_PATH)
    classification = _read_json(CLASSIFICATION_METADATA_PATH) or combined.get("classification", {})
    regression = _read_json(REGRESSION_METADATA_PATH) or combined.get("regression", {})
    return {
        "model_version": PROJECT_VERSION,
        "classifier": classification.get("selected_model", "unavailable"),
        "regressor": regression.get("selected_model", "unavailable"),
        "classification_metrics": classification.get("test_metrics", {}),
        "regression_metrics": regression.get("test_metrics", {}),
        "test_period": classification.get("test_period"),
        "dataset_version": classification.get("dataset_version"),
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
    }


def historical_demand_tool(
    frame: pd.DataFrame,
    *,
    hour: int,
    zone_id: str | None = None,
    zone_type: str | None = None,
) -> dict[str, Any]:
    """Compare historical next-hour connection means using target only as an outcome."""
    selected = frame.loc[frame["hour"].eq(hour)]
    if zone_id:
        selected = selected.loc[selected["zone_id"].eq(zone_id)]
    if zone_type:
        selected = selected.loc[selected["zone_type"].eq(zone_type)]
    summary = (
        selected.groupby(["zone_id", "zone_name", "zone_type"], observed=True)
        .agg(
            mean_connections=("connections_next_hour", "mean"),
            high_share=("demand_level", lambda values: values.eq("HIGH").mean()),
            rows=("demand_level", "size"),
        )
        .reset_index()
        .sort_values(["mean_connections", "zone_name"], ascending=[False, True])
    )
    return {
        "hour": hour,
        "rows": summary.to_dict(orient="records"),
        "note": "Historical outcomes are queried for analysis only and are never scenario features.",
    }


def model_explanation_tool(prediction: dict[str, Any]) -> dict[str, Any]:
    """Return sensitivity factors from the actual scenario prediction result."""
    return prediction.get("explanation", {})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
