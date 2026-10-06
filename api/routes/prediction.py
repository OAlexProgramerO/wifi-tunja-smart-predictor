"""HTTP routes for health, model metadata, and demand prediction."""

from __future__ import annotations

import json
import logging
from functools import lru_cache

from fastapi import APIRouter, HTTPException

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ScenarioPredictionResponse,
    ScenarioPredictRequest,
)
from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    PROJECT_NAME,
    PROJECT_VERSION,
    REGRESSION_MODEL_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.exceptions import (
    LocationResolutionError,
    ModelNotFoundError,
    PredictionError,
    ScenarioBuildError,
)
from wifi_tunja_smart_predictor.models.predict import (
    load_model,
    predict_demand,
    predict_probability,
)
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService

router = APIRouter()
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _scenario_service() -> ScenarioPredictionService:
    """Share model artifacts and source frame across requests in this process."""
    return ScenarioPredictionService()


@lru_cache(maxsize=1)
def _location_catalog() -> list[dict]:
    """Cache the small AP map catalog instead of exposing historical rows."""
    frame = load_raw_dataset()
    return (
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
        .to_dict(orient="records")
    )


def _metadata() -> dict:
    if MODEL_METADATA_PATH.is_file():
        return json.loads(MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    return {}


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(project=PROJECT_NAME, version=PROJECT_VERSION)


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    meta = _metadata()
    try:
        load_model()
        loaded = True
    except ModelNotFoundError:
        loaded = False
    return ModelInfoResponse(
        model_loaded=loaded,
        project_version=PROJECT_VERSION,
        selected_model=meta.get("selected_model"),
        feature_count=len(MODEL_INPUT_COLUMNS),
        disclaimer=SYNTHETIC_DATA_DISCLAIMER,
        train_period=meta.get("train_period"),
        validation_period=meta.get("validation_period"),
        test_period=meta.get("test_period"),
        regression_model_loaded=REGRESSION_MODEL_PATH.is_file(),
        regression_model=meta.get("regression", {}).get("selected_model"),
    )


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        model = load_model()
        features = payload.model_dump()
        labels = predict_demand(features, model=model)
        probabilities = predict_probability(features, model=model)
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Required model artifacts are unavailable.",
            },
        ) from exc
    except PredictionError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "prediction_input_error",
                "message": "Prediction inputs could not be processed.",
            },
        ) from exc
    except Exception:
        logger.exception("Prediction failed while applying the model pipeline.")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check that the request matches the trained feature contract.",
        ) from None

    meta = _metadata()
    proba = probabilities[0] if probabilities else None
    return PredictResponse(
        prediction=labels[0],  # type: ignore[arg-type]
        prediction_probability=proba,
        model_version=PROJECT_VERSION,
        selected_model=meta.get("selected_model"),
        disclaimer=SYNTHETIC_DATA_DISCLAIMER,
    )


@router.get("/locations")
def locations() -> dict:
    """Return simulated access-point metadata for map and selector components."""
    return {
        "locations": _location_catalog(),
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
        "source": "synthetic access-point catalog",
    }


@router.post("/scenario/predict", response_model=ScenarioPredictionResponse)
def scenario_predict(payload: ScenarioPredictRequest) -> ScenarioPredictionResponse:
    """Build a feature row from location/time and produce combined model estimates."""
    location = payload.location
    context = (
        payload.advanced_context.model_dump(exclude_none=True) if payload.advanced_context else {}
    )
    request = ScenarioRequest(
        datetime=payload.datetime,
        zone=location.zone,
        zone_id=location.zone_id,
        access_point_id=location.access_point_id,
        latitude=location.latitude,
        longitude=location.longitude,
        context_overrides=context,
    )
    try:
        result = _scenario_service().predict(request)
        return ScenarioPredictionResponse(**result.to_dict())
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Required model artifacts are unavailable.",
            },
        ) from exc
    except LocationResolutionError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unknown_location",
                "message": "Location is not in the synthetic catalog.",
            },
        ) from exc
    except ScenarioBuildError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "scenario_unavailable",
                "message": "The requested scenario is unavailable.",
            },
        ) from exc
    except PredictionError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "prediction_failed",
                "message": "Scenario prediction could not be completed.",
            },
        ) from exc
    except Exception:
        logger.exception("Scenario prediction failed in the model pipeline.")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "scenario_prediction_failed",
                "message": "Scenario prediction could not be completed.",
            },
        ) from None
