"""HTTP routes for health, model metadata, and demand prediction."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException

from api.schemas import HealthResponse, ModelInfoResponse, PredictRequest, PredictResponse
from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    PROJECT_NAME,
    PROJECT_VERSION,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.exceptions import ModelNotFoundError, PredictionError
from wifi_tunja_smart_predictor.models.predict import (
    load_model,
    predict_demand,
    predict_probability,
)

router = APIRouter()
logger = logging.getLogger(__name__)


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
    )


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        model = load_model()
        features = payload.model_dump()
        labels = predict_demand(features, model=model)
        probabilities = predict_probability(features, model=model)
    except ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except PredictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
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
