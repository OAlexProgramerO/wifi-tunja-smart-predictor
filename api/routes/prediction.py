"""HTTP routes for health, model metadata, and demand prediction."""

from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionTraceability,
    PredictRequest,
    PredictResponse,
    ScenarioPredictionResponse,
    ScenarioPredictRequest,
)
from wifi_tunja_smart_predictor import deployment
from wifi_tunja_smart_predictor.config import (
    CLASSIFICATION_METADATA_PATH,
    MODEL_INPUT_COLUMNS,
    MODEL_METADATA_PATH,
    PROJECT_NAME,
    PROJECT_VERSION,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.data.loader import load_raw_dataset
from wifi_tunja_smart_predictor.exceptions import (
    LocationResolutionError,
    ModelArtifactError,
    ModelMetadataError,
    ModelNotFoundError,
    PredictionError,
    ScenarioBuildError,
)
from wifi_tunja_smart_predictor.models.artifacts import (
    load_classifier_metadata,
    load_regression_metadata,
    load_regression_model,
)
from wifi_tunja_smart_predictor.models.predict import (
    classifier_supports_demand_labels,
    load_model,
    load_model_artifact,
    predict_demand,
    predict_probability,
)
from wifi_tunja_smart_predictor.models.traceability import build_prediction_traceability
from wifi_tunja_smart_predictor.observability import (
    current_request_id,
    emit_event,
    exception_diagnostics,
)
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService

router = APIRouter()
logger = logging.getLogger(__name__)


def _prediction_event(
    operation: str,
    outcome: str,
    started: float,
    *,
    error_category: str | None = None,
    exception_type: str | None = None,
    stack_frames: list[str] | None = None,
    model_version: str | None = None,
    selected_model: str | None = None,
) -> None:
    emit_event(
        logger,
        f"prediction.{operation}.{outcome}",
        level=logging.ERROR if outcome == "failed" else logging.INFO,
        request_id=current_request_id(),
        operation=operation,
        outcome=outcome,
        duration_ms=round((perf_counter() - started) * 1000, 3),
        error_category=error_category,
        exception_type=exception_type,
        stack_frames=stack_frames,
        model_version=model_version,
        selected_model=selected_model,
    )


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
    return load_classifier_metadata(MODEL_METADATA_PATH, CLASSIFICATION_METADATA_PATH)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(project=PROJECT_NAME, version=PROJECT_VERSION)


@router.get("/ready")
def readiness() -> JSONResponse:
    """Report whether required runtime files and model artifacts are usable."""
    if not deployment.runtime_assets_available() or not deployment.model_artifacts_available():
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Required synthetic data or model artifacts are unavailable.",
            },
        )
    return JSONResponse(status_code=200, content={"status": "ready"})


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    meta = {}
    loaded = False
    try:
        meta = _metadata()
        loaded_model = load_model()
        loaded = (
            callable(getattr(loaded_model, "predict", None))
            and callable(getattr(loaded_model, "predict_proba", None))
            and classifier_supports_demand_labels(loaded_model)
        )
    except (ModelArtifactError, ModelMetadataError, ModelNotFoundError, OSError, ValueError):
        pass
    regression_meta = {}
    try:
        regression_meta = load_regression_metadata()
        regression_model = load_regression_model()
        regression_loaded = callable(getattr(regression_model, "predict", None))
        if loaded:
            combined_regression = meta.get("regression")
            regression_loaded = regression_loaded and isinstance(combined_regression, dict)
            if isinstance(combined_regression, dict):
                regression_loaded = regression_loaded and (
                    combined_regression.get("project_version") == regression_meta["project_version"]
                    and combined_regression.get("selected_model")
                    == regression_meta["selected_model"]
                    and combined_regression.get("feature_count") == regression_meta["feature_count"]
                    and combined_regression.get("feature_names") == regression_meta["feature_names"]
                )
    except (ModelArtifactError, ModelMetadataError, ModelNotFoundError, OSError, ValueError):
        regression_loaded = False
    return ModelInfoResponse(
        model_loaded=loaded,
        project_version=PROJECT_VERSION,
        model_version=meta.get("project_version") if loaded else None,
        selected_model=meta.get("selected_model"),
        feature_count=len(MODEL_INPUT_COLUMNS),
        disclaimer=SYNTHETIC_DATA_DISCLAIMER,
        train_period=meta.get("train_period"),
        validation_period=meta.get("validation_period"),
        test_period=meta.get("test_period"),
        regression_model_loaded=regression_loaded,
        regression_model=regression_meta.get("selected_model") if regression_loaded else None,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    started = perf_counter()
    try:
        meta = _metadata()
        loaded_artifact = load_model_artifact()
        model = loaded_artifact.estimator
        features = payload.model_dump()
        labels = predict_demand(features, model=model)
        probabilities = predict_probability(features, model=model)
    except ModelNotFoundError as exc:
        _prediction_event("standard", "failed", started, error_category="model_unavailable")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Required model artifacts are unavailable.",
            },
        ) from exc
    except ModelArtifactError as exc:
        _prediction_event("standard", "failed", started, error_category="model_unavailable")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Required model artifacts are unavailable.",
            },
        ) from exc
    except ModelMetadataError as exc:
        _prediction_event(
            "standard", "failed", started, error_category="model_metadata_unavailable"
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_metadata_unavailable",
                "message": "Required model metadata is unavailable or invalid.",
            },
        ) from exc
    except PredictionError as exc:
        _prediction_event("standard", "failed", started, error_category="prediction_input_error")
        raise HTTPException(
            status_code=400,
            detail={
                "code": "prediction_input_error",
                "message": "Prediction inputs could not be processed.",
            },
        ) from exc
    except Exception as exc:
        _prediction_event(
            "standard",
            "failed",
            started,
            error_category="internal_error",
            **exception_diagnostics(exc),
        )
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check that the request matches the trained feature contract.",
        ) from None

    selected_model = meta.get("selected_model")
    _prediction_event(
        "standard",
        "completed",
        started,
        model_version=meta["project_version"],
        selected_model=selected_model if isinstance(selected_model, str) else None,
    )
    proba = probabilities[0] if probabilities else None
    return PredictResponse(
        prediction=labels[0],  # type: ignore[arg-type]
        prediction_probability=proba,
        # The artifact can outlive the application image in persisted volumes.
        # Report the version recorded at training, as scenario predictions do.
        model_version=meta["project_version"],
        selected_model=selected_model,
        disclaimer=SYNTHETIC_DATA_DISCLAIMER,
        traceability=PredictionTraceability(
            **build_prediction_traceability(
                classifier_metadata=meta,
                classifier_artifact=loaded_artifact,
            ),
            request_id=current_request_id(),
        ),
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
    started = perf_counter()
    try:
        result = _scenario_service().predict(request)
        result_payload = result.to_dict()
        traceability = result_payload.get("traceability")
        if isinstance(traceability, dict):
            traceability["request_id"] = current_request_id()
        classification = result_payload.get("classification", {})
        model_version = classification.get("model_version")
        selected_model = classification.get("selected_model")
        _prediction_event(
            "scenario",
            "completed",
            started,
            model_version=model_version if isinstance(model_version, str) else None,
            selected_model=selected_model if isinstance(selected_model, str) else None,
        )
        return ScenarioPredictionResponse(**result_payload)
    except ModelNotFoundError as exc:
        _prediction_event("scenario", "failed", started, error_category="model_unavailable")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Required model artifacts are unavailable.",
            },
        ) from exc
    except (ModelArtifactError, ModelMetadataError) as exc:
        _prediction_event("scenario", "failed", started, error_category="model_unavailable")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Required model artifacts or metadata are unavailable.",
            },
        ) from exc
    except LocationResolutionError as exc:
        _prediction_event("scenario", "failed", started, error_category="unknown_location")
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unknown_location",
                "message": "Location is not in the synthetic catalog.",
            },
        ) from exc
    except ScenarioBuildError as exc:
        _prediction_event("scenario", "failed", started, error_category="scenario_unavailable")
        raise HTTPException(
            status_code=422,
            detail={
                "code": "scenario_unavailable",
                "message": "The requested scenario is unavailable.",
            },
        ) from exc
    except PredictionError as exc:
        _prediction_event("scenario", "failed", started, error_category="prediction_failed")
        raise HTTPException(
            status_code=503,
            detail={
                "code": "prediction_failed",
                "message": "Scenario prediction could not be completed.",
            },
        ) from exc
    except Exception as exc:
        _prediction_event(
            "scenario",
            "failed",
            started,
            error_category="internal_error",
            **exception_diagnostics(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "scenario_prediction_failed",
                "message": "Scenario prediction could not be completed.",
            },
        ) from None
