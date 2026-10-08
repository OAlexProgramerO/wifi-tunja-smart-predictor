"""FastAPI application for synthetic WiFi demand classification."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.routes.prediction import router as prediction_router
from api.security import install_api_security
from wifi_tunja_smart_predictor.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.exceptions import ModelNotFoundError, PredictionError

app = FastAPI(
    title=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=(
        "REST API for synthetic WiFi demand classification and location/time scenario estimates. "
        + SYNTHETIC_DATA_DISCLAIMER
        + " Do not interpret outputs as forecasts of real municipal infrastructure."
    ),
)
install_api_security(app, max_body_bytes=256 * 1024, service="main_api")


@app.exception_handler(RequestValidationError)
def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": "invalid_request", "message": "Request validation failed."}},
    )


@app.exception_handler(ModelNotFoundError)
def _missing_model(_: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "detail": {
                "code": "model_unavailable",
                "message": "Required model artifacts are unavailable.",
            }
        },
    )


@app.exception_handler(PredictionError)
def _bad_prediction(_: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content={
            "detail": {
                "code": "prediction_input_error",
                "message": "Prediction inputs could not be processed.",
            }
        },
    )


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "docs": "/docs",
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
    }


app.include_router(prediction_router)
