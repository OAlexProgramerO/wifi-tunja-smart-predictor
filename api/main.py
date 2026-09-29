"""FastAPI application for synthetic WiFi demand classification."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes.prediction import router as prediction_router
from wifi_tunja_smart_predictor.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.exceptions import ModelNotFoundError, PredictionError

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title=PROJECT_NAME,
    version=PROJECT_VERSION,
    description=(
        "REST API for LOW/HIGH synthetic WiFi demand classification. "
        + SYNTHETIC_DATA_DISCLAIMER
        + " Do not interpret outputs as forecasts of real municipal infrastructure."
    ),
)


@app.exception_handler(ModelNotFoundError)
def _missing_model(_: Request, exc: ModelNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(PredictionError)
def _bad_prediction(_: Request, exc: PredictionError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/")
def root() -> dict[str, str]:
    return {
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "docs": "/docs",
        "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
    }


app.include_router(prediction_router)
