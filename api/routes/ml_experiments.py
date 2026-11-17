"""Synchronous lifecycle API for reproducible synthetic ML experiments."""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from fastapi import APIRouter, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse

from wifi_tunja_smart_predictor.config import experiment_database_path
from wifi_tunja_smart_predictor.experiments.ml_comparison_schemas import (
    MLExperimentComparisonRequest,
    MLExperimentComparisonResponse,
)
from wifi_tunja_smart_predictor.experiments.ml_comparison_service import (
    MLExperimentComparisonService,
    MLExperimentNotFoundError,
)
from wifi_tunja_smart_predictor.experiments.ml_orchestrator import MLExperimentOrchestrator
from wifi_tunja_smart_predictor.experiments.ml_schemas import (
    MLExperimentCreate,
    MLExperimentPage,
    MLExperimentRecord,
)
from wifi_tunja_smart_predictor.experiments.ml_store import IdempotencyConflict, MLExperimentStore

router = APIRouter(prefix="/ml-experiments", tags=["ml-experiments"])


@lru_cache(maxsize=1)
def _orchestrator() -> MLExperimentOrchestrator:
    return MLExperimentOrchestrator(MLExperimentStore(experiment_database_path()))


@router.post("", status_code=201, response_model=MLExperimentRecord)
def create_ml_experiment(
    payload: MLExperimentCreate,
    response: Response,
    idempotency_key: str = Header(alias="X-Idempotency-Key", min_length=8, max_length=128),
) -> dict:
    try:
        record = _orchestrator().create(payload, idempotency_key)
    except IdempotencyConflict as exc:
        raise HTTPException(
            status_code=409, detail="Idempotency key conflicts with a saved run."
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail="Experiment persistence is unavailable."
        ) from exc
    if record.get("replayed"):
        response.status_code = 200
    return record


@router.get("", response_model=MLExperimentPage)
def list_ml_experiments(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
) -> dict:
    try:
        items, total = _orchestrator().list_page(limit=limit, offset=offset)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Experiment history is unavailable.") from exc
    return {"items": items, "limit": limit, "offset": offset, "total_count": total}


@router.post("/compare", response_model=MLExperimentComparisonResponse)
def compare_ml_experiments(payload: MLExperimentComparisonRequest) -> dict:
    try:
        return MLExperimentComparisonService(_orchestrator().store).compare(payload.experiment_ids)
    except MLExperimentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail="Experiment comparison is unavailable."
        ) from exc


@router.get("/{experiment_id}/report")
def get_ml_experiment_report(experiment_id: str) -> dict:
    try:
        report = _orchestrator().report(experiment_id)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Experiment history is unavailable.") from exc
    if report is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return JSONResponse(
        content=report, headers={"Content-Disposition": "attachment; filename=ml-experiment.json"}
    )


@router.get("/{experiment_id}", response_model=MLExperimentRecord)
def get_ml_experiment(experiment_id: str) -> dict:
    try:
        record = _orchestrator().get(experiment_id)
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Experiment history is unavailable.") from exc
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return record
