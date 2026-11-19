"""Synchronous lifecycle API for reproducible synthetic ML experiments."""

from __future__ import annotations

import sqlite3
from functools import lru_cache

from fastapi import APIRouter, Header, HTTPException, Path, Query, Response
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
from wifi_tunja_smart_predictor.experiments.ml_gates_schemas import (
    MLExperimentGateRequest,
    MLExperimentGateResponse,
)
from wifi_tunja_smart_predictor.experiments.ml_gates_service import (
    MLExperimentGateNotFoundError,
    MLExperimentGateService,
)
from wifi_tunja_smart_predictor.experiments.ml_orchestrator import MLExperimentOrchestrator
from wifi_tunja_smart_predictor.experiments.ml_policy_schemas import (
    MLEvaluationPolicyCreate,
    MLEvaluationPolicyPage,
    MLEvaluationPolicyRecord,
    MLEvaluationPolicyRunRequest,
    MLEvaluationPolicyRunResponse,
    MLEvaluationPolicyVersionPage,
)
from wifi_tunja_smart_predictor.experiments.ml_policy_service import (
    MLEvaluationPolicyNotFoundError,
    MLEvaluationPolicyService,
)
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


@router.post("/gates", response_model=MLExperimentGateResponse)
def evaluate_ml_experiment_gates(payload: MLExperimentGateRequest) -> dict:
    try:
        return MLExperimentGateService(_orchestrator().store).evaluate(
            payload.experiment_ids,
            payload.task,
            [criterion.model_dump() for criterion in payload.criteria],
        )
    except MLExperimentGateNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail="Experiment evaluation is unavailable."
        ) from exc


@router.post("/policies", status_code=201, response_model=MLEvaluationPolicyRecord)
def create_ml_evaluation_policy(payload: MLEvaluationPolicyCreate) -> dict:
    criteria = {
        "classification": [item.model_dump() for item in (payload.classification_criteria or [])],
        "regression": [item.model_dump() for item in (payload.regression_criteria or [])],
    }
    try:
        return MLEvaluationPolicyService(_orchestrator().store).create_version(
            policy_id=payload.policy_id,
            name=payload.name,
            description=payload.description,
            tags=payload.tags,
            criteria=criteria,
        )
    except MLEvaluationPolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail="Evaluation policy storage is unavailable."
        ) from exc


@router.get("/policies", response_model=MLEvaluationPolicyPage)
def list_ml_evaluation_policies(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
) -> dict:
    try:
        items, total = MLEvaluationPolicyService(_orchestrator().store).list_policies(
            limit=limit, offset=offset
        )
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail="Evaluation policy storage is unavailable."
        ) from exc
    return {"items": items, "limit": limit, "offset": offset, "total_count": total}


@router.get("/policies/{policy_id}", response_model=MLEvaluationPolicyRecord)
def get_ml_evaluation_policy(policy_id: str, version: int | None = Query(None, ge=1)) -> dict:
    try:
        return MLEvaluationPolicyService(_orchestrator().store).get_policy(policy_id, version)
    except MLEvaluationPolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail="Evaluation policy storage is unavailable."
        ) from exc


@router.get("/policies/{policy_id}/versions", response_model=MLEvaluationPolicyVersionPage)
def list_ml_evaluation_policy_versions(
    policy_id: str,
    limit: int = Query(100, ge=1, le=100),
    offset: int = Query(0, ge=0, le=100_000),
) -> dict:
    try:
        items, total = MLEvaluationPolicyService(_orchestrator().store).list_versions(
            policy_id, limit=limit, offset=offset
        )
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=503, detail="Evaluation policy storage is unavailable."
        ) from exc
    if total == 0:
        raise HTTPException(status_code=404, detail="Evaluation policy was not found.")
    return {"items": items, "limit": limit, "offset": offset, "total_count": total}


@router.post(
    "/policies/{policy_id}/versions/{version}/evaluate",
    response_model=MLEvaluationPolicyRunResponse,
)
def evaluate_ml_experiment_with_policy(
    policy_id: str,
    version: int = Path(ge=1),
    payload: MLEvaluationPolicyRunRequest = ...,
) -> dict:
    try:
        return MLEvaluationPolicyService(_orchestrator().store).evaluate(
            policy_id, version, payload.experiment_id
        )
    except MLEvaluationPolicyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise HTTPException(status_code=503, detail="Policy evaluation is unavailable.") from exc


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
