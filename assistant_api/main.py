"""Independent deterministic WiFi Tunja Assistant service (default port 8001)."""

from __future__ import annotations

import logging
from functools import lru_cache
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.security import install_api_security
from wifi_tunja_smart_predictor.assistant.queries import DatasetQuery
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest, ChatResponse
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.assistant.tools import dataset_query_tool
from wifi_tunja_smart_predictor.config import (
    MODEL_PATH,
    PROJECT_NAME,
    PROJECT_VERSION,
    RAW_DATASET_PATH,
    REGRESSION_MODEL_PATH,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.exceptions import ModelNotFoundError
from wifi_tunja_smart_predictor.observability import (
    current_request_id,
    emit_event,
    exception_diagnostics,
)

logger = logging.getLogger(__name__)
app = FastAPI(
    title=f"{PROJECT_NAME} Assistant",
    version=PROJECT_VERSION,
    description=(
        "Tool-grounded assistant for synthetic demand scenarios and dataset questions. "
        + SYNTHETIC_DATA_DISCLAIMER
    ),
)
install_api_security(app, max_body_bytes=64 * 1024, service="assistant_api")


@app.exception_handler(RequestValidationError)
def _invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"detail": {"code": "invalid_request", "message": "Request validation failed."}},
    )


@lru_cache(maxsize=1)
def _assistant() -> AssistantService:
    """Return the process-local tool dispatcher and lightweight session store."""
    return AssistantService()


@app.get("/health")
def health() -> dict[str, str | bool]:
    """Report assistant process health and deterministic provider status."""
    return {
        "status": "ok",
        "project": PROJECT_NAME,
        "version": PROJECT_VERSION,
        "llm_enabled": False,
    }


@app.get("/ready")
def readiness() -> JSONResponse:
    """Report whether the dataset and model resources used by chat are present."""
    if not (
        RAW_DATASET_PATH.is_file() and MODEL_PATH.is_file() and REGRESSION_MODEL_PATH.is_file()
    ):
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "message": "Required synthetic data or model artifacts are unavailable.",
            },
        )
    return JSONResponse(status_code=200, content={"status": "ready"})


@app.get("/suggestions")
def suggestions() -> dict[str, list[str]]:
    """Return supported questions for the assistant UI."""
    return {
        "questions": [
            "What is the expected demand downtown today at 6 PM?",
            "How many connections are expected?",
            "What are the main factors behind this prediction?",
            "How many synthetic access points are in the dataset?",
            "Which synthetic zone has historically shown higher demand at 7 PM?",
            "What is the test F1 score?",
            "Explain the dashboard.",
            "How is demand historically?",
            "When is demand usually highest?",
            "Compare two zones.",
            "Which zone has higher demand?",
            "What day has the highest demand?",
            "¿Cómo es históricamente la demanda?",
            "¿Cuándo suele ser más alta?",
            "Compara dos zonas.",
            "¿Qué zona tiene mayor demanda?",
            "¿Qué día tiene mayor demanda?",
        ]
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Answer with project tools and source metadata; no LLM or user code execution."""
    started = perf_counter()
    try:
        return _assistant().handle(request)
    except ModelNotFoundError as exc:
        emit_event(
            logger,
            "assistant.scenario.failed",
            level=logging.ERROR,
            request_id=current_request_id(),
            operation="scenario",
            outcome="failed",
            error_category="model_unavailable",
            exception_type=type(exc).__name__,
            duration_ms=round((perf_counter() - started) * 1000, 3),
        )
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Train both models before requesting a scenario.",
            },
        ) from exc
    except Exception as exc:
        emit_event(
            logger,
            "assistant.chat.failed",
            level=logging.ERROR,
            request_id=current_request_id(),
            operation="chat",
            outcome="failed",
            error_category="internal_error",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            **exception_diagnostics(exc),
        )
        raise HTTPException(
            status_code=500,
            detail={
                "code": "assistant_error",
                "message": "The assistant could not complete this request.",
            },
        ) from None


@app.post("/dataset/query")
def dataset_query(query: DatasetQuery) -> dict:
    """Run a validated allowlisted query for clients that need structured aggregates."""
    started = perf_counter()
    try:
        service = _assistant()
        result = dataset_query_tool(service.queries, query)
        emit_event(
            logger,
            "assistant.dataset_query.completed",
            request_id=current_request_id(),
            operation="dataset_query",
            outcome="completed",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            metric=query.metric,
            column=query.column,
            group_by=query.group_by,
        )
        return {
            "result": result,
            "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
            "query": query.model_dump(),
        }
    except ValueError as exc:
        emit_event(
            logger,
            "assistant.dataset_query.rejected",
            level=logging.WARNING,
            request_id=current_request_id(),
            operation="dataset_query",
            outcome="rejected",
            duration_ms=round((perf_counter() - started) * 1000, 3),
            error_category="unsupported_dataset_query",
            exception_type=type(exc).__name__,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_dataset_query",
                "message": "The dataset query contains an unsupported filter or value.",
            },
        ) from exc
