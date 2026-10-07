"""Independent deterministic WiFi Tunja Assistant service (default port 8001)."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.security import install_api_security
from wifi_tunja_smart_predictor.assistant.queries import DatasetQuery
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest, ChatResponse
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.assistant.tools import dataset_query_tool
from wifi_tunja_smart_predictor.config import (
    PROJECT_NAME,
    PROJECT_VERSION,
    SYNTHETIC_DATA_DISCLAIMER,
)
from wifi_tunja_smart_predictor.exceptions import ModelNotFoundError

logger = logging.getLogger(__name__)
app = FastAPI(
    title=f"{PROJECT_NAME} Assistant",
    version=PROJECT_VERSION,
    description=(
        "Tool-grounded assistant for synthetic demand scenarios and dataset questions. "
        + SYNTHETIC_DATA_DISCLAIMER
    ),
)
install_api_security(app, max_body_bytes=64 * 1024)


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
    try:
        return _assistant().handle(request)
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": "Train both models before requesting a scenario.",
            },
        ) from exc
    except Exception:
        logger.exception("Assistant request failed while executing a project tool.")
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
    try:
        service = _assistant()
        return {
            "result": dataset_query_tool(service.queries, query),
            "disclaimer": SYNTHETIC_DATA_DISCLAIMER,
            "query": query.model_dump(),
        }
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "unsupported_dataset_query",
                "message": "The dataset query contains an unsupported filter or value.",
            },
        ) from exc
