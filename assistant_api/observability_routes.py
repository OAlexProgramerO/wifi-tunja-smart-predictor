"""Read-only assistant-process metrics endpoint used by operational diagnostics."""

from __future__ import annotations

from fastapi import APIRouter

from wifi_tunja_smart_predictor import deployment
from wifi_tunja_smart_predictor.observability.metrics import metrics_snapshot

router = APIRouter(tags=["observability"])


def _readiness_status() -> str:
    try:
        if deployment.runtime_assets_available() and deployment.model_artifacts_available():
            return "ready"
    except Exception:
        return "unknown"
    return "not_ready"


@router.get("/observability/metrics")
def assistant_metrics() -> dict:
    """Return process-local counters and a readiness check without request data."""
    return {
        "service": "assistant_api",
        "availability": "available",
        "readiness": _readiness_status(),
        "metrics": metrics_snapshot("assistant_api"),
    }
