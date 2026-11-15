"""Read-only synthetic feature drift report endpoint."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query

from wifi_tunja_smart_predictor.models.drift import build_current_drift_report

router = APIRouter(tags=["synthetic drift"])


@router.get("/drift/monitor")
def drift_monitor(
    evaluation_split: Literal["validation", "test"] = Query("test"),
) -> dict[str, Any]:
    """Calculate synthetic input-feature drift without writing reports or model artifacts."""
    try:
        return build_current_drift_report(evaluation_split)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Synthetic drift data is unavailable.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=503, detail="Synthetic drift data could not be read."
        ) from exc
