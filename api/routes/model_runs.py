"""Read-only comparisons of persisted synthetic model evaluation candidates."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from wifi_tunja_smart_predictor.models.run_comparison import build_model_run_comparison

router = APIRouter(tags=["model runs"])


@router.get("/model-runs/comparison")
def model_run_comparison() -> dict[str, Any]:
    """Compare saved classifier/regressor candidate metrics within their task and split."""
    return build_model_run_comparison()
