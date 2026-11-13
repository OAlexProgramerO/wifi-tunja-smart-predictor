"""Read-only model artifact inventory endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from wifi_tunja_smart_predictor.models.inventory import build_artifact_inventory

router = APIRouter(tags=["model artifacts"])


@router.get("/model-artifacts/inventory")
def model_artifact_inventory() -> dict[str, Any]:
    """Return artifact availability, safe metadata summaries, and compatibility checks."""
    return build_artifact_inventory()
