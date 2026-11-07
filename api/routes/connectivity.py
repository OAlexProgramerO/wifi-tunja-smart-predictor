"""Read-only API for persisted official connectivity snapshots."""

from __future__ import annotations

import json
import sqlite3

from fastapi import APIRouter, HTTPException, Query

from wifi_tunja_smart_predictor.connectivity.analysis import analyze_tunja
from wifi_tunja_smart_predictor.connectivity.mintic import (
    SourceUnavailableError,
    get_refresh_history,
    latest_refresh_status,
    load_snapshot,
)
from wifi_tunja_smart_predictor.connectivity.sources import source_registry

router = APIRouter(prefix="/connectivity", tags=["official-connectivity"])


@router.get("/sources")
def list_connectivity_sources() -> dict:
    try:
        snapshot = load_snapshot()
    except (OSError, ValueError, sqlite3.DatabaseError):
        snapshot = None
    retrieval = (
        {
            snapshot["batch"]["source_id"]: {
                "last_successful_retrieval_utc": snapshot["batch"]["retrieved_at_utc"],
                "last_source_update_utc": snapshot["batch"]["source_updated_at_utc"],
                "status": "stale_local_cache" if snapshot["stale"] else "available_locally",
            }
        }
        if snapshot
        else {}
    )
    return {"sources": source_registry(retrieval)}


@router.get("/tunja")
def tunja_connectivity() -> dict:
    try:
        snapshot = load_snapshot()
    except (OSError, ValueError, sqlite3.DatabaseError, SourceUnavailableError) as exc:
        raise HTTPException(
            status_code=503, detail="Official connectivity snapshot is unavailable."
        ) from exc
    if snapshot is None:
        raise HTTPException(
            status_code=503, detail="No official connectivity snapshot is available."
        )
    batch = snapshot["batch"]
    return {
        "source": source_registry(
            {
                batch["source_id"]: {
                    "last_successful_retrieval_utc": batch["retrieved_at_utc"],
                    "last_source_update_utc": batch["source_updated_at_utc"],
                    "status": "stale_local_cache" if snapshot["stale"] else "available_locally",
                }
            }
        )[0],
        "retrieved_at_utc": batch["retrieved_at_utc"],
        "source_updated_at_utc": batch["source_updated_at_utc"],
        "stale": snapshot["stale"],
        "provenance": {
            "dataset_id": batch["dataset_id"],
            "original_url": batch["original_url"],
            "content_sha256": batch["content_sha256"],
            "retrieved_count": batch["retrieved_count"],
            "accepted_count": batch["accepted_count"],
            "rejected_count": batch["rejected_count"],
            "rejection_reasons": json.loads(batch["error_summary"]),
            "validation_status": batch["validation_status"],
            "transformation_version": batch["transformation_version"],
        },
        "reporting_periods": sorted({record["period"] for record in snapshot["records"]}),
        "analysis": analyze_tunja(snapshot["records"]),
        "refresh_error": "last_refresh_failed" if snapshot.get("refresh_error") else None,
        "saturation_ground_truth_available": False,
    }


@router.get("/refresh/latest")
def connectivity_refresh_latest() -> dict:
    """Report latest refresh attempt separately from the locally cached snapshot."""
    try:
        return latest_refresh_status()
    except (OSError, sqlite3.DatabaseError) as exc:
        raise HTTPException(
            status_code=503, detail="Connectivity refresh history is unavailable."
        ) from exc


@router.get("/refresh/history")
def connectivity_refresh_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Return paginated refresh-attempt history, newest first."""
    try:
        return get_refresh_history(limit=limit, offset=offset)
    except (OSError, sqlite3.DatabaseError) as exc:
        raise HTTPException(
            status_code=503, detail="Connectivity refresh history is unavailable."
        ) from exc
