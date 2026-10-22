"""Scenario planning and experiment history API."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from math import isfinite

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response

from api.routes.prediction import _scenario_service
from api.schemas import (
    ExperimentComparisonRequest,
    ExperimentPage,
    ExperimentRecord,
    PlanningScenarioRequest,
)
from wifi_tunja_smart_predictor.config import experiment_database_path
from wifi_tunja_smart_predictor.exceptions import (
    LocationResolutionError,
    ModelArtifactError,
    ModelMetadataError,
    ModelNotFoundError,
    PredictionError,
    ScenarioBuildError,
)
from wifi_tunja_smart_predictor.experiments.service import ScenarioExperimentService
from wifi_tunja_smart_predictor.experiments.store import ExperimentStore
from wifi_tunja_smart_predictor.reporting.service import (
    build_experiment_report,
    render_experiment_markdown,
    serialize_experiment_json,
)

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _comparison_zone_map(result: dict) -> dict[str, dict]:
    zones = result.get("zones")
    if not isinstance(zones, list):
        raise HTTPException(status_code=409, detail="Experiment zone data is malformed.")
    mapped = {}
    for row in zones:
        if not isinstance(row, dict) or not isinstance(row.get("zone_id"), str):
            raise HTTPException(status_code=409, detail="Experiment zone data is malformed.")
        zone_id = row["zone_id"]
        if zone_id in mapped:
            raise HTTPException(status_code=409, detail="Experiment zone data is malformed.")
        for field in ("demand", "effective_capacity", "overload_connections"):
            number = row.get(field)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not isfinite(number)
            ):
                raise HTTPException(status_code=409, detail="Experiment zone data is malformed.")
        mapped[zone_id] = row
    return mapped


@lru_cache(maxsize=1)
def _store() -> ExperimentStore:
    return ExperimentStore(experiment_database_path())


def _read_store(operation):
    try:
        return operation()
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        raise HTTPException(
            status_code=503, detail="Experiment storage is unavailable or contains invalid data."
        ) from exc


@router.post("/scenarios", status_code=201, response_model=ExperimentRecord)
def create_scenario(payload: PlanningScenarioRequest) -> dict:
    try:
        service = ScenarioExperimentService(_scenario_service(), _store())
        return _read_store(
            lambda: service.run(
                name=payload.name,
                description=payload.description,
                when=payload.datetime,
                zones=[zone.model_dump() for zone in payload.zones],
                resource_budget_connections=payload.resource_budget_connections,
            )
        )
    except LocationResolutionError as exc:
        raise HTTPException(status_code=422, detail="Unknown synthetic zone_id.") from exc
    except ScenarioBuildError as exc:
        raise HTTPException(
            status_code=422, detail="Scenario could not be built from synthetic data."
        ) from exc
    except ModelNotFoundError as exc:
        raise HTTPException(
            status_code=503, detail="Required model artifacts are unavailable."
        ) from exc
    except (ModelArtifactError, ModelMetadataError, PredictionError) as exc:
        raise HTTPException(
            status_code=503, detail="Scenario model inference is unavailable."
        ) from exc


@router.get("", response_model=ExperimentPage)
def list_experiments(limit: int = Query(50, ge=1, le=100), offset: int = Query(0, ge=0)) -> dict:
    return {
        "items": _read_store(lambda: _store().list(limit, offset)),
        "limit": limit,
        "offset": offset,
    }


@router.post("/compare")
def compare_experiments(payload: ExperimentComparisonRequest) -> dict:
    records = [
        _read_store(lambda experiment_id=experiment_id: _store().get(experiment_id))
        for experiment_id in payload.experiment_ids
    ]
    if any(record is None for record in records):
        raise HTTPException(status_code=404, detail="One or more experiments were not found.")
    valid_records = [record for record in records if record is not None]
    signatures = []
    for record in valid_records:
        result = record["result"]
        provenance = result.get("provenance")
        if not isinstance(provenance, dict):
            raise HTTPException(status_code=409, detail="Experiment provenance is malformed.")
        dataset_sha256 = provenance.get("dataset_sha256")
        dataset_version = provenance.get("dataset_version")
        dataset_fingerprint_algorithm = provenance.get("dataset_fingerprint_algorithm")
        if (
            not isinstance(dataset_sha256, str)
            or len(dataset_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in dataset_sha256)
            or not isinstance(dataset_version, str)
            or not isinstance(dataset_fingerprint_algorithm, str)
        ):
            raise HTTPException(
                status_code=409,
                detail="Experiment dataset fingerprint is missing; comparison is unsafe.",
            )
        model_signatures = set()
        model_traces = provenance.get("models")
        if not isinstance(model_traces, list) or not model_traces:
            raise HTTPException(
                status_code=409,
                detail="Experiment model fingerprints are missing; comparison is unsafe.",
            )
        for trace in model_traces:
            if not isinstance(trace, dict):
                raise HTTPException(status_code=409, detail="Experiment model data is malformed.")
            classifier = trace.get("classifier")
            regressor = trace.get("regressor")
            if not isinstance(classifier, dict) or not isinstance(regressor, dict):
                raise HTTPException(status_code=409, detail="Experiment model data is malformed.")
            classifier_hash = classifier.get("artifact_sha256")
            regressor_hash = regressor.get("artifact_sha256")
            if (
                not isinstance(classifier_hash, str)
                or len(classifier_hash) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in classifier_hash)
                or not isinstance(regressor_hash, str)
                or len(regressor_hash) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in regressor_hash)
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Experiment model fingerprints are missing; comparison is unsafe.",
                )
            model_signatures.add(
                (
                    classifier_hash,
                    classifier.get("model_version"),
                    regressor_hash,
                    regressor.get("model_version"),
                )
            )
        signatures.append(
            (
                result.get("time_window"),
                dataset_version,
                dataset_sha256,
                dataset_fingerprint_algorithm,
                tuple(sorted(model_signatures)),
            )
        )
        if not isinstance(result.get("time_window"), str) or not model_signatures:
            raise HTTPException(status_code=409, detail="Experiment comparison data is incomplete.")
    if len(set(signatures)) != 1:
        raise HTTPException(
            status_code=409,
            detail="Experiments use incompatible time windows, datasets, or model artifacts.",
        )
    baseline = valid_records[0]
    baseline_zones = _comparison_zone_map(baseline["result"])
    comparisons = []
    for record in valid_records[1:]:
        zones = _comparison_zone_map(record["result"])
        shared = sorted(set(baseline_zones) & set(zones))
        comparisons.append(
            {
                "experiment_id": record["id"],
                "name": record["name"],
                "baseline_experiment_id": baseline["id"],
                "shared_zones": shared,
                "changes": {
                    zone_id: {
                        "demand_connections": zones[zone_id]["demand"]
                        - baseline_zones[zone_id]["demand"],
                        "capacity_connections": zones[zone_id]["effective_capacity"]
                        - baseline_zones[zone_id]["effective_capacity"],
                        "overload_connections": zones[zone_id]["overload_connections"]
                        - baseline_zones[zone_id]["overload_connections"],
                    }
                    for zone_id in shared
                },
                "interpretation": "Synthetic scenario differences only; not a real-network forecast.",
            }
        )
    return {
        "schema_version": "1.0",
        "baseline_experiment_id": baseline["id"],
        "comparisons": comparisons,
    }


@router.get("/{experiment_id}", response_model=ExperimentRecord)
def get_experiment(experiment_id: str) -> dict:
    record = _read_store(lambda: _store().get(experiment_id))
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    return record


@router.get("/{experiment_id}/report")
def export_experiment(
    experiment_id: str, format: str = Query("json", pattern="^(json|markdown)$")
) -> Response:
    record = _read_store(lambda: _store().get(experiment_id))
    if record is None:
        raise HTTPException(status_code=404, detail="Experiment not found.")
    try:
        report = build_experiment_report(record)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail="Experiment report data is invalid.") from exc
    if format == "json":
        return Response(
            content=serialize_experiment_json(report),
            media_type="application/json",
        )
    return Response(
        content=render_experiment_markdown(report), media_type="text/markdown; charset=utf-8"
    )
