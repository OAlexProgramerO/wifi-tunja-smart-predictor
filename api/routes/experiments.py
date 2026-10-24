"""Scenario planning and experiment history API."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from math import isfinite

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response

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


class ExperimentComparisonError(ValueError):
    """Safe, structured comparison diagnostic for malformed saved result data."""

    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


def _comparison_error(
    status_code: int, reason_code: str, message: str, reason_codes: list[str] | None = None
) -> JSONResponse:
    content = {"detail": message, "reason_code": reason_code}
    if reason_codes:
        content["reason_codes"] = reason_codes
    return JSONResponse(status_code=status_code, content=content)


def _comparison_zone_map(result: dict) -> dict[str, dict]:
    zones = result.get("zones")
    if not isinstance(zones, list):
        raise ExperimentComparisonError("INVALID_ZONE_DATA", "Experiment zone data is malformed.")
    mapped = {}
    for row in zones:
        if not isinstance(row, dict) or not isinstance(row.get("zone_id"), str):
            raise ExperimentComparisonError(
                "INVALID_ZONE_DATA", "Experiment zone data is malformed."
            )
        zone_id = row["zone_id"]
        if zone_id in mapped:
            raise ExperimentComparisonError(
                "INVALID_ZONE_DATA", "Experiment zone data is malformed."
            )
        for field in ("demand", "effective_capacity", "overload_connections"):
            number = row.get(field)
            if (
                isinstance(number, bool)
                or not isinstance(number, (int, float))
                or not isfinite(number)
            ):
                raise ExperimentComparisonError(
                    "INVALID_ZONE_DATA", "Experiment zone data is malformed."
                )
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
    items, integrity_warnings = _read_store(lambda: _store().list_page(limit, offset))
    return {
        "items": items,
        "limit": limit,
        "offset": offset,
        "integrity_warnings": integrity_warnings,
    }


@router.post("/compare", response_model=None)
def compare_experiments(payload: ExperimentComparisonRequest) -> dict | JSONResponse:
    try:
        records = [
            _read_store(lambda experiment_id=experiment_id: _store().get(experiment_id))
            for experiment_id in payload.experiment_ids
        ]
    except HTTPException as exc:
        if exc.status_code == 503:
            return _comparison_error(
                503, "INVALID_EXPERIMENT_RECORD", "A saved experiment could not be verified."
            )
        raise
    if any(record is None for record in records):
        return _comparison_error(
            404, "EXPERIMENT_NOT_FOUND", "One or more experiments were not found."
        )
    valid_records = [record for record in records if record is not None]
    signatures = []
    incompatible_reasons = set()
    for record in valid_records:
        if record["integrity"]["status"] == "invalid":
            return _comparison_error(
                409, "INVALID_EXPERIMENT_RECORD", "Saved experiment data failed integrity checks."
            )
        if record["integrity"]["status"] == "incomplete":
            return _comparison_error(
                409,
                "INCOMPLETE_PROVENANCE",
                "Experiment provenance is incomplete; comparison is unsafe.",
                record["integrity"]["reason_codes"],
            )
        result = record["result"]
        provenance = result.get("provenance")
        if not isinstance(provenance, dict):
            return _comparison_error(
                409, "INVALID_PROVENANCE", "Experiment provenance is malformed."
            )
        dataset_sha256 = provenance.get("dataset_sha256")
        dataset_version = provenance.get("dataset_version")
        dataset_fingerprint_algorithm = provenance.get("dataset_fingerprint_algorithm")
        if (
            not isinstance(dataset_sha256, str)
            or len(dataset_sha256) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in dataset_sha256)
            or not isinstance(dataset_version, str)
            or not dataset_version
            or not isinstance(dataset_fingerprint_algorithm, str)
            or not dataset_fingerprint_algorithm
        ):
            return _comparison_error(
                409,
                "DATASET_PROVENANCE_INCOMPLETE",
                "Experiment dataset fingerprint is missing or invalid; comparison is unsafe.",
            )
        model_signatures = set()
        model_traces = provenance.get("models")
        if not isinstance(model_traces, list) or not model_traces:
            return _comparison_error(
                409,
                "MODEL_PROVENANCE_INCOMPLETE",
                "Experiment model fingerprints are missing; comparison is unsafe.",
            )
        for trace in model_traces:
            if not isinstance(trace, dict):
                return _comparison_error(
                    409, "INVALID_MODEL_PROVENANCE", "Experiment model data is malformed."
                )
            classifier = trace.get("classifier")
            regressor = trace.get("regressor")
            if not isinstance(classifier, dict) or not isinstance(regressor, dict):
                return _comparison_error(
                    409, "INVALID_MODEL_PROVENANCE", "Experiment model data is malformed."
                )
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
                return _comparison_error(
                    409,
                    "MODEL_PROVENANCE_INCOMPLETE",
                    "Experiment model fingerprints are missing or invalid; comparison is unsafe.",
                )
            if not classifier.get("model_version") or not regressor.get("model_version"):
                return _comparison_error(
                    409,
                    "MODEL_VERSION_UNAVAILABLE",
                    "Experiment model versions are unavailable; comparison is unsafe.",
                )
            model_signatures.add(
                (
                    classifier_hash,
                    classifier.get("model_version"),
                    regressor_hash,
                    regressor.get("model_version"),
                )
            )
        time_window = result.get("time_window")
        if not isinstance(time_window, str) or not time_window:
            incompatible_reasons.add("TIME_WINDOW_UNAVAILABLE")
            time_window = None
        application_versions = {
            trace.get("application_version")
            for trace in model_traces
            if isinstance(trace, dict) and trace.get("application_version")
        }
        if len(application_versions) != 1:
            incompatible_reasons.add("APPLICATION_VERSION_UNAVAILABLE")
        signatures.append(
            {
                "time_window": time_window,
                "dataset": (dataset_version, dataset_sha256, dataset_fingerprint_algorithm),
                "models": tuple(sorted(model_signatures)),
                "application_version": next(iter(application_versions), None),
            }
        )
    baseline_signature = signatures[0]
    reason_codes = incompatible_reasons
    for signature in signatures[1:]:
        if signature["time_window"] != baseline_signature["time_window"]:
            reason_codes.add("INCOMPATIBLE_TIME_WINDOW")
        if signature["dataset"] != baseline_signature["dataset"]:
            reason_codes.add("INCOMPATIBLE_DATASET_PROVENANCE")
        if signature["models"] != baseline_signature["models"]:
            reason_codes.add("INCOMPATIBLE_MODEL_PROVENANCE")
        if signature["application_version"] != baseline_signature["application_version"]:
            reason_codes.add("INCOMPATIBLE_APPLICATION_VERSION")
    if reason_codes:
        return _comparison_error(
            409,
            "INCOMPATIBLE_PROVENANCE",
            "Experiments use incompatible time windows, datasets, model artifacts, or application versions.",
            sorted(reason_codes),
        )
    baseline = valid_records[0]
    try:
        baseline_zones = _comparison_zone_map(baseline["result"])
    except ExperimentComparisonError as exc:
        return _comparison_error(409, exc.reason_code, str(exc))
    comparisons = []
    for record in valid_records[1:]:
        try:
            zones = _comparison_zone_map(record["result"])
        except ExperimentComparisonError as exc:
            return _comparison_error(409, exc.reason_code, str(exc))
        shared = sorted(set(baseline_zones) & set(zones))
        comparisons.append(
            {
                "experiment_id": record["id"],
                "name": record["name"],
                "baseline_experiment_id": baseline["id"],
                "shared_zones": shared,
                "comparison_status": "comparable" if shared else "no_shared_zones",
                "scientific_comparability": "not_assessable" if not shared else "limited",
                "comparability_reasons": (
                    ["The experiments share no zone IDs."] if not shared else []
                ),
                "comparability_reason_codes": ["NO_SHARED_ZONES"] if not shared else [],
                "interpretation_detail": (
                    "Synthetic scenario differences only; not a real-network forecast."
                    if shared
                    else "These experiments share no zone IDs, so there are no zone-level results to compare."
                ),
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
    try:
        markdown = render_experiment_markdown(report)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise HTTPException(status_code=503, detail="Experiment report data is invalid.") from exc
    return Response(content=markdown, media_type="text/markdown; charset=utf-8")
