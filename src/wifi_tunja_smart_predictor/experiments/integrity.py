"""Non-destructive integrity assessment for saved synthetic experiments."""

from __future__ import annotations

import re
from datetime import datetime
from math import isfinite
from typing import Any

_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value)


def _contains_non_finite(value: Any) -> bool:
    if isinstance(value, float):
        return not isfinite(value)
    if isinstance(value, dict):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_non_finite(item) for item in value)
    return False


def inspect_experiment_integrity(record: dict[str, Any]) -> dict[str, Any]:
    """Classify stored experiment structure and provenance without modifying it.

    Missing provenance remains ``incomplete`` for older records. Malformed or
    contradictory values are ``invalid`` and must not be used for comparison.
    """
    incomplete: set[str] = set()
    invalid: set[str] = set()

    for field in ("created_at_utc", "updated_at_utc"):
        value = record.get(field)
        if not isinstance(value, str):
            invalid.add("INVALID_UTC_TIMESTAMP")
            continue
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            invalid.add("INVALID_UTC_TIMESTAMP")
            continue
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            invalid.add("UTC_TIMESTAMP_WITHOUT_OFFSET")

    configuration = record.get("configuration")
    result = record.get("result")
    if not isinstance(configuration, dict) or not isinstance(result, dict):
        invalid.add("INVALID_RECORD_SHAPE")
        configuration = configuration if isinstance(configuration, dict) else {}
        result = result if isinstance(result, dict) else {}
    if _contains_non_finite(configuration) or _contains_non_finite(result):
        invalid.add("NON_FINITE_VALUE")

    scenario_kind = configuration.get("scenario_kind") if isinstance(configuration, dict) else None
    is_derived_optimization = scenario_kind in {
        "allocation_optimization",
        "robustness_analysis",
    }
    is_scenario_record = (
        record.get("experiment_type") == "capacity_scenario" and not is_derived_optimization
    )
    if is_scenario_record and not configuration:
        incomplete.add("SCENARIO_PARAMETERS_MISSING")
        incomplete.add("SCENARIO_ZONES_MISSING")
    elif is_scenario_record and "datetime" not in configuration:
        incomplete.add("SCENARIO_PARAMETERS_MISSING")
    if is_scenario_record and configuration and "zones" not in configuration:
        incomplete.add("SCENARIO_ZONES_MISSING")
    elif is_scenario_record and configuration and not isinstance(configuration["zones"], list):
        invalid.add("INVALID_SCENARIO_ZONES")
    elif is_scenario_record and configuration:
        zone_ids: set[str] = set()
        for zone in configuration["zones"]:
            if not isinstance(zone, dict) or not isinstance(zone.get("zone_id"), str):
                invalid.add("INVALID_SCENARIO_ZONES")
                continue
            zone_id = zone["zone_id"]
            if not zone_id or zone_id in zone_ids:
                invalid.add("INVALID_SCENARIO_ZONES")
            zone_ids.add(zone_id)
            capacity = zone.get("capacity_connections")
            if capacity is not None and (not _is_number(capacity) or capacity < 0):
                invalid.add("INVALID_SCENARIO_ZONES")

    zones = result.get("zones")
    if zones is None:
        incomplete.add("ZONE_RESULTS_MISSING")
    elif not isinstance(zones, list):
        invalid.add("INVALID_ZONE_RESULTS")
    else:
        zone_ids = set()
        for zone in zones:
            if not isinstance(zone, dict) or not isinstance(zone.get("zone_id"), str):
                invalid.add("INVALID_ZONE_RESULTS")
                continue
            zone_id = zone["zone_id"]
            if not zone_id or zone_id in zone_ids:
                invalid.add("INVALID_ZONE_RESULTS")
            zone_ids.add(zone_id)
            for field in ("demand", "effective_capacity", "overload_connections"):
                value = zone.get(field)
                if not _is_number(value) or value < 0:
                    invalid.add("INVALID_ZONE_RESULTS")
                    break

    provenance = result.get("provenance")
    if is_derived_optimization and provenance is None:
        provenance = configuration.get("parent_provenance")
    if provenance is None:
        incomplete.update({"DATASET_PROVENANCE_MISSING", "MODEL_PROVENANCE_MISSING"})
        provenance = {}
    elif not isinstance(provenance, dict):
        invalid.add("INVALID_PROVENANCE")
        provenance = {}

    if not provenance:
        incomplete.update({"DATASET_PROVENANCE_MISSING", "MODEL_PROVENANCE_MISSING"})
    dataset_hash = provenance.get("dataset_sha256")
    if dataset_hash is None:
        incomplete.add("DATASET_FINGERPRINT_MISSING")
    elif not isinstance(dataset_hash, str) or not _SHA256.fullmatch(dataset_hash):
        invalid.add("DATASET_FINGERPRINT_INVALID")
    for field, code in (
        ("dataset_version", "DATASET_VERSION_MISSING"),
        ("dataset_fingerprint_algorithm", "DATASET_FINGERPRINT_ALGORITHM_MISSING"),
    ):
        value = provenance.get(field)
        if value is None or value == "":
            incomplete.add(code)
        elif not isinstance(value, str):
            invalid.add("INVALID_PROVENANCE")

    model_traces = provenance.get("models")
    if model_traces is None or model_traces == []:
        incomplete.add("MODEL_PROVENANCE_MISSING")
    elif not isinstance(model_traces, list):
        invalid.add("INVALID_MODEL_PROVENANCE")
    else:
        application_versions = set()
        for trace in model_traces:
            if not isinstance(trace, dict):
                invalid.add("INVALID_MODEL_PROVENANCE")
                continue
            for role in ("classifier", "regressor"):
                artifact = trace.get(role)
                if artifact is None:
                    incomplete.add(f"{role.upper()}_PROVENANCE_MISSING")
                    continue
                if not isinstance(artifact, dict):
                    invalid.add("INVALID_MODEL_PROVENANCE")
                    continue
                fingerprint = artifact.get("artifact_sha256")
                if fingerprint is None:
                    incomplete.add(f"{role.upper()}_FINGERPRINT_MISSING")
                elif not isinstance(fingerprint, str) or not _SHA256.fullmatch(fingerprint):
                    invalid.add("MODEL_FINGERPRINT_INVALID")
                if not artifact.get("model_version"):
                    incomplete.add(f"{role.upper()}_VERSION_MISSING")
            if not trace.get("application_version"):
                incomplete.add("APPLICATION_VERSION_MISSING")
            else:
                application_versions.add(trace["application_version"])
        if len(application_versions) > 1:
            invalid.add("INCONSISTENT_APPLICATION_VERSION")

    status = "invalid" if invalid else "incomplete" if incomplete else "verified"
    reasons = sorted(invalid | incomplete)
    return {
        "status": status,
        "assessment_status": "assessable" if status == "verified" else "not_assessable",
        "reason_codes": reasons,
    }
