"""Deterministic integrity helpers for policy evaluation audit events."""

from __future__ import annotations

import hashlib
import json
from typing import Any

AUDIT_SCHEMA_VERSION = "1.0"


def canonical_audit_json(event: dict[str, Any]) -> str:
    """Serialize an audit event using sorted keys and compact UTF-8 JSON."""
    return json.dumps(
        event,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def audit_digest(event: dict[str, Any]) -> str:
    """Return the SHA-256 digest of the documented canonical event encoding."""
    return hashlib.sha256(canonical_audit_json(event).encode("utf-8")).hexdigest()


def build_policy_audit_event(
    report: dict[str, Any], *, execution_source: str = "api"
) -> dict[str, Any]:
    """Build a privacy-safe event from persisted evaluation provenance."""
    policy = report.get("policy") if isinstance(report.get("policy"), dict) else {}
    experiment = report.get("experiment") if isinstance(report.get("experiment"), dict) else {}
    dataset = experiment.get("dataset") if isinstance(experiment.get("dataset"), dict) else {}
    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_id": report["run_id"],
        "run_id": report["run_id"],
        "experiment_id": experiment.get("experiment_id"),
        "policy_id": policy.get("policy_id"),
        "policy_version": policy.get("version"),
        "timestamp_utc": report["created_at_utc"],
        "execution_source": execution_source,
        "outcome": report.get("status"),
        "provenance": {
            "project_version": report.get("project_version"),
            "task": experiment.get("task"),
            "dataset_kind": dataset.get("kind"),
            "dataset_fingerprint_algorithm": dataset.get("fingerprint_algorithm"),
            "dataset_fingerprint": dataset.get("fingerprint"),
            "evaluation_split": experiment.get("evaluation_split"),
            "evaluation_window": experiment.get("evaluation_window"),
        },
    }
