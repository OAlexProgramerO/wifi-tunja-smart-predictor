"""Privacy-safe JSON and CSV rendering for saved policy audit events."""

from __future__ import annotations

import csv
import io
import json
from datetime import UTC, datetime
from typing import Any

AUDIT_EXPORT_SCHEMA_VERSION = "1.0"
AUDIT_CSV_FIELDS = (
    "row_type",
    "schema_version",
    "generated_at_utc",
    "applied_filters_json",
    "record_count",
    "audit_id",
    "run_id",
    "event_run_id",
    "experiment_id",
    "policy_id",
    "policy_version",
    "timestamp_utc",
    "execution_source",
    "outcome",
    "digest_sha256",
    "verification_status",
    "verification_reasons",
    "provenance_json",
)
_PROVENANCE_FIELDS = (
    "project_version",
    "task",
    "dataset_kind",
    "dataset_fingerprint_algorithm",
    "dataset_fingerprint",
    "evaluation_split",
    "evaluation_window",
)


def build_policy_audit_export(
    audits: list[dict[str, Any]],
    applied_filters: dict[str, Any],
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create a stable, allowlisted export document from verified history rows."""
    generated = generated_at_utc or datetime.now(UTC).isoformat()
    records = []
    ordered_audits = sorted(audits, key=lambda item: item.get("audit_id") or "")
    ordered_audits.sort(key=lambda item: item.get("timestamp_utc") or "", reverse=True)
    for audit in ordered_audits:
        event = audit.get("event") if isinstance(audit.get("event"), dict) else {}
        provenance = event.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        verification = audit.get("verification")
        verification = verification if isinstance(verification, dict) else {}
        records.append(
            {
                "audit_id": audit.get("audit_id"),
                "run_id": audit.get("run_id"),
                "event_run_id": event.get("run_id"),
                "experiment_id": event.get("experiment_id", audit.get("experiment_id")),
                "policy_id": event.get("policy_id", audit.get("policy_id")),
                "policy_version": event.get("policy_version", audit.get("policy_version")),
                "timestamp_utc": audit.get("timestamp_utc"),
                "execution_source": audit.get("execution_source"),
                "outcome": audit.get("outcome"),
                "digest_sha256": audit.get("digest_sha256"),
                "verification_status": verification.get("status", "UNVERIFIABLE"),
                "verification_reasons": sorted(verification.get("reasons", [])),
                "provenance": {key: provenance.get(key) for key in _PROVENANCE_FIELDS},
            }
        )
    return {
        "export_schema_version": AUDIT_EXPORT_SCHEMA_VERSION,
        "generated_at_utc": generated,
        "applied_filters": applied_filters,
        "record_count": len(records),
        "records": records,
    }


def render_policy_audit_csv(export: dict[str, Any]) -> str:
    """Render export metadata as a first CSV row and audit events as following rows."""
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=AUDIT_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerow(
        {
            "row_type": "metadata",
            "schema_version": export["export_schema_version"],
            "generated_at_utc": export["generated_at_utc"],
            "applied_filters_json": json.dumps(
                export["applied_filters"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ),
            "record_count": export["record_count"],
        }
    )
    for record in export["records"]:
        writer.writerow(
            {
                "row_type": "audit_record",
                "schema_version": export["export_schema_version"],
                "audit_id": record["audit_id"],
                "run_id": record["run_id"],
                "event_run_id": record["event_run_id"],
                "experiment_id": record["experiment_id"],
                "policy_id": record["policy_id"],
                "policy_version": record["policy_version"],
                "timestamp_utc": record["timestamp_utc"],
                "execution_source": record["execution_source"],
                "outcome": record["outcome"],
                "digest_sha256": record["digest_sha256"],
                "verification_status": record["verification_status"],
                "verification_reasons": ";".join(record["verification_reasons"]),
                "provenance_json": json.dumps(
                    record["provenance"],
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            }
        )
    return buffer.getvalue()
