"""Offline manifests and integrity verification for policy audit exports."""

from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from wifi_tunja_smart_predictor.experiments.ml_policy_export import (
    AUDIT_CSV_FIELDS,
    AUDIT_EXPORT_SCHEMA_VERSION,
)

AUDIT_MANIFEST_SCHEMA_VERSION = "1.0"
AUDIT_RECORD_CANONICALIZATION = "json-sort-keys-compact-utf8-v1"
_MANIFEST_FIELDS = {
    "manifest_schema_version",
    "export_format",
    "canonicalization",
    "generated_at_utc",
    "record_count",
    "applied_filters",
    "records_sha256",
}
_RECORD_FIELDS = {
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
    "provenance",
}
_PROVENANCE_FIELDS = {
    "project_version",
    "task",
    "dataset_kind",
    "dataset_fingerprint_algorithm",
    "dataset_fingerprint",
    "evaluation_split",
    "evaluation_window",
}
_FILTER_FIELDS = {
    "policy_id",
    "policy_version",
    "experiment_id",
    "outcome",
    "execution_source",
    "created_from_utc",
    "created_to_utc",
}
_EXPORT_FIELDS = {
    "export_schema_version",
    "generated_at_utc",
    "applied_filters",
    "record_count",
    "records",
}
_EVALUATION_WINDOW_FIELDS = {"start", "end", "rows", "cutoff"}
_VERIFICATION_STATUSES = {"VERIFIED", "INVALID", "UNVERIFIABLE"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class UnsupportedAuditEvidence(ValueError):
    """Raised when evidence uses a known-unknown schema, format, or canonicalizer."""


class MalformedAuditEvidence(ValueError):
    """Raised when evidence does not match the supported export structure."""


def canonical_audit_records_json(records: list[dict[str, Any]]) -> str:
    """Return the versioned canonical representation shared by JSON and CSV files."""
    validate_audit_records(records)
    try:
        return json.dumps(
            records,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MalformedAuditEvidence("Audit records are not canonical JSON values.") from exc


def audit_records_sha256(records: list[dict[str, Any]]) -> str:
    """Hash the canonical UTF-8 representation of the ordered records array."""
    canonical = canonical_audit_records_json(records).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def validate_audit_records(records: Any) -> None:
    """Validate the privacy-allowlisted record shape shared by both export formats."""
    if not isinstance(records, list):
        raise MalformedAuditEvidence("The records value must be a JSON array.")
    for index, record in enumerate(records):
        if not isinstance(record, dict) or set(record) != _RECORD_FIELDS:
            raise MalformedAuditEvidence(f"Audit record {index} has an invalid field set.")
        for key in _RECORD_FIELDS - {"policy_version", "verification_reasons", "provenance"}:
            if record[key] is not None and not isinstance(record[key], str):
                raise MalformedAuditEvidence(f"Audit record {index} has an invalid {key} value.")
        version = record["policy_version"]
        if version is not None and (not isinstance(version, int) or isinstance(version, bool)):
            raise MalformedAuditEvidence(f"Audit record {index} has an invalid policy_version.")
        reasons = record["verification_reasons"]
        if not isinstance(reasons, list) or any(not isinstance(item, str) for item in reasons):
            raise MalformedAuditEvidence(f"Audit record {index} has invalid verification reasons.")
        if record["verification_status"] not in _VERIFICATION_STATUSES:
            raise MalformedAuditEvidence(
                f"Audit record {index} has an unknown verification status."
            )
        provenance = record["provenance"]
        if not isinstance(provenance, dict) or set(provenance) != _PROVENANCE_FIELDS:
            raise MalformedAuditEvidence(f"Audit record {index} has invalid provenance fields.")
        if any(
            value is not None and not isinstance(value, (str, dict))
            for value in provenance.values()
        ):
            raise MalformedAuditEvidence(f"Audit record {index} has invalid provenance values.")
        window = provenance["evaluation_window"]
        if isinstance(window, dict) and (
            not set(window).issubset(_EVALUATION_WINDOW_FIELDS)
            or any(
                value is not None and not isinstance(value, (str, int)) for value in window.values()
            )
        ):
            raise MalformedAuditEvidence(f"Audit record {index} has invalid evaluation window.")


def _parse_timestamp(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise MalformedAuditEvidence(f"{field} must be an ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise MalformedAuditEvidence(f"{field} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise MalformedAuditEvidence(f"{field} must include a timezone.")
    return value


def _validate_export_metadata(
    *,
    schema_version: Any,
    generated_at_utc: Any,
    applied_filters: Any,
    record_count: Any,
    records: Any,
) -> dict[str, Any]:
    if schema_version is None:
        raise MalformedAuditEvidence("The export schema version is missing.")
    if not isinstance(schema_version, str):
        raise MalformedAuditEvidence("The export schema version must be a string.")
    if schema_version != AUDIT_EXPORT_SCHEMA_VERSION:
        raise UnsupportedAuditEvidence(f"Unsupported export schema version: {schema_version!r}.")
    generated = _parse_timestamp(generated_at_utc, "generated_at_utc")
    if not isinstance(applied_filters, dict):
        raise MalformedAuditEvidence("applied_filters must be a JSON object.")
    if not set(applied_filters).issubset(_FILTER_FIELDS):
        raise MalformedAuditEvidence("applied_filters contains unsupported fields.")
    validate_audit_records(records)
    if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 0:
        raise MalformedAuditEvidence("record_count must be a non-negative integer.")
    if record_count != len(records):
        raise MalformedAuditEvidence("record_count does not match the exported records.")
    return {
        "export_schema_version": schema_version,
        "generated_at_utc": generated,
        "applied_filters": applied_filters,
        "record_count": record_count,
        "records": records,
    }


def _normalize_csv_record(row: dict[str, str]) -> dict[str, Any]:
    try:
        policy_version = int(row["policy_version"]) if row["policy_version"] else None
        reasons = row["verification_reasons"].split(";") if row["verification_reasons"] else []
        provenance = json.loads(row["provenance_json"])
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        raise MalformedAuditEvidence("A CSV row contains malformed typed evidence.") from exc
    if not isinstance(provenance, dict):
        raise MalformedAuditEvidence("A CSV row provenance_json value must be an object.")
    record = {
        "audit_id": row["audit_id"] or None,
        "run_id": row["run_id"] or None,
        "event_run_id": row["event_run_id"] or None,
        "experiment_id": row["experiment_id"] or None,
        "policy_id": row["policy_id"] or None,
        "policy_version": policy_version,
        "timestamp_utc": row["timestamp_utc"] or None,
        "execution_source": row["execution_source"] or None,
        "outcome": row["outcome"] or None,
        "digest_sha256": row["digest_sha256"] or None,
        "verification_status": row["verification_status"],
        "verification_reasons": reasons,
        "provenance": provenance,
    }
    validate_audit_records([record])
    return record


def parse_policy_audit_export(data: bytes | str, export_format: str) -> dict[str, Any]:
    """Parse a JSON or CSV export into the shared validated export structure."""
    if export_format not in {"json", "csv"}:
        raise UnsupportedAuditEvidence(f"Unsupported export format: {export_format!r}.")
    try:
        text = data.decode("utf-8") if isinstance(data, bytes) else data
    except UnicodeDecodeError as exc:
        raise MalformedAuditEvidence("The export file is not valid UTF-8.") from exc
    if not isinstance(text, str):
        raise MalformedAuditEvidence("The export file must contain text evidence.")
    if export_format == "json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise MalformedAuditEvidence("The JSON export is malformed.") from exc
        if not isinstance(payload, dict):
            raise MalformedAuditEvidence("The JSON export must be an object.")
        if set(payload) != _EXPORT_FIELDS:
            raise MalformedAuditEvidence("The JSON export has missing or unknown fields.")
        return _validate_export_metadata(
            schema_version=payload.get("export_schema_version"),
            generated_at_utc=payload.get("generated_at_utc"),
            applied_filters=payload.get("applied_filters"),
            record_count=payload.get("record_count"),
            records=payload.get("records"),
        )

    reader = csv.DictReader(io.StringIO(text, newline=""))
    if tuple(reader.fieldnames or ()) != AUDIT_CSV_FIELDS:
        raise MalformedAuditEvidence("The CSV export header is not supported.")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise MalformedAuditEvidence("The CSV export contains a row with an invalid field count.")
    if not rows or rows[0].get("row_type") != "metadata":
        raise MalformedAuditEvidence("The CSV export is missing its metadata row.")
    metadata = rows[0]
    if any(row.get("row_type") != "audit_record" for row in rows[1:]):
        raise MalformedAuditEvidence("The CSV export contains an unexpected row type.")
    try:
        filters = json.loads(metadata["applied_filters_json"])
        count = int(metadata["record_count"])
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        raise MalformedAuditEvidence("The CSV metadata row is malformed.") from exc
    records = []
    for row in rows[1:]:
        if row.get("schema_version") != metadata.get("schema_version"):
            raise MalformedAuditEvidence("CSV row schema versions do not match.")
        records.append(_normalize_csv_record(row))
    return _validate_export_metadata(
        schema_version=metadata.get("schema_version"),
        generated_at_utc=metadata.get("generated_at_utc"),
        applied_filters=filters,
        record_count=count,
        records=records,
    )


def create_policy_audit_manifest(
    export_format: str,
    export: dict[str, Any],
    *,
    generated_at_utc: str | None = None,
) -> dict[str, Any]:
    """Create manifest metadata and digest from a validated JSON/CSV export object."""
    normalized = _validate_export_metadata(
        schema_version=export.get("export_schema_version"),
        generated_at_utc=export.get("generated_at_utc"),
        applied_filters=export.get("applied_filters"),
        record_count=export.get("record_count"),
        records=export.get("records"),
    )
    if export_format not in {"json", "csv"}:
        raise UnsupportedAuditEvidence(f"Unsupported export format: {export_format!r}.")
    return {
        "manifest_schema_version": AUDIT_MANIFEST_SCHEMA_VERSION,
        "export_format": export_format,
        "canonicalization": AUDIT_RECORD_CANONICALIZATION,
        "generated_at_utc": generated_at_utc or datetime.now(UTC).isoformat(),
        "record_count": normalized["record_count"],
        "applied_filters": normalized["applied_filters"],
        "records_sha256": audit_records_sha256(normalized["records"]),
    }


def create_policy_audit_manifest_from_bytes(
    export_data: bytes, export_format: str, *, generated_at_utc: str | None = None
) -> dict[str, Any]:
    """Build a manifest for the exact downloaded bytes, independent of CSV layout."""
    parsed = parse_policy_audit_export(export_data, export_format)
    return create_policy_audit_manifest(export_format, parsed, generated_at_utc=generated_at_utc)


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MalformedAuditEvidence("The manifest file cannot be read as valid JSON.") from exc
    if not isinstance(manifest, dict):
        raise MalformedAuditEvidence("The manifest must be a JSON object.")
    if set(manifest) != _MANIFEST_FIELDS:
        raise MalformedAuditEvidence("The manifest has missing or unknown fields.")
    if not isinstance(manifest.get("manifest_schema_version"), str):
        raise MalformedAuditEvidence("The manifest schema version must be a string.")
    if manifest["manifest_schema_version"] != AUDIT_MANIFEST_SCHEMA_VERSION:
        raise UnsupportedAuditEvidence(
            f"Unsupported manifest schema version: {manifest.get('manifest_schema_version')!r}."
        )
    if not isinstance(manifest.get("export_format"), str):
        raise MalformedAuditEvidence("The manifest export_format must be a string.")
    if manifest["export_format"] not in {"json", "csv"}:
        raise UnsupportedAuditEvidence(
            f"Unsupported export format: {manifest.get('export_format')!r}."
        )
    if not isinstance(manifest.get("canonicalization"), str):
        raise MalformedAuditEvidence("The manifest canonicalization must be a string.")
    if manifest["canonicalization"] != AUDIT_RECORD_CANONICALIZATION:
        raise UnsupportedAuditEvidence(
            f"Unsupported record canonicalization: {manifest.get('canonicalization')!r}."
        )
    _parse_timestamp(manifest.get("generated_at_utc"), "manifest generated_at_utc")
    if (
        not isinstance(manifest.get("record_count"), int)
        or isinstance(manifest.get("record_count"), bool)
        or manifest["record_count"] < 0
    ):
        raise MalformedAuditEvidence("Manifest record_count must be a non-negative integer.")
    if not isinstance(manifest.get("applied_filters"), dict):
        raise MalformedAuditEvidence("Manifest applied_filters must be a JSON object.")
    if not isinstance(manifest.get("records_sha256"), str) or not _SHA256_RE.fullmatch(
        manifest["records_sha256"]
    ):
        raise MalformedAuditEvidence(
            "Manifest records_sha256 must be a lowercase SHA-256 hex digest."
        )
    return manifest


def verify_policy_audit_export_files(
    export_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Verify exported records offline and return VALID/INVALID/UNSUPPORTED evidence."""
    export_file = Path(export_path)
    try:
        manifest = _load_manifest(Path(manifest_path))
        export_format = manifest["export_format"]
        suffix = export_file.suffix.lower()
        if suffix in {".json", ".csv"} and suffix[1:] != export_format:
            raise MalformedAuditEvidence(
                "Manifest format does not match the export file extension."
            )
        export = parse_policy_audit_export(export_file.read_bytes(), export_format)
        if manifest["record_count"] != export["record_count"]:
            raise MalformedAuditEvidence("Manifest record_count does not match the export.")
        if manifest["applied_filters"] != export["applied_filters"]:
            raise MalformedAuditEvidence("Manifest filters do not match the export metadata.")
        computed = audit_records_sha256(export["records"])
        if not hmac.compare_digest(computed, manifest["records_sha256"]):
            return {
                "status": "INVALID",
                "message": "The exported records do not match the manifest digest.",
                "record_count": export["record_count"],
                "computed_sha256": computed,
                "manifest_sha256": manifest["records_sha256"],
            }
        return {
            "status": "VALID",
            "message": "Export structure and canonical records digest match the manifest.",
            "record_count": export["record_count"],
            "computed_sha256": computed,
            "manifest_sha256": manifest["records_sha256"],
        }
    except UnsupportedAuditEvidence as exc:
        return {"status": "UNSUPPORTED", "message": str(exc)}
    except (MalformedAuditEvidence, OSError) as exc:
        return {"status": "INVALID", "message": str(exc)}


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; uses only local files and the standard library."""
    import argparse
    import sys

    from wifi_tunja_smart_predictor.experiments.ml_policy_export_history import (
        VerificationHistoryError,
        build_verification_history_record,
        history_cli_main,
        save_verification_history,
    )

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "history":
        return history_cli_main(arguments[1:])

    parser = argparse.ArgumentParser(description="Verify a downloaded policy audit export offline.")
    parser.add_argument("export_file", type=Path, help="Downloaded JSON or CSV audit export")
    parser.add_argument("manifest_file", type=Path, help="Matching audit export manifest JSON")
    parser.add_argument(
        "--save-history",
        type=Path,
        metavar="SQLITE_FILE",
        help="Opt in to saving metadata-only verification history locally",
    )
    args = parser.parse_args(arguments)
    result = verify_policy_audit_export_files(args.export_file, args.manifest_file)
    print(f"{result['status']}: {result['message']}")
    if args.save_history is not None:
        try:
            record = build_verification_history_record(args.export_file, args.manifest_file, result)
            save_verification_history(args.save_history, record)
            print(f"Saved verification ID: {record['verification_id']}")
        except (VerificationHistoryError, OSError) as exc:
            print(f"History error: {exc}", file=sys.stderr)
            return 3
    return {"VALID": 0, "INVALID": 1, "UNSUPPORTED": 2}[result["status"]]
