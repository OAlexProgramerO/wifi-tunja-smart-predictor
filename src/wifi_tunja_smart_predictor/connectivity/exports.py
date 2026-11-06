"""Build deterministic, provenance-rich exports from validated local snapshots."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from wifi_tunja_smart_predictor.connectivity.analysis import analyze_tunja
from wifi_tunja_smart_predictor.connectivity.sources import source_registry

CSV_FIELDS = (
    "export_record_type",
    "metadata_json",
    "raw_source_record_json",
    "processed_year",
    "processed_quarter",
    "processed_reporting_period",
    "processed_provider",
    "processed_department_code",
    "processed_department",
    "processed_municipality_code",
    "processed_municipality",
    "processed_segment",
    "processed_technology",
    "processed_downstream_mbps",
    "processed_upstream_mbps",
    "processed_reported_accesses",
    "record_validation_status",
)
NORMALIZED_REQUIRED_FIELDS = {
    "year",
    "quarter",
    "period",
    "provider",
    "department_code",
    "department",
    "municipality_code",
    "municipality",
    "segment",
    "technology",
    "downstream_mbps",
    "upstream_mbps",
    "reported_accesses",
}


def _source_metadata(batch: dict[str, Any], *, stale: bool) -> dict[str, Any]:
    source_id = batch.get("source_id", "mintic_fixed_accesses")
    retrieval = {
        source_id: {
            "last_successful_retrieval_utc": batch.get("retrieved_at_utc"),
            "last_source_update_utc": batch.get("source_updated_at_utc"),
            "status": "stale_local_cache" if stale else "available_locally",
        }
    }
    return next(
        (source for source in source_registry(retrieval) if source["source_id"] == source_id),
        {
            "source_id": source_id,
            "institution": None,
            "dataset_name": None,
            "official_url": batch.get("original_url"),
            "dataset_id": batch.get("dataset_id"),
        },
    )


def build_export_bundle(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Return processed analysis and source provenance, retaining each original row."""
    records = snapshot.get("records") or []
    batch = snapshot.get("batch") or {}
    stale = bool(snapshot.get("stale", False))
    source = _source_metadata(batch, stale=stale)
    records_complete = all(
        isinstance(record, dict) and NORMALIZED_REQUIRED_FIELDS <= record.keys()
        for record in records
    )
    try:
        if not records_complete:
            raise ValueError("Normalized record fields are incomplete.")
        analysis = {**analyze_tunja(records), "analysis_status": "available"}
    except (KeyError, TypeError, ValueError):
        analysis = {
            "municipality": "Tunja",
            "municipality_code": "15001",
            "geographic_granularity": "municipality",
            "observation_count": len(records),
            "sum_of_reported_access_counts_across_quarterly_rows": None,
            "latest_reporting_period": None,
            "latest_period_reported_access_total": None,
            "provider_accesses_across_periods": [],
            "technology_accesses_across_periods": [],
            "latest_period_provider_accesses": [],
            "latest_period_technology_accesses": [],
            "quarterly_accesses": [],
            "latest_period_downstream_values": [],
            "latest_period_upstream_values": [],
            "source_id": batch.get("source_id", "mintic_fixed_accesses"),
            "methodology_version": "mintic-tunja-descriptive-v1",
            "analysis_status": "unavailable_incomplete_normalized_fields",
            "interpretation": "Aggregate analysis is unavailable because normalized record fields are incomplete.",
        }
    periods = sorted(
        {
            str(record.get("period"))
            for record in records
            if isinstance(record, dict) and record.get("period")
        }
    )
    processed_fields = CSV_FIELDS[3:-1]
    exported_records = []
    for record in records:
        safe_record = record if isinstance(record, dict) else {}
        exported_records.append(
            {
                "raw_source_record": safe_record.get("raw") or {},
                "processed": {
                    field.removeprefix("processed_"): safe_record.get(
                        field.removeprefix("processed_")
                    )
                    for field in processed_fields
                },
                "validation_status": (
                    "accepted"
                    if isinstance(record, dict) and NORMALIZED_REQUIRED_FIELDS <= record.keys()
                    else "incomplete_normalized_record"
                ),
            }
        )
    return {
        "export_schema": "official-connectivity-export-v1",
        "data_classification": "REAL_OFFICIAL_DATA; separate from synthetic model data",
        "snapshot_status": "stale_local_snapshot" if stale else "validated_local_snapshot",
        "source": {
            "source_id": source.get("source_id"),
            "institution": source.get("institution"),
            "dataset_name": source.get("dataset_name"),
            "official_source_url": source.get("official_url"),
            "api_url": source.get("api_url"),
            "dataset_id": source.get("dataset_id") or batch.get("dataset_id"),
            "status": source.get("status"),
        },
        "provenance": {
            "retrieval_timestamp_utc": batch.get("retrieved_at_utc"),
            "source_update_timestamp_utc": batch.get("source_updated_at_utc"),
            "content_sha256": batch.get("content_sha256"),
            "retrieved_record_count": batch.get("retrieved_count"),
            "accepted_record_count": batch.get("accepted_count", len(records)),
            "rejected_record_count": batch.get("rejected_count"),
            "rejection_reasons": _parse_rejection_reasons(batch.get("error_summary")),
            "validation_status": batch.get("validation_status", "unknown"),
            "export_validation_status": (
                "complete" if records_complete else "incomplete_snapshot_records"
            ),
            "transformation_version": batch.get("transformation_version"),
        },
        "coverage": {
            "municipality": analysis.get("municipality", "Tunja"),
            "municipality_code": analysis.get("municipality_code", "15001"),
            "geographic_granularity": analysis.get("geographic_granularity", "municipality"),
            "reporting_periods": periods,
            "latest_reporting_period": periods[-1] if periods else None,
        },
        "record_count": len(records),
        "raw_records": {
            "description": "Original source JSON records corresponding to accepted normalized records.",
            "items": [item["raw_source_record"] for item in exported_records],
        },
        "processed_records": {
            "description": (
                "Validated normalized fields. Speed values are represented in Mbps; source values "
                "are retained separately in raw_records. Quarter and municipality codes are normalized."
            ),
            "items": [
                {
                    **item["processed"],
                    "record_validation_status": item["validation_status"],
                }
                for item in exported_records
            ],
        },
        "analysis": analysis,
        "limitations": (
            "Quarterly fixed-internet access aggregates at municipality granularity; not WiFi telemetry, "
            "unique people, simultaneous users, measured throughput, neighborhood data, congestion, "
            "or saturation. Existing model data and results are synthetic."
        ),
    }


def _parse_rejection_reasons(value: Any) -> dict[str, Any]:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def build_connectivity_csv(snapshot: dict[str, Any]) -> str:
    """Serialize original source JSON and separate normalized fields as CSV."""
    records = snapshot.get("records") or []
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    bundle = build_export_bundle(snapshot)
    writer.writerow(
        {
            "export_record_type": "metadata",
            "metadata_json": json.dumps(
                {
                    "data_classification": bundle["data_classification"],
                    "snapshot_status": bundle["snapshot_status"],
                    "source": bundle["source"],
                    "provenance": bundle["provenance"],
                    "coverage": bundle["coverage"],
                    "record_count": bundle["record_count"],
                    "validation_status": bundle["provenance"]["export_validation_status"],
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
    )
    for record in records:
        safe_record = record if isinstance(record, dict) else {}
        row = {
            "export_record_type": "record",
            "raw_source_record_json": json.dumps(
                safe_record.get("raw") or {}, ensure_ascii=False, sort_keys=True
            ),
            "record_validation_status": (
                "accepted"
                if NORMALIZED_REQUIRED_FIELDS <= safe_record.keys()
                else "incomplete_normalized_record"
            ),
        }
        row.update(
            {field: safe_record.get(field.removeprefix("processed_")) for field in CSV_FIELDS[3:-1]}
        )
        writer.writerow(row)
    return output.getvalue()


def build_connectivity_json(snapshot: dict[str, Any]) -> str:
    """Serialize analysis and provenance with stable keys and readable Unicode."""
    return json.dumps(build_export_bundle(snapshot), ensure_ascii=False, indent=2, sort_keys=True)
