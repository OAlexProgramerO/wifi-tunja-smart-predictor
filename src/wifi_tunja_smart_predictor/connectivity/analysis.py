"""Descriptive analysis of compatible quarterly MinTIC Tunja observations."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any

from wifi_tunja_smart_predictor.connectivity.mintic import REQUIRED_FIELDS

QUALITY_INDICATOR_FIELDS = {
    "reported_accesses": {"accesos_fijos_internet"},
    "provider_accesses": {"proveedor", "accesos_fijos_internet"},
    "technology_accesses": {"tecnologia", "accesos_fijos_internet"},
    "downstream_band_accesses": {"velocidad_bajada", "accesos_fijos_internet"},
    "upstream_band_accesses": {"velocidad_subida", "accesos_fijos_internet"},
}


def build_tunja_quality_report(
    records: list[dict[str, Any]],
    raw_records: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize field completeness and exact duplicates without changing source rows."""
    provenance = provenance or {}
    source_rows = raw_records
    if source_rows is None:
        source_rows = [
            record.get("raw", record) if isinstance(record, dict) else record for record in records
        ]

    missing_counts = {field: 0 for field in sorted(REQUIRED_FIELDS)}
    duplicate_count = 0
    seen_rows: set[str] = set()
    periods: dict[str, dict[str, Any]] = {}
    invalid_source_rows = 0
    unassigned_period_records = 0
    for row in source_rows:
        if not isinstance(row, dict):
            invalid_source_rows += 1
            unassigned_period_records += 1
            continue
        missing = {
            field
            for field in REQUIRED_FIELDS
            if field not in row or row[field] is None or not str(row[field]).strip()
        }
        for field in missing:
            missing_counts[field] += 1
        serialized = json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        is_duplicate = serialized in seen_rows
        if is_duplicate:
            duplicate_count += 1
        else:
            seen_rows.add(serialized)

        period = _source_reporting_period(row)
        if not period:
            unassigned_period_records += 1
            continue
        period_report = periods.setdefault(
            period,
            {"record_count": 0, "missing_fields": {}, "duplicate_records": 0},
        )
        period_report["record_count"] += 1
        period_report["duplicate_records"] += int(is_duplicate)
        for field in missing:
            period_report["missing_fields"][field] = (
                period_report["missing_fields"].get(field, 0) + 1
            )

    period_reports = []
    for period, report in sorted(periods.items()):
        missing = set(report["missing_fields"])
        unavailable = {
            indicator: sorted(fields & missing)
            for indicator, fields in QUALITY_INDICATOR_FIELDS.items()
            if fields & missing
        }
        report["incomplete"] = bool(missing)
        report["unavailable_indicators"] = unavailable
        period_reports.append({"period": period, **report})

    indicator_availability = {}
    for indicator, required_fields in QUALITY_INDICATOR_FIELDS.items():
        absent = sorted(field for field in required_fields if missing_counts[field])
        indicator_availability[indicator] = {
            "available": bool(source_rows) and not absent and not invalid_source_rows,
            "status": (
                "available"
                if source_rows and not absent and not invalid_source_rows
                else "unavailable"
            ),
            "missing_fields": absent,
            "invalid_source_record_count": invalid_source_rows,
        }

    accepted_count = provenance.get("accepted_count", len(records))
    rejected_count = provenance.get("rejected_count", max(0, len(source_rows) - len(records)))
    if not source_rows:
        quality_status = "empty"
    elif (
        any(missing_counts.values())
        or duplicate_count
        or rejected_count
        or invalid_source_rows
        or provenance.get("validation_status", "unknown") not in {"valid", "unchanged_source"}
    ):
        quality_status = "issues_found"
    else:
        quality_status = "passed"
    return {
        "methodology_version": "mintic-data-quality-v1",
        "municipality": "Tunja",
        "municipality_code": "15001",
        "geographic_granularity": "municipality",
        "source_record_count": len(source_rows),
        "accepted_record_count": accepted_count,
        "rejected_record_count": rejected_count,
        "invalid_source_record_count": invalid_source_rows,
        "unassigned_period_record_count": unassigned_period_records,
        "duplicate_record_count": duplicate_count,
        "missing_fields": missing_counts,
        "available_reporting_periods": [report["period"] for report in period_reports],
        "periods": period_reports,
        "incomplete_periods": [
            report["period"] for report in period_reports if report["incomplete"]
        ],
        "indicator_availability": indicator_availability,
        "validation_status": provenance.get("validation_status", "unknown"),
        "quality_status": quality_status,
        "source_rejection_reasons": provenance.get("rejection_reasons", {}),
        "interpretation": (
            "The report inspects official MinTIC source rows and local validation metadata without "
            "editing or filling records. Missing values are unavailable, not numeric zero. Period "
            "indicators affected by missing required fields are not calculable. Official access "
            "records are separate from synthetic ML datasets and predictions."
        ),
    }


def _source_reporting_period(row: dict[str, Any]) -> str | None:
    try:
        year = int(row["anno"])
        quarter = int(row["trimestre"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if year < 1 or quarter not in {1, 2, 3, 4}:
        return None
    return f"{year}-Q{quarter}"


def analyze_tunja(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize subscription counts and reported rate bands without saturation claims."""
    by_provider: dict[str, int] = defaultdict(int)
    by_technology: dict[str, int] = defaultdict(int)
    by_period: dict[str, int] = defaultdict(int)
    by_provider_period: dict[tuple[str, str], int] = defaultdict(int)
    by_technology_period: dict[tuple[str, str], int] = defaultdict(int)
    downstream_by_period: dict[tuple[str, str], int] = defaultdict(int)
    upstream_by_period: dict[tuple[str, str], int] = defaultdict(int)
    total = 0
    for row in records:
        count = int(row["reported_accesses"])
        total += count
        by_provider[row["provider"]] += count
        by_technology[row["technology"]] += count
        by_period[row["period"]] += count
        by_provider_period[(row["period"], row["provider"])] += count
        by_technology_period[(row["period"], row["technology"])] += count
        downstream_by_period[(row["period"], f'{row["downstream_mbps"]} Mbps')] += count
        upstream_by_period[(row["period"], f'{row["upstream_mbps"]} Mbps')] += count
    latest_period = max(by_period) if by_period else ""
    return {
        "municipality": "Tunja",
        "municipality_code": "15001",
        "geographic_granularity": "municipality",
        "observation_count": len(records),
        "sum_of_reported_access_counts_across_quarterly_rows": total,
        "provider_accesses_across_periods": _sorted(by_provider),
        "technology_accesses_across_periods": _sorted(by_technology),
        "latest_reporting_period": latest_period or None,
        "latest_period_reported_access_total": by_period.get(latest_period, 0),
        "latest_period_provider_accesses": _sorted(
            {
                provider: count
                for (period, provider), count in by_provider_period.items()
                if period == latest_period
            }
        ),
        "latest_period_technology_accesses": _sorted(
            {
                technology: count
                for (period, technology), count in by_technology_period.items()
                if period == latest_period
            }
        ),
        "quarterly_accesses": [
            {"period": key, "reported_accesses": by_period[key]} for key in sorted(by_period)
        ],
        "latest_period_downstream_values": _sorted(
            {
                value: count
                for (period, value), count in downstream_by_period.items()
                if period == latest_period
            }
        ),
        "latest_period_upstream_values": _sorted(
            {
                value: count
                for (period, value), count in upstream_by_period.items()
                if period == latest_period
            }
        ),
        "source_id": "mintic_fixed_accesses",
        "methodology_version": "mintic-tunja-descriptive-v1",
        "interpretation": (
            "Quarterly subscription/access counts grouped by the source. Provider and technology "
            "all-period totals sum repeated quarterly observations and are not unique subscribers. "
            "Speed-band counts are reported categories. The data do not measure traffic, "
            "simultaneous users, physical availability, congestion, or saturation."
        ),
    }


def compare_tunja_periods(
    records: list[dict[str, Any]], baseline_period: str, comparison_period: str
) -> dict[str, Any]:
    """Compare published period aggregates without filling absent categories or periods."""
    valid_rows: list[dict[str, Any]] = []
    invalid_records = 0
    for row in records:
        normalized = _validated_comparison_row(row)
        if normalized is None:
            invalid_records += 1
        else:
            valid_rows.append(normalized)

    available_periods = sorted({row["period"] for row in valid_rows})
    periods_available = {
        baseline_period: baseline_period in available_periods,
        comparison_period: comparison_period in available_periods,
    }
    response = {
        "methodology_version": "mintic-period-comparison-v1",
        "geographic_granularity": "municipality",
        "baseline_period": baseline_period,
        "comparison_period": comparison_period,
        "available_periods": available_periods,
        "periods_available": periods_available,
        "invalid_records_ignored": invalid_records,
        "indicators": {},
        "interpretation": (
            "Reported fixed-internet accesses and source bandwidth categories are compared as "
            "municipal quarterly aggregates. They do not measure WiFi traffic, unique people, "
            "congestion, or saturation. Model data and predictions remain synthetic."
        ),
    }
    if not all(periods_available.values()):
        response["status"] = "period_unavailable"
        response["message"] = (
            "Comparison unavailable because one or both requested reporting periods are absent "
            "from the validated snapshot. No values were interpolated or inferred."
        )
        return response

    by_period = {
        period: [row for row in valid_rows if row["period"] == period]
        for period in (baseline_period, comparison_period)
    }
    summaries = {period: _period_indicators(rows) for period, rows in by_period.items()}
    response["status"] = "available"
    response["indicators"] = {
        "reported_accesses_total": _difference(
            summaries[baseline_period]["reported_accesses_total"],
            summaries[comparison_period]["reported_accesses_total"],
        ),
        "provider_reported_accesses": _category_differences(
            summaries[baseline_period]["providers"], summaries[comparison_period]["providers"]
        ),
        "technology_reported_accesses": _category_differences(
            summaries[baseline_period]["technologies"],
            summaries[comparison_period]["technologies"],
        ),
        "downstream_band_reported_accesses": _category_differences(
            summaries[baseline_period]["downstream_bands"],
            summaries[comparison_period]["downstream_bands"],
        ),
        "upstream_band_reported_accesses": _category_differences(
            summaries[baseline_period]["upstream_bands"],
            summaries[comparison_period]["upstream_bands"],
        ),
    }
    return response


def available_tunja_periods(records: list[dict[str, Any]]) -> list[str]:
    """Return only periods represented by records with complete comparison fields."""
    return sorted(
        {
            row["period"]
            for record in records
            if (row := _validated_comparison_row(record)) is not None
        }
    )


def _validated_comparison_row(row: Any) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    period = row.get("period")
    if not isinstance(period, str) or len(period) != 7 or period[4:6] != "-Q":
        return None
    try:
        year = int(period[:4])
        quarter = int(period[6:])
        count_value = float(row["reported_accesses"])
        downstream = float(row["downstream_mbps"])
        upstream = float(row["upstream_mbps"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        year < 1
        or quarter not in {1, 2, 3, 4}
        or not math.isfinite(count_value)
        or count_value < 0
        or not count_value.is_integer()
        or not math.isfinite(downstream)
        or downstream < 0
        or not math.isfinite(upstream)
        or upstream < 0
    ):
        return None
    provider = row.get("provider")
    technology = row.get("technology")
    if not isinstance(provider, str) or not provider.strip():
        return None
    if not isinstance(technology, str) or not technology.strip():
        return None
    return {
        "period": period,
        "reported_accesses": int(count_value),
        "provider": provider,
        "technology": technology,
        "downstream_mbps": downstream,
        "upstream_mbps": upstream,
    }


def _period_indicators(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "reported_accesses_total": 0,
        "providers": defaultdict(int),
        "technologies": defaultdict(int),
        "downstream_bands": defaultdict(int),
        "upstream_bands": defaultdict(int),
    }
    for row in rows:
        count = row["reported_accesses"]
        summary["reported_accesses_total"] += count
        summary["providers"][row["provider"]] += count
        summary["technologies"][row["technology"]] += count
        summary["downstream_bands"][row["downstream_mbps"]] += count
        summary["upstream_bands"][row["upstream_mbps"]] += count
    return summary


def _difference(baseline: int | float, comparison: int | float) -> dict[str, Any]:
    absolute = comparison - baseline
    percentage = (absolute / baseline * 100) if baseline != 0 else None
    return {
        "baseline_value": baseline,
        "comparison_value": comparison,
        "absolute_difference": absolute,
        "percentage_difference": round(percentage, 4) if percentage is not None else None,
        "percentage_status": "available" if percentage is not None else "undefined_zero_baseline",
        "status": "available",
    }


def _category_differences(
    baseline: dict[Any, int], comparison: dict[Any, int]
) -> list[dict[str, Any]]:
    changes = []
    for category in sorted(set(baseline) | set(comparison), key=str):
        in_baseline = category in baseline
        in_comparison = category in comparison
        if not (in_baseline and in_comparison):
            changes.append(
                {
                    "category": category,
                    "baseline_value": baseline.get(category),
                    "comparison_value": comparison.get(category),
                    "absolute_difference": None,
                    "percentage_difference": None,
                    "percentage_status": "unavailable_missing_category_period",
                    "status": "missing_in_baseline" if not in_baseline else "missing_in_comparison",
                }
            )
            continue
        changes.append(
            {"category": category, **_difference(baseline[category], comparison[category])}
        )
    return changes


def _sorted(values: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"name": name, "reported_accesses": count}
        for name, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ]
