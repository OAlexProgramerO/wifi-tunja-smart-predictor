"""Descriptive analysis of compatible quarterly MinTIC Tunja observations."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


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


def _sorted(values: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {"name": name, "reported_accesses": count}
        for name, count in sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ]
