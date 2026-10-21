"""Deterministic, rules-based capacity planning; values are connection counts."""

from __future__ import annotations

from math import isfinite
from typing import Any


def plan_capacity(zones: list[dict[str, Any]], resource_budget: float = 0) -> dict[str, Any]:
    """Allocate an optional finite capacity budget by descending estimated demand.

    The simple greedy rule assigns units to the highest-demand zones first, with
    stable zone-id tie breaking. It is illustrative and does not optimize a real network.
    """
    if not isfinite(resource_budget) or resource_budget < 0:
        raise ValueError("resource_budget must be a finite non-negative connection count")
    seen: set[str] = set()
    normalized = []
    for row in zones:
        zone_id = str(row["zone_id"])
        demand = float(row["demand"])
        capacity = float(row["capacity"])
        if zone_id in seen:
            raise ValueError("zone_id values must be unique")
        seen.add(zone_id)
        if not all(isfinite(value) and value >= 0 for value in (demand, capacity)):
            raise ValueError("demand and capacity must be finite non-negative connection counts")
        normalized.append({"zone_id": zone_id, "demand": demand, "capacity": capacity})

    remaining = float(resource_budget)
    for row in sorted(
        normalized,
        key=lambda item: (-max(0.0, item["demand"] - item["capacity"]), item["zone_id"]),
    ):
        extra = min(remaining, max(0.0, row["demand"] - row["capacity"]))
        row["allocated_capacity"] = extra
        remaining -= extra
    results = []
    for row in sorted(normalized, key=lambda item: item["zone_id"]):
        effective = row["capacity"] + row["allocated_capacity"]
        demand = row["demand"]
        overload = max(0.0, demand - effective)
        if overload > 0:
            status = "SIMULATED_OVER_CAPACITY"
            explanation = "Estimated synthetic demand exceeds effective simulated capacity."
        elif effective == 0:
            status = "ZERO_CAPACITY"
            explanation = "Effective capacity is zero; utilization is undefined."
        else:
            status = "WITHIN_SIMULATED_CAPACITY"
            explanation = "Estimated synthetic demand does not exceed effective capacity."
        results.append(
            {
                **row,
                "effective_capacity": effective,
                "utilization_pct": demand / effective * 100 if effective else None,
                "headroom_connections": max(0.0, effective - demand),
                "overload_connections": overload,
                "unmet_demand_connections": overload,
                "status": status,
                "explanation": explanation,
            }
        )
    total_demand = sum(row["demand"] for row in results)
    total_capacity = sum(row["effective_capacity"] for row in results)
    total_overload = sum(row["overload_connections"] for row in results)
    return {
        "schema_version": "1.0",
        "units": "estimated connections",
        "allocation_method": "greedy_descending_demand_then_zone_id",
        "resource_budget_connections": float(resource_budget),
        "allocated_connections": float(sum(row["allocated_capacity"] for row in results)),
        "remaining_budget_connections": float(remaining),
        "aggregate": {
            "estimated_demand_connections": total_demand,
            "effective_capacity_connections": total_capacity,
            "utilization_pct": total_demand / total_capacity * 100 if total_capacity else None,
            "headroom_connections": sum(row["headroom_connections"] for row in results),
            "overload_connections": total_overload,
            "unmet_demand_connections": sum(row["unmet_demand_connections"] for row in results),
        },
        "zones": results,
        "assumptions": [
            "Demand and capacity are synthetic connection counts, not throughput.",
            "Greedy allocation is illustrative and is not a real-network optimum.",
        ],
    }
