"""Bounded, privacy-safe in-process HTTP request metrics."""

from __future__ import annotations

from threading import Lock
from typing import Any

MAX_ROUTES_PER_SERVICE = 64


class MetricsRegistry:
    """Aggregate route-template counters without retaining request-level data."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._services: dict[str, dict[str, dict[str, Any]]] = {}

    def record(
        self,
        *,
        service: str,
        route: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """Record one completed response using bounded dimensions only."""
        safe_service = service if service in {"main_api", "assistant_api"} else "other"
        safe_route = route[:128] if route.startswith("/") else "unmatched"
        try:
            code = int(status_code)
        except (TypeError, ValueError):
            code = 500
        if not 100 <= code <= 599:
            code = 500
        try:
            elapsed = max(0.0, float(duration_ms))
        except (TypeError, ValueError):
            elapsed = 0.0

        with self._lock:
            routes = self._services.setdefault(safe_service, {})
            if safe_route not in routes and len(routes) >= MAX_ROUTES_PER_SERVICE - 1:
                safe_route = "other"
            item = routes.setdefault(
                safe_route,
                {
                    "requests_total": 0,
                    "errors_total": 0,
                    "responses_by_status": {},
                    "duration_ms": {"count": 0, "total": 0.0, "max": 0.0},
                },
            )
            item["requests_total"] += 1
            item["errors_total"] += int(code >= 400)
            status_counts = item["responses_by_status"]
            status_key = str(code)
            status_counts[status_key] = status_counts.get(status_key, 0) + 1
            duration = item["duration_ms"]
            duration["count"] += 1
            duration["total"] = round(duration["total"] + elapsed, 3)
            duration["max"] = round(max(duration["max"], elapsed), 3)

    def snapshot(self, service: str) -> dict[str, Any]:
        """Return a detached aggregate snapshot for a registered service."""
        safe_service = service if service in {"main_api", "assistant_api"} else "other"
        with self._lock:
            metrics = self._services.get(safe_service, {})
            endpoints = {
                route: {
                    **values,
                    "responses_by_status": dict(sorted(values["responses_by_status"].items())),
                    "duration_ms": dict(values["duration_ms"]),
                }
                for route, values in sorted(metrics.items())
            }
        return {
            "service": safe_service,
            "requests_total": sum(item["requests_total"] for item in endpoints.values()),
            "errors_total": sum(item["errors_total"] for item in endpoints.values()),
            "endpoints": endpoints,
            "metrics_scope": "process_local_since_start",
        }


request_metrics = MetricsRegistry()


def metrics_snapshot(service: str) -> dict[str, Any]:
    """Return aggregate metrics from this process for the named API service."""
    return request_metrics.snapshot(service)
