"""Read-only operational diagnostics for the main API and assistant service."""

from __future__ import annotations

import json
import os
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from fastapi import APIRouter

from wifi_tunja_smart_predictor import deployment
from wifi_tunja_smart_predictor.observability.metrics import metrics_snapshot

router = APIRouter(tags=["observability"])
ASSISTANT_PROBE_TIMEOUT_SECONDS = 1.5
MAX_PROBE_RESPONSE_BYTES = 256_000


def _unavailable(message: str) -> dict[str, Any]:
    return {
        "availability": "unknown",
        "check_status": "unavailable",
        "readiness": "unknown",
        "message": message,
        "metrics": None,
    }


def _assistant_service_diagnostics(base_url: str | None) -> dict[str, Any]:
    """Fetch a bounded diagnostics payload without returning its configured URL."""
    if not base_url or not base_url.strip():
        return _unavailable("Assistant service diagnostics are not configured.")
    try:
        parts = urlsplit(base_url.strip())
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.username
            or parts.password
            or parts.query
            or parts.fragment
        ):
            raise ValueError
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError
    except ValueError:
        return _unavailable("Assistant service diagnostics endpoint is invalid or unavailable.")

    endpoint = base_url.strip().rstrip("/") + "/observability/metrics"
    request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=ASSISTANT_PROBE_TIMEOUT_SECONDS) as response:
            status = getattr(response, "status", response.getcode())
            if status != 200:
                return {
                    "availability": "failed",
                    "check_status": "verified",
                    "readiness": "unknown",
                    "message": "Assistant diagnostics returned an unexpected HTTP status.",
                    "metrics": None,
                }
            payload = json.loads(response.read(MAX_PROBE_RESPONSE_BYTES + 1))
    except HTTPError as exc:
        return {
            "availability": "not_ready" if exc.code == 503 else "failed",
            "check_status": "verified",
            "readiness": "not_ready" if exc.code == 503 else "unknown",
            "message": "Assistant diagnostics returned an HTTP error.",
            "metrics": None,
        }
    except (URLError, TimeoutError, socket.timeout, OSError):
        return _unavailable("Assistant diagnostics could not reach the service.")
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return {
            "availability": "failed",
            "check_status": "verified",
            "readiness": "unknown",
            "message": "Assistant diagnostics returned an invalid response.",
            "metrics": None,
        }

    if (
        not isinstance(payload, dict)
        or payload.get("service") != "assistant_api"
        or payload.get("availability") != "available"
        or payload.get("readiness") not in {"ready", "not_ready", "unknown"}
        or not isinstance(payload.get("metrics"), dict)
    ):
        return {
            "availability": "failed",
            "check_status": "verified",
            "readiness": "unknown",
            "message": "Assistant diagnostics returned an invalid response.",
            "metrics": None,
        }

    return {
        "availability": "available",
        "check_status": "verified",
        "readiness": payload["readiness"],
        "message": "Assistant metrics endpoint responded to a bounded diagnostics check.",
        "metrics": payload["metrics"],
    }


def _readiness_status() -> str:
    try:
        if deployment.runtime_assets_available() and deployment.model_artifacts_available():
            return "ready"
    except Exception:
        return "unknown"
    return "not_ready"


@router.get("/observability/diagnostics")
def observability_diagnostics() -> dict[str, Any]:
    """Return local aggregate metrics and a verified/unavailable assistant check."""
    return {
        "schema_version": "1.0",
        "metrics_scope": "process_local_since_start",
        "services": {
            "main_api": {
                "availability": "available",
                "check_status": "verified",
                "readiness": _readiness_status(),
                "message": "This response confirms the main API diagnostics route is responding.",
                "metrics": metrics_snapshot("main_api"),
            },
            "assistant_api": _assistant_service_diagnostics(os.getenv("ASSISTANT_API_URL")),
        },
        "privacy": {
            "request_content_retained": False,
            "sensitive_headers_retained": False,
            "dimensions": ["registered_route_template", "http_status"],
        },
    }
