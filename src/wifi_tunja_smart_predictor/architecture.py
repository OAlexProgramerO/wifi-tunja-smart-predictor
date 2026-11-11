"""Read-only, sanitized diagnostics for the deployed application components."""

from __future__ import annotations

import json
import os
import socket
import sqlite3
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from wifi_tunja_smart_predictor import deployment
from wifi_tunja_smart_predictor.config import experiment_database_path
from wifi_tunja_smart_predictor.connectivity import mintic

ASSISTANT_PROBE_TIMEOUT_SECONDS = 1.5


def _component(
    component_id: str,
    name: str,
    responsibility: str,
    status: str,
    check_status: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "id": component_id,
        "name": name,
        "responsibility": responsibility,
        "status": status,
        "check_status": check_status,
        "message": message,
        **details,
    }


def probe_assistant_api(base_url: str | None) -> dict[str, str]:
    """Probe the configured assistant readiness route without echoing its URL or errors."""
    if not base_url or not base_url.strip():
        return {
            "status": "unknown",
            "check_status": "unavailable",
            "message": "No assistant readiness endpoint is configured for this API process.",
        }
    try:
        parsed = urlsplit(base_url.strip())
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        if parsed.port is not None and not 1 <= parsed.port <= 65535:
            raise ValueError
    except ValueError:
        return {
            "status": "unknown",
            "check_status": "unavailable",
            "message": "The assistant readiness endpoint is not configured with a valid HTTP origin.",
        }

    endpoint = base_url.strip().rstrip("/") + "/ready"
    request = Request(endpoint, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(request, timeout=ASSISTANT_PROBE_TIMEOUT_SECONDS) as response:
            if getattr(response, "status", response.getcode()) != 200:
                return {
                    "status": "failed",
                    "check_status": "verified",
                    "message": "The assistant readiness route returned an unexpected HTTP status.",
                }
            payload = json.loads(response.read(4097))
    except HTTPError as exc:
        if exc.code == 503:
            return {
                "status": "not_ready",
                "check_status": "verified",
                "message": "The assistant API responded but reports required resources are not ready.",
            }
        return {
            "status": "failed",
            "check_status": "verified",
            "message": "The assistant readiness route returned an HTTP error.",
        }
    except (URLError, TimeoutError, socket.timeout, OSError):
        return {
            "status": "unknown",
            "check_status": "unavailable",
            "message": "The configured assistant readiness check could not reach the service.",
        }
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
        return {
            "status": "failed",
            "check_status": "verified",
            "message": "The assistant readiness route returned an invalid response.",
        }

    if isinstance(payload, dict) and payload.get("status") == "ready":
        return {
            "status": "available",
            "check_status": "verified",
            "message": "The assistant API readiness route reports ready.",
        }
    if isinstance(payload, dict) and payload.get("status") == "not_ready":
        return {
            "status": "not_ready",
            "check_status": "verified",
            "message": "The assistant API responded but reports required resources are not ready.",
        }
    return {
        "status": "failed",
        "check_status": "verified",
        "message": "The assistant readiness route returned an invalid response.",
    }


def _inspect_sqlite_store(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {
            "status": "unavailable",
            "check_status": "unavailable",
            "message": "The local database has not been initialized.",
        }
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=0.5) as db:
            db.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
    except sqlite3.DatabaseError:
        return {
            "status": "failed",
            "check_status": "verified",
            "message": "The local database could not be read.",
        }
    except OSError:
        return {
            "status": "unknown",
            "check_status": "unavailable",
            "message": "The local database read check could not be completed.",
        }
    return {
        "status": "available",
        "check_status": "verified",
        "message": "The local database accepted a read-only SQLite query.",
    }


def _synthetic_pipeline_status() -> dict[str, str]:
    try:
        ready = deployment.runtime_assets_available() and deployment.model_artifacts_available()
    except Exception:
        return {
            "status": "failed",
            "check_status": "verified",
            "message": "The synthetic model readiness check raised an error.",
        }
    if ready:
        return {
            "status": "available",
            "check_status": "verified",
            "message": "Required synthetic runtime assets and persisted models passed readiness checks.",
        }
    return {
        "status": "not_ready",
        "check_status": "verified",
        "message": "Required synthetic runtime assets or persisted models are not ready.",
    }


def _official_snapshot_status() -> dict[str, str]:
    try:
        snapshot = mintic.load_snapshot()
    except (OSError, ValueError, sqlite3.DatabaseError):
        return {
            "status": "failed",
            "check_status": "verified",
            "message": "The local official-data snapshot could not be read.",
        }
    if snapshot is None:
        return {
            "status": "unavailable",
            "check_status": "verified",
            "message": "No validated local MinTIC snapshot is currently available.",
        }
    if snapshot.get("stale"):
        return {
            "status": "stale_cached",
            "check_status": "verified",
            "message": "A validated local snapshot is available but marked stale; it is not live data.",
        }
    return {
        "status": "available_cached",
        "check_status": "verified",
        "message": "A validated local MinTIC snapshot is available; the source was not checked live.",
    }


def build_architecture_diagnostics(
    *, main_api_status: dict[str, str] | None = None
) -> dict[str, Any]:
    """Return known component roles and safe status checks without exposing configuration."""
    assistant = probe_assistant_api(os.getenv("ASSISTANT_API_URL"))
    synthetic = _synthetic_pipeline_status()
    snapshot = _official_snapshot_status()
    main_api = main_api_status or {
        "status": "unknown",
        "check_status": "unavailable",
        "message": "The main API process status is available only through the live diagnostics route.",
    }
    stores = {
        "experiment_history": _inspect_sqlite_store(experiment_database_path()),
        "official_connectivity_snapshot": _inspect_sqlite_store(mintic._database_path()),
    }
    if any(store["status"] == "failed" for store in stores.values()):
        persistence_status, persistence_check = "failed", "verified"
    elif any(store["status"] == "available" for store in stores.values()):
        persistence_status, persistence_check = "available", "verified"
    else:
        persistence_status, persistence_check = "unavailable", "unavailable"

    return {
        "diagnostics_version": "architecture-diagnostics-v1",
        "components": [
            _component(
                "streamlit_dashboard",
                "Streamlit dashboard",
                "Presents user workflows and communicates with both FastAPI services.",
                "unknown",
                "unavailable",
                "The main API cannot independently verify the dashboard process.",
            ),
            _component(
                "main_api",
                "Main FastAPI service",
                "Serves synthetic predictions, experiments, and read-only official-data analytics.",
                main_api["status"],
                main_api["check_status"],
                main_api["message"],
            ),
            _component(
                "assistant_api",
                "Assistant API",
                "Provides deterministic chat and allowlisted dataset queries.",
                assistant["status"],
                assistant["check_status"],
                assistant["message"],
            ),
            _component(
                "synthetic_ml_pipeline",
                "Synthetic ML pipeline and persisted models",
                "Creates synthetic training data and uses persisted classifier/regressor artifacts for predictions.",
                synthetic["status"],
                synthetic["check_status"],
                synthetic["message"],
            ),
            _component(
                "official_mintic_ingestion",
                "Official MinTIC ingestion",
                "Explicitly refreshes and validates municipal fixed-internet records; it is not a continuous feed.",
                "on_demand",
                "not_applicable",
                "Ingestion is operator-triggered; this API does not report a live ingestion process.",
            ),
            _component(
                "official_mintic_snapshot",
                "Official MinTIC local snapshot",
                "Persists raw and normalized source records and provenance separately from synthetic ML data.",
                snapshot["status"],
                snapshot["check_status"],
                snapshot["message"],
            ),
            _component(
                "sqlite_persistence",
                "SQLite persistence",
                "Stores experiment history and official connectivity snapshots in separate local databases.",
                persistence_status,
                persistence_check,
                "Read-only database checks are reported individually.",
                stores=stores,
            ),
            _component(
                "docker_compose",
                "Docker Compose services",
                "Runs runtime-init, main-api, assistant-api, and dashboard with named persistent volumes.",
                "unknown",
                "unavailable",
                "The application does not access the Docker daemon; use Compose health checks for service state.",
            ),
        ],
        "status_legend": {
            "available": "A named check completed successfully.",
            "available_cached": "A validated local snapshot is readable; this is not a live source check.",
            "stale_cached": "A readable validated snapshot is marked stale.",
            "not_ready": "A readiness check completed and required resources are not ready.",
            "failed": "A check completed and reported an error.",
            "unavailable": "A check ran but the dependency/data is not present.",
            "unknown": "No reliable status check was available.",
            "on_demand": "The component runs only when explicitly invoked.",
        },
        "data_boundary": (
            "Synthetic datasets, persisted ML models, predictions, and scenarios are separate from "
            "official MinTIC descriptive municipal subscription records."
        ),
    }
