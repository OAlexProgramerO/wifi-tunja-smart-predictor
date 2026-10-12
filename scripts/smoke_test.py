#!/usr/bin/env python3
"""Exercise a running local Compose deployment through its HTTP interfaces."""

from __future__ import annotations

import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

from wifi_tunja_smart_predictor.api_client import (
    ApiClientError,
    api_base_url,
    assistant_api_request,
    main_api_request,
)
from wifi_tunja_smart_predictor.config import (
    CATEGORICAL_ALLOWED_VALUES,
    MODEL_INPUT_COLUMNS,
    PROJECT_VERSION,
)

DEFAULT_DASHBOARD_URL = "http://127.0.0.1:8501"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _valid_prediction_payload() -> dict:
    payload = {
        feature: (
            sorted(CATEGORICAL_ALLOWED_VALUES[feature])[0]
            if feature in CATEGORICAL_ALLOWED_VALUES
            else 0
        )
        for feature in MODEL_INPUT_COLUMNS
    }
    payload["month"] = 1
    payload["time_period"] = "NIGHT"
    payload["traffic_level"] = "LOW"
    payload["event_type"] = "NONE"
    payload["access_point_capacity"] = 150
    payload["signal_strength_dbm"] = -60
    return payload


def _dashboard_health(dashboard_url: str) -> None:
    try:
        parsed = urlsplit(dashboard_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        with urlopen(f"{dashboard_url.rstrip('/')}/_stcore/health", timeout=5) as response:
            _require(response.status == 200, "expected HTTP 200")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        raise ApiClientError("Dashboard HTTP health check failed.") from None


def _checks(dashboard_url: str) -> tuple[tuple[str, object], ...]:
    scenario_payload = {
        "datetime": "2025-06-15T18:00:00-05:00",
        "location": {"zone": "downtown"},
    }
    greeting = {"message": "hello"}
    return (
        (
            "Main API health",
            lambda: _require(main_api_request("/health").get("status") == "ok", "unexpected body"),
        ),
        (
            "Main API readiness",
            lambda: _require(
                main_api_request("/ready").get("status") == "ready", "required assets not ready"
            ),
        ),
        (
            "Assistant API health",
            lambda: _require(
                assistant_api_request("/health").get("status") == "ok",
                "unexpected body",
            ),
        ),
        (
            "Assistant API readiness",
            lambda: _require(
                assistant_api_request("/ready").get("status") == "ready",
                "required assets not ready",
            ),
        ),
        (
            "Model information",
            _check_model_info,
        ),
        (
            "Standard prediction",
            lambda: _require(
                main_api_request("/predict", _valid_prediction_payload()).get("prediction")
                in {"LOW", "HIGH"},
                "prediction response is invalid",
            ),
        ),
        (
            "Scenario prediction",
            lambda: _require(
                "classification" in main_api_request("/scenario/predict", scenario_payload),
                "scenario response is invalid",
            ),
        ),
        (
            "Assistant greeting",
            lambda: _require(
                (answer := assistant_api_request("/chat", greeting)).get("intent") == "GREETING"
                and bool(answer.get("answer")),
                "greeting response is invalid",
            ),
        ),
        ("Dashboard HTTP and backend network", lambda: _dashboard_health(dashboard_url)),
    )


def _check_model_info() -> None:
    info = main_api_request("/model-info")
    _require(
        info.get("model_loaded") is True and info.get("regression_model_loaded") is True,
        "one or more trained artifacts could not be loaded",
    )
    _require(info.get("project_version") == PROJECT_VERSION, "package version does not match")


def run_smoke_tests(dashboard_url: str | None = None) -> int:
    """Run all checks, returning nonzero for any failure without printing payloads."""
    dashboard_url = dashboard_url or os.getenv("DASHBOARD_URL", DEFAULT_DASHBOARD_URL)
    try:
        api_base_url("main")
        api_base_url("assistant")
    except (ApiClientError, ValueError) as exc:
        print(f"FAIL API configuration: {exc}", file=sys.stderr)
        return 1

    failed = False
    for name, check in _checks(dashboard_url):
        try:
            check()
        except (ApiClientError, ValueError) as exc:
            print(f"FAIL {name}: {exc}", file=sys.stderr)
            failed = True
        except Exception:
            print(f"FAIL {name}: request failed; inspect service logs.", file=sys.stderr)
            failed = True
        else:
            print(f"PASS {name}")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(run_smoke_tests())
