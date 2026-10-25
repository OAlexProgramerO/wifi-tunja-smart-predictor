"""Bounded server-side HTTP clients used by dashboard API-backed interactions."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

DEFAULT_API_URLS = {
    "main": ("MAIN_API_URL", "http://127.0.0.1:8000"),
    "assistant": ("ASSISTANT_API_URL", "http://127.0.0.1:8001"),
}
REQUEST_TIMEOUT_SECONDS = 5
MAX_TEXT_REPORT_BYTES = 1_000_000


class ApiClientError(RuntimeError):
    """A safe, concise error from a configured backend request."""


def api_base_url(service: str, environ: Mapping[str, str] | None = None) -> str:
    """Read and validate a service URL without silently substituting a fallback."""
    if service not in DEFAULT_API_URLS:
        raise ValueError("service must be 'main' or 'assistant'")
    variable, default = DEFAULT_API_URLS[service]
    values = os.environ if environ is None else environ
    configured = values.get(variable, default).strip()
    try:
        parsed = urlsplit(configured)
        valid_port = parsed.port is None or 1 <= parsed.port <= 65535
    except ValueError:
        valid_port = False
        parsed = urlsplit("")
    if (
        not configured
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or not valid_port
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
    ):
        raise ValueError(f"{variable} must be an absolute HTTP(S) origin with an optional port.")
    return configured.rstrip("/")


def request_json(
    service: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> dict:
    """Call one fixed backend route and decode a JSON object with a short timeout."""
    if not path.startswith("/") or path.startswith("//") or ".." in path.split("/"):
        raise ValueError("path must be an application route beginning with one slash")
    variable, _ = DEFAULT_API_URLS[service]
    try:
        url = api_base_url(service) + path
    except ValueError as exc:
        raise ApiClientError(str(exc)) from None
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=body,
        headers={
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body else {}),
        },
        method="POST" if body else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read())
    except HTTPError as exc:
        raise ApiClientError(f"{service.title()} API returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError):
        raise ApiClientError(
            f"{service.title()} API is unavailable at the configured {variable} endpoint."
        ) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ApiClientError(f"{service.title()} API returned an invalid response.") from None
    if not isinstance(decoded, dict):
        raise ApiClientError(f"{service.title()} API returned an invalid response.")
    return decoded


def main_api_request(path: str, payload: dict | None = None) -> dict:
    """Call the main prediction API."""
    return request_json("main", path, payload)


def main_api_text_request(path: str, *, timeout: float = REQUEST_TIMEOUT_SECONDS) -> str:
    """Fetch a bounded plain-text report without attempting JSON decoding."""
    if not path.startswith("/") or path.startswith("//") or ".." in path.split("/"):
        raise ValueError("path must be an application route beginning with one slash")
    try:
        url = api_base_url("main") + path
    except ValueError as exc:
        raise ApiClientError(str(exc)) from None
    request = Request(url, headers={"Accept": "text/markdown"}, method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_TEXT_REPORT_BYTES + 1)
            if len(body) > MAX_TEXT_REPORT_BYTES:
                raise ApiClientError("Main API text report exceeds the configured size limit.")
            return body.decode("utf-8")
    except HTTPError as exc:
        raise ApiClientError(f"Main API returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError, OSError):
        raise ApiClientError(
            "Main API is unavailable at the configured MAIN_API_URL endpoint."
        ) from None
    except UnicodeDecodeError:
        raise ApiClientError("Main API returned an invalid text response.") from None


def assistant_api_request(path: str, payload: dict | None = None) -> dict:
    """Call the deterministic assistant API."""
    return request_json("assistant", path, payload)
