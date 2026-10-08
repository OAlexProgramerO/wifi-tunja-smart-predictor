"""HTTP request IDs, correlation context, and one completion event per request."""

from __future__ import annotations

import logging
import re
import time
from contextvars import ContextVar
from uuid import uuid4

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from wifi_tunja_smart_predictor.observability.logging import (
    configure_structured_logging,
    emit_event,
)

_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_request_id: ContextVar[str | None] = ContextVar("wifi_predictor_request_id", default=None)
_logger = logging.getLogger("wifi_tunja_smart_predictor.http")


def current_request_id() -> str | None:
    """Return the request ID for this execution context, if an HTTP request set one."""
    return _request_id.get()


def _safe_request_id(raw_value: bytes | None) -> str:
    if raw_value is not None:
        try:
            candidate = raw_value.decode("ascii")
        except UnicodeDecodeError:
            candidate = ""
        if _REQUEST_ID_PATTERN.fullmatch(candidate):
            return candidate
    return str(uuid4())


def _route_identifier(scope: Scope) -> str:
    route = scope.get("route")
    template = getattr(route, "path", None)
    return template if isinstance(template, str) else "unmatched"


class RequestObservabilityMiddleware:
    """Correlate an HTTP request and emit exactly one completion record."""

    def __init__(self, app: ASGIApp, *, service: str) -> None:
        self.app = app
        self.service = service
        configure_structured_logging()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        raw_request_id = dict(scope.get("headers", [])).get(b"x-request-id")
        request_id = _safe_request_id(raw_request_id)
        token = _request_id.set(request_id)
        started = time.perf_counter()
        status_code = 500

        async def send_correlated(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                headers = [
                    (name, value)
                    for name, value in message.get("headers", [])
                    if name.lower() != b"x-request-id"
                ]
                headers.append((b"x-request-id", request_id.encode("ascii")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_correlated)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            outcome = (
                "success"
                if status_code < 400
                else "client_error" if status_code < 500 else "server_error"
            )
            emit_event(
                _logger,
                "http.request.completed",
                level=logging.ERROR if status_code >= 500 else logging.INFO,
                request_id=request_id,
                service=self.service,
                method=scope.get("method", "UNKNOWN"),
                route=_route_identifier(scope),
                status_code=status_code,
                duration_ms=round(duration_ms, 3),
                outcome=outcome,
                error_category="server_error" if status_code >= 500 else None,
            )
            _request_id.reset(token)
