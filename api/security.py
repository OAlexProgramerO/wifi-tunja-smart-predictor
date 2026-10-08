"""Small, shared HTTP protections for the two FastAPI applications."""

from __future__ import annotations

import json
import os
from collections import deque
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from wifi_tunja_smart_predictor.observability import RequestObservabilityMiddleware

DEFAULT_CORS_ORIGINS = ("http://localhost:8501", "http://127.0.0.1:8501")


def _cors_origins() -> list[str]:
    configured = os.getenv("CORS_ALLOWED_ORIGINS")
    origins = (
        [origin.strip() for origin in configured.split(",") if origin.strip()]
        if configured is not None
        else list(DEFAULT_CORS_ORIGINS)
    )
    for origin in origins:
        parsed = urlsplit(origin)
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username
            or parsed.password
        ):
            raise ValueError("CORS_ALLOWED_ORIGINS must contain explicit HTTP(S) origins only")
    return origins


class RequestBodyLimitMiddleware:
    """Buffer only bounded mutation bodies before passing them to the app."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                too_large = int(content_length) > self.max_bytes
            except ValueError:
                too_large = True
            if too_large:
                await self._reject(send)
                return

        buffered: list[Message] = []
        body_size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            buffered.append(message)
            body_size += len(message.get("body", b""))
            if body_size > self.max_bytes:
                await self._reject(send)
                return
            if not message.get("more_body", False):
                break

        replay = deque(buffered)

        async def receive_buffered() -> Message:
            if replay:
                return replay.popleft()
            return await receive()

        await self.app(scope, receive_buffered, send)

    @staticmethod
    async def _reject(send: Send) -> None:
        body = json.dumps(
            {"detail": {"code": "request_too_large", "message": "Request body is too large."}}
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Apply API-appropriate browser protections without constraining Swagger UI."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                existing = {key.lower() for key, _ in headers}
                for name, value in (
                    (b"x-content-type-options", b"nosniff"),
                    (b"x-frame-options", b"DENY"),
                    (b"referrer-policy", b"strict-origin-when-cross-origin"),
                    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                ):
                    if name not in existing:
                        headers.append((name, value))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_with_headers)


def install_api_security(app: FastAPI, *, max_body_bytes: int, service: str = "api") -> None:
    """Install restrictive CORS, bounded request bodies, and response headers."""
    app.add_middleware(RequestBodyLimitMiddleware, max_bytes=max_body_bytes)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins(),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )
    # Added last so correlation also covers security middleware rejections.
    app.add_middleware(
        RequestObservabilityMiddleware,
        service=service,
    )
