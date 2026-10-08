"""Small standard-library JSON logging helpers with safe event fields."""

from __future__ import annotations

import json
import logging
import math
import threading
from datetime import datetime, timezone
from os.path import basename
from traceback import extract_tb
from typing import Any

_CONFIG_LOCK = threading.Lock()
_FORMATTER_MARKER = "_wifi_predictor_json_formatter"


class JsonLogFormatter(logging.Formatter):
    """Render each record as one parseable JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        fields = getattr(record, "event_fields", None)
        output: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
        }
        if isinstance(fields, dict):
            output.update(fields)
            output.setdefault("event", "application.event")
        else:
            output["event"] = "application.log"
            output["message"] = record.getMessage()
        return json.dumps(
            output,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
            allow_nan=False,
        )


def configure_structured_logging() -> None:
    """Install one JSON formatter on configured root handlers, safely repeatable."""
    root = logging.getLogger()
    with _CONFIG_LOCK:
        if not root.handlers:
            root.addHandler(logging.StreamHandler())
        for handler in root.handlers:
            formatter = getattr(handler, "formatter", None)
            if not getattr(formatter, _FORMATTER_MARKER, False):
                handler.setFormatter(JsonLogFormatter())
                setattr(handler.formatter, _FORMATTER_MARKER, True)
        # Lifecycle events replace Uvicorn access lines, which include client IPs.
        logging.getLogger("uvicorn.access").disabled = True
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)


def emit_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    **fields: Any,
) -> None:
    """Emit a JSON event with bounded primitive fields and no arbitrary object dumps."""
    configure_structured_logging()
    safe_fields: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            continue
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, str):
                value = value[:160]
            safe_fields[key] = value
        elif isinstance(value, (list, tuple)):
            safe_fields[key] = [str(item)[:80] for item in value[:8]]
    logger.log(level, event, extra={"event_fields": safe_fields})


def exception_diagnostics(exc: BaseException) -> dict[str, Any]:
    """Return bounded traceback location metadata without exception text or locals."""
    frames = extract_tb(exc.__traceback__)[-6:]
    return {
        "exception_type": type(exc).__name__[:80],
        "stack_frames": [
            f"{basename(frame.filename)[:80]}:{frame.lineno}:{frame.name[:80]}" for frame in frames
        ],
    }
