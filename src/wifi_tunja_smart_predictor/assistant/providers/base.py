"""Provider interface for grounded answer composition."""

from __future__ import annotations

from typing import Any, Protocol


class ResponseProvider(Protocol):
    """Compose text from a recognized intent and trusted tool results."""

    def compose(self, intent: str, tool_result: dict[str, Any]) -> str: ...
