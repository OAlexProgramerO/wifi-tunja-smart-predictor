"""Small deterministic intent and entity recognizer for project-specific questions."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

LOCAL_ZONE = ZoneInfo("America/Bogota")
ZONE_TERMS = (
    "downtown",
    "north",
    "south",
    "university",
    "commercial",
    "transport",
    "residential",
    "park",
    "public service",
    "healthcare",
    "education",
    "tourism",
    "historic quarter",
    "medical district",
    "shopping plaza",
    "campus north",
    "campus south",
    "bus terminal",
)


@dataclass(frozen=True)
class ParsedMessage:
    """Recognized high-level task and explicit scenario entities."""

    intent: str
    zone: str | None
    hour: int | None
    date: datetime | None


def parse_message(message: str, *, has_context: bool = False) -> ParsedMessage:
    """Classify supported project questions without inventing extracted values."""
    text = message.casefold().strip()
    zone = next((item for item in ZONE_TERMS if re.search(rf"\b{re.escape(item)}\b", text)), None)
    hour = _parse_hour(text)
    parsed_date = (
        datetime.now(LOCAL_ZONE).replace(tzinfo=None)
        if "today" in text or "tonight" in text
        else None
    )

    if any(word in text for word in ("limit", "limitation", "weakness", "cannot", "can't")):
        intent = "LIMITATIONS"
    elif (
        zone
        and any(term in text for term in ("i'm in", "i am in", "my location", "i am at"))
        and not any(
            word in text
            for word in ("demand", "predict", "expect", "connection", "capacity", "high")
        )
    ):
        intent = "LOCATION_INFO"
    elif any(word in text for word in ("why", "factors behind", "main factors", "drivers")) or (
        has_context and "model know" in text
    ):
        intent = "EXPLANATION" if has_context else "PREDICTION"
    elif "chart" in text or "dashboard" in text or "explain this" in text:
        intent = "DASHBOARD_HELP"
    elif ("traffic" in text and "demand" in text) or "what does the csv" in text:
        intent = "DATA_QUERY"
    elif any(
        term in text
        for term in (
            "historically",
            "pattern",
            "compare",
            "weekend",
            "average by hour",
            "month",
            "weekday",
            "day of week",
            "previous hours",
            "last few hours",
            "what happened",
        )
    ):
        intent = "DATA_QUERY"
    elif any(
        term in text
        for term in (
            "how many rows",
            "how many access point",
            "how many synthetic access points",
            "how many aps",
            "how many zones",
            "available zones",
            "list available zones",
            "show zones",
            "available access points",
            "list access points",
            "date range",
        )
    ):
        intent = "DATA_QUERY"
    elif ("connection" in text and ("expect" in text or "how many" in text)) or any(
        term in text for term in ("what about", "capacity", "expected connections")
    ):
        intent = "FOLLOW_UP" if has_context and not zone else "PREDICTION"
    elif (
        "predict" in text
        or "demand" in text
        or "high" in text
        or "wifi" in text
        or hour is not None
    ):
        intent = "PREDICTION" if zone or has_context or hour is not None else "HISTORICAL_DEMAND"
    elif (
        "compare" in text
        or "historically" in text
        or "weekend" in text
        or "pattern" in text
        or "average" in text
    ):
        intent = "DATA_QUERY"
    elif "model" in text or "f1" in text or "accuracy" in text or "version" in text:
        intent = "MODEL_INFO"
    elif any(word in text for word in ("methodology", "how does it work", "how was it built")):
        intent = "METHODOLOGY"
    elif "zone" in text or "access point" in text or "where am i" in text or "location" in text:
        intent = "LOCATION_INFO"
    elif any(term in text for term in ("rows", "date range", "how many", "csv", "traffic")):
        intent = "DATA_QUERY"
    else:
        intent = "DASHBOARD_HELP"
    return ParsedMessage(intent=intent, zone=zone, hour=hour, date=parsed_date)


def _parse_hour(text: str) -> int | None:
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if match:
        hour = int(match.group(1))
        meridiem = match.group(3)
        if not 1 <= hour <= 12:
            return None
        if meridiem == "am":
            return hour % 12
        return hour % 12 + 12
    match = re.search(r"\b(?:at\s+)?(\d{1,2}):00\b", text)
    if match and 0 <= int(match.group(1)) <= 23:
        return int(match.group(1))
    return None
