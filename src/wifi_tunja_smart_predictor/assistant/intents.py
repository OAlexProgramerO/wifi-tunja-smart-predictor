"""Small deterministic intent and entity recognizer for project-specific questions."""

from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

LOCAL_ZONE = ZoneInfo("America/Bogota")
ZONE_TERMS = (
    "historic quarter",
    "medical district",
    "shopping plaza",
    "campus north",
    "campus south",
    "north zone",
    "south zone",
    "bus terminal",
    "downtown",
    "center",
    "centro",
    "north",
    "norte",
    "south",
    "sur",
    "university",
    "universidad",
    "commercial",
    "comercial",
    "transport",
    "transporte",
    "residential",
    "park",
    "public service",
    "healthcare",
    "education",
    "tourism",
)
ZONE_ALIASES = {
    "center": "downtown",
    "centro": "downtown",
    "north zone": "north",
    "norte": "north",
    "south zone": "south",
    "sur": "south",
    "universidad": "university",
    "comercial": "commercial",
    "transporte": "transport",
}


@dataclass(frozen=True)
class ParsedMessage:
    """Recognized high-level task and explicit scenario entities."""

    intent: str
    zone: str | None
    hour: int | None
    date: datetime | None
    invalid_time: bool = False
    dashboard_section: str | None = None


def parse_message(message: str, *, has_context: bool = False) -> ParsedMessage:
    """Classify supported questions and extract simple deterministic entities."""
    text = normalize_message(message)
    comparable = _comparison_text(text)
    dashboard_section = _mentioned_section(comparable)
    raw_zone = next(
        (item for item in ZONE_TERMS if re.search(rf"\b{re.escape(item)}\b", text)), None
    )
    demand_question = _is_demand_question(text)
    if raw_zone is None and demand_question:
        unknown_match = re.search(
            r"\b(?:la zona|zona|in|at|en|del|de la)\s+(?:(?:the|la|el)\s+)?([a-záéíóúñ][\wáéíóúñ-]*)\b",
            text,
        )
        raw_zone = unknown_match.group(1) if unknown_match else None
    zone = ZONE_ALIASES.get(raw_zone, raw_zone)
    hour, invalid_time = _parse_hour(text, allow_bare=demand_question)
    parsed_date = (
        datetime.now(LOCAL_ZONE).replace(tzinfo=None)
        if "today" in text or "tonight" in text or "hoy" in text
        else None
    )

    if text in {"hi", "hello", "hey", "hola", "buenas"}:
        intent = "GREETING"
    elif text in {
        "who are you",
        "what are you",
        "who is this",
        "quién eres",
        "qué eres",
        "cómo te llamas",
    }:
        intent = "IDENTITY"
    elif text in {
        "what can you do",
        "what can you help me with",
        "qué puedes hacer",
        "en qué puedes ayudarme",
    }:
        intent = "CAPABILITIES"
    elif _metric_name(comparable):
        intent = "METRIC_EXPLANATION"
    elif dashboard_section or _is_dashboard_question(comparable):
        intent = "DASHBOARD_SECTION_HELP"
    elif _is_explicit_historical_or_dataset(text):
        intent = "DATA_QUERY"
    elif any(word in text for word in ("limit", "limitation", "weakness", "cannot", "can't")):
        intent = "LIMITATIONS"
    elif (
        zone
        and any(
            term in text for term in ("i'm in", "i am in", "my location", "i am at", "estoy en")
        )
        and not demand_question
    ):
        intent = "LOCATION_INFO"
    elif any(word in text for word in ("why", "factors behind", "main factors", "drivers")) or (
        has_context and "model know" in text
    ):
        intent = "EXPLANATION" if has_context else "PREDICTION"
    elif "chart" in text or "dashboard" in text or "explain this" in text:
        intent = "DASHBOARD_HELP"
    elif demand_question:
        intent = "DEMAND_SCENARIO"
    elif "traffic" in text and "demand" in text or "what does the csv" in text:
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
    return ParsedMessage(intent, zone, hour, parsed_date, invalid_time, dashboard_section)


def normalize_message(message: str) -> str:
    """Case-fold and trim common surrounding whitespace and punctuation."""
    return message.casefold().strip().strip(string.punctuation + "¡¿").strip()


def _comparison_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))


def _mentioned_section(text: str) -> str | None:
    sections = {
        "overview": ("overview", "dashboard overview"),
        "live_scenario": ("live scenario", "scenario prediction", "scenario"),
        "demand_explorer": ("demand explorer", "demand analysis"),
        "geographic_analysis": ("geographic analysis", "geographic", "geography"),
        "network_analysis": ("network analysis", "network"),
        "model_performance": ("model performance", "performance"),
        "ai_assistant": ("ai assistant", "assistant"),
        "advanced_prediction": ("advanced prediction", "advanced"),
        "about": ("about",),
    }
    for section, aliases in sections.items():
        if any(
            re.search(rf"\b{re.escape(alias)}\b", text) for alias in aliases if alias != "about"
        ) or (
            section == "about"
            and (text == "about" or "about section" in text or "explain about" in text)
        ):
            return section
    return None


def _metric_name(text: str) -> str | None:
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc auc",
        "roc-auc",
        "mae",
        "rmse",
        "r2",
        "r²",
    ):
        if re.search(rf"\b{re.escape(metric)}\b", text):
            return metric.replace("-", " ").replace("²", "2").upper()
    return None


def _is_dashboard_question(text: str) -> bool:
    terms = (
        "what am i looking at",
        "what does this",
        "what is this",
        "what does this mean",
        "explain this",
        "what does this chart",
        "what does this graph",
        "explain the chart",
        "explain the graph",
        "explain the indicator",
        "explain model performance",
        "what does this indicator",
        "what is the purpose",
        "why is demand high",
        "why is this high",
        "what does high mean",
        "what does low mean",
        "what does this interval",
        "this number mean",
        "what is this number",
        "why?",
        "why is this",
        "prediction interval",
        "capacity indicator",
        "capacity utilization",
        "current prediction",
        "que estoy viendo",
        "que significa esta",
        "que muestra este",
        "que representa este",
        "explicame",
        "para que sirve esta",
        "por que la demanda es alta",
        "por que este valor",
        "que significa high",
        "que significa low",
        "intervalo de prediccion",
        "indicador de capacidad",
    )
    return any(term in text for term in terms)


def _is_demand_question(text: str) -> bool:
    return bool(
        re.search(r"\b(demand|demanda|connections?|conexiones)\b", text)
        or "habrá alta" in text
        or "se esperan" in text
    )


def _is_explicit_historical_or_dataset(text: str) -> bool:
    return any(
        term in text
        for term in (
            "historically",
            "historical",
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
            "traffic and demand",
            "what does the csv",
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
            "demand by zone",
        )
    )


def _parse_hour(text: str, *, allow_bare: bool = False) -> tuple[int | None, bool]:
    """Parse common English/Spanish hour forms and mark malformed explicit times."""
    spanish = re.search(r"\b(\d{1,2})\s+de\s+la\s+(tarde|noche|mañana)\b", text)
    if spanish:
        hour = int(spanish.group(1))
        if not 1 <= hour <= 12:
            return None, True
        period = spanish.group(2)
        if period in {"tarde", "noche"}:
            return (hour if hour == 12 else hour + 12), False
        return (hour if hour < 12 else 0), False
    match = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2) or 0)
        if not 1 <= hour <= 12 or minute > 59:
            return None, True
        return (hour % 12 + (12 if match.group(3) == "pm" else 0)), False
    match = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        return (hour, False) if hour <= 23 and minute <= 59 else (None, True)
    if allow_bare:
        match = re.search(r"\b(?:at|around|a las|a eso de)\s+(\d{1,2})\b", text)
        if match is None:
            match = re.search(r"\b(\d{1,2})\b", text)
        if match:
            hour = int(match.group(1))
            return (hour, False) if hour <= 23 else (None, True)
    return None, False
