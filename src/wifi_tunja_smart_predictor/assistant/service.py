"""Deterministic natural-language dispatcher backed only by project tools."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from threading import RLock
from typing import Any
from uuid import uuid4

import pandas as pd

from wifi_tunja_smart_predictor.assistant.intents import LOCAL_ZONE, ParsedMessage, parse_message
from wifi_tunja_smart_predictor.assistant.providers.deterministic import DeterministicProvider
from wifi_tunja_smart_predictor.assistant.queries import DatasetQuery, DatasetQueryEngine
from wifi_tunja_smart_predictor.assistant.schemas import AssistantContext, ChatRequest, ChatResponse
from wifi_tunja_smart_predictor.assistant.tools import (
    dashboard_summary_tool,
    dataset_query_tool,
    historical_demand_tool,
    location_tool,
    model_explanation_tool,
    model_info_tool,
    prediction_tool,
)
from wifi_tunja_smart_predictor.data.loader import load_analysis_dataset
from wifi_tunja_smart_predictor.exceptions import AssistantQueryError, LocationResolutionError
from wifi_tunja_smart_predictor.geospatial.locations import LocationResolver
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService

logger = logging.getLogger(__name__)


class ChatSessionStore:
    """Small process-local store for reusable scenario context, with bounded entries."""

    def __init__(self, max_sessions: int = 1024, ttl_minutes: int = 60) -> None:
        self._items: dict[str, tuple[datetime, dict[str, Any]]] = {}
        self._max_sessions = max_sessions
        self._ttl = timedelta(minutes=ttl_minutes)
        self._lock = RLock()

    def get(self, session_id: str) -> dict[str, Any]:
        """Return a shallow copy of unexpired session context."""
        with self._lock:
            entry = self._items.get(session_id)
            now = datetime.now(LOCAL_ZONE)
            if entry is None or now - entry[0] > self._ttl:
                self._items.pop(session_id, None)
                return {}
            return dict(entry[1])

    def put(self, session_id: str, context: dict[str, Any]) -> None:
        """Store context and evict the oldest session when the bound is reached."""
        with self._lock:
            if session_id not in self._items and len(self._items) >= self._max_sessions:
                oldest = min(self._items, key=lambda key: self._items[key][0])
                self._items.pop(oldest, None)
            self._items[session_id] = (datetime.now(LOCAL_ZONE), dict(context))


class AssistantService:
    """Route supported questions to fixed tools; never execute user code or guess metrics."""

    def __init__(
        self,
        frame: pd.DataFrame | None = None,
        prediction_service: ScenarioPredictionService | None = None,
        sessions: ChatSessionStore | None = None,
    ) -> None:
        self.frame = frame if frame is not None else load_analysis_dataset()
        self.predictions = prediction_service
        self.sessions = sessions or ChatSessionStore()
        self.queries = DatasetQueryEngine(self.frame)
        self.locations = LocationResolver(self.frame)
        self.provider = DeterministicProvider()

    def handle(self, request: ChatRequest) -> ChatResponse:
        """Interpret a supported project question and compose an answer from tool output."""
        session_id = request.session_id or str(uuid4())
        state = self.sessions.get(session_id)
        self._merge_context(state, request.context)
        parsed = parse_message(request.message, has_context=bool(state.get("location")))
        if parsed.zone:
            state["zone"] = parsed.zone
        try:
            intent, result, sources, scenario = self._dispatch(parsed, request.message, state)
        except (AssistantQueryError, LocationResolutionError, ValueError) as exc:
            return ChatResponse(
                answer=str(exc),
                intent=parsed.intent,
                sources=["validated_project_tools"],
                scenario=None,
                structured_result={"error": str(exc)},
                session_id=session_id,
            )
        self.sessions.put(session_id, state)
        answer = self.provider.compose(intent, result)
        return ChatResponse(
            answer=answer,
            intent=intent,
            sources=sources,
            scenario=scenario,
            structured_result=result,
            session_id=session_id,
        )

    @staticmethod
    def _merge_context(state: dict[str, Any], context: AssistantContext | None) -> None:
        if context is None:
            return
        for key, value in context.model_dump(exclude_none=True).items():
            state[key] = value.isoformat() if isinstance(value, datetime) else value

    def _dispatch(
        self, parsed: ParsedMessage, message: str, state: dict[str, Any]
    ) -> tuple[str, dict[str, Any], list[str], dict[str, Any] | None]:
        text = message.casefold()
        if parsed.intent == "LOCATION_INFO" and state.get("zone"):
            resolved = location_tool(self.locations, state["zone"])
            state["location"] = resolved
            return (
                "LOCATION_INFO",
                {"location": resolved},
                ["location_tool", "synthetic_dataset"],
                None,
            )
        if parsed.intent in {"PREDICTION", "FOLLOW_UP", "EXPLANATION"}:
            return self._predict(parsed, text, state)
        if parsed.intent == "MODEL_INFO":
            return (
                "MODEL_INFO",
                model_info_tool(),
                ["model_info_tool", "generated_model_metadata"],
                None,
            )
        if parsed.intent == "LIMITATIONS":
            return "LIMITATIONS", {"synthetic_data": True}, ["project_methodology"], None
        if parsed.intent == "METHODOLOGY":
            return (
                "METHODOLOGY",
                {
                    "answer": "The project validates and deduplicates synthetic observations, uses ordered train/validation/test periods, fits separate classification and regression pipelines on training rows, selects candidates on validation metrics, and reports final synthetic holdout results. Location/time scenarios aggregate deterministic historical analogs; target columns are outcomes only.",
                },
                ["project_methodology"],
                None,
            )
        if parsed.intent == "DATA_QUERY":
            result = self._natural_query(message, parsed, state)
            return "DATA_QUERY", result, ["dataset_query_tool", "synthetic_dataset"], None
        if parsed.intent == "HISTORICAL_DEMAND":
            zone_type = self._zone_type(parsed.zone or state.get("zone"))
            result = historical_demand_tool(
                self.frame, hour=parsed.hour if parsed.hour is not None else 18, zone_type=zone_type
            )
            return (
                "HISTORICAL_DEMAND",
                result,
                ["historical_demand_tool", "synthetic_dataset"],
                None,
            )
        if parsed.intent == "DASHBOARD_HELP":
            result = self._dashboard_help(text)
            return "DASHBOARD_HELP", result, ["dashboard_summary_tool", "synthetic_dataset"], None
        return (
            "DASHBOARD_HELP",
            self._dashboard_help(text),
            ["dashboard_summary_tool"],
            None,
        )

    def _predict(
        self, parsed: ParsedMessage, text: str, state: dict[str, Any]
    ) -> tuple[str, dict[str, Any], list[str], dict[str, Any] | None]:
        zone = parsed.zone or state.get("zone")
        location = state.get("location")
        if zone is None and location is not None:
            zone = location.get("zone_id")
        if not zone and not (
            state.get("latitude") is not None and state.get("longitude") is not None
        ):
            return (
                "PREDICTION",
                {
                    "answer": "Choose a synthetic zone such as downtown, transport, university, or residential first."
                },
                ["location_tool"],
                None,
            )
        existing_datetime = state.get("datetime")
        if isinstance(existing_datetime, str):
            try:
                existing_datetime = datetime.fromisoformat(existing_datetime)
            except ValueError:
                existing_datetime = None
        base = parsed.date or existing_datetime or datetime.now(LOCAL_ZONE).replace(tzinfo=None)
        hour = (
            parsed.hour
            if parsed.hour is not None
            else (existing_datetime.hour if existing_datetime is not None else base.hour)
        )
        when = datetime.combine(base.date(), time(hour=hour))
        request = ScenarioRequest(
            datetime=when,
            zone=zone if zone and not str(zone).startswith("ZONE_") else None,
            zone_id=zone if zone and str(zone).startswith("ZONE_") else None,
            access_point_id=state.get("access_point_id") or (location or {}).get("access_point_id"),
            latitude=state.get("latitude"),
            longitude=state.get("longitude"),
        )
        if self.predictions is None:
            self.predictions = ScenarioPredictionService(self.frame)
        output = prediction_tool(self.predictions, request)
        state["location"] = output["scenario"]["location"]
        state["zone"] = output["scenario"]["location"]["zone_id"]
        state["datetime"] = when.isoformat()
        state["last_prediction"] = output
        intent = "EXPLANATION" if parsed.intent == "EXPLANATION" else "PREDICTION"
        sources = [
            "location_tool",
            "historical_demand_tool",
            "scenario_prediction",
            "model_explanation_tool",
        ]
        if intent == "EXPLANATION":
            output["explanation"] = model_explanation_tool(output)
            output["explanation_question"] = True
        return intent, output, sources, output["scenario"]

    def _natural_query(
        self, message: str, parsed: ParsedMessage, state: dict[str, Any]
    ) -> dict[str, Any]:
        text = message.casefold()
        if "access point" in text and any(word in text for word in ("available", "list", "show")):
            catalog = self.locations.catalog
            entries = catalog[["wifi_id", "zone_name"]].to_dict(orient="records")
            return {
                "answer": "Available simulated access points: "
                + "; ".join(f"{row['wifi_id']} ({row['zone_name']})" for row in entries),
                "access_points": entries,
            }
        if "zone" in text and any(word in text for word in ("available", "list", "show")):
            zones = self.frame[["zone_id", "zone_name", "zone_type"]].drop_duplicates()
            entries = zones.sort_values("zone_name").to_dict(orient="records")
            return {
                "answer": "Available synthetic zones: "
                + "; ".join(f"{row['zone_name']} ({row['zone_type']})" for row in entries),
                "zones": entries,
            }
        if "rows" in text or "row" in text:
            result = dashboard_summary_tool(self.frame)
            return {"answer": f"The synthetic dataset contains {result['rows']:,} rows.", **result}
        if "access point" in text or "aps" in text:
            result = dashboard_summary_tool(self.frame)
            return {
                "answer": f"The synthetic dataset contains {result['access_points']} simulated access points.",
                **result,
            }
        if "zone" in text and "how many" in text:
            count = int(self.frame["zone_id"].nunique())
            return {
                "answer": f"The synthetic dataset contains {count} simulated zones.",
                "synthetic_zones": count,
            }
        if "date range" in text:
            result = dashboard_summary_tool(self.frame)
            return {
                "answer": f"The synthetic observations span {result['date_start']} through {result['date_end']}.",
                **result,
            }
        if any(term in text for term in ("previous hours", "last few hours", "what happened")):
            anchor = state.get("datetime")
            if isinstance(anchor, str):
                try:
                    anchor = datetime.fromisoformat(anchor)
                except ValueError:
                    anchor = None
            timestamps = pd.to_datetime(self.frame["timestamp"])
            if anchor is not None:
                end = pd.Timestamp(anchor)
                if end.tzinfo is not None:
                    end = end.tz_convert(LOCAL_ZONE).tz_localize(None)
            else:
                end = timestamps.max() + pd.Timedelta(hours=1)
            recent = self.frame.loc[
                (timestamps < end) & (timestamps >= end - pd.Timedelta(hours=4))
            ]
            summary = recent.groupby("timestamp", as_index=False).agg(
                mean_connections=("connections_next_hour", "mean"),
                high_share=("demand_level", lambda values: values.eq("HIGH").mean()),
                observations=("wifi_id", "size"),
            )
            return {
                "answer": f"The synthetic history has {len(summary)} hourly groups in the four hours before {end}; mean connections and HIGH share are listed by timestamp.",
                "rows": summary.to_dict(orient="records"),
            }
        if "dataset" in text and any(term in text for term in ("location", "area", "zone")):
            location = state.get("location") or {}
            zone_id = location.get("zone_id") or state.get("zone")
            subset = self.frame.loc[self.frame["zone_id"].eq(zone_id)] if zone_id else self.frame
            return {
                "answer": f"The synthetic dataset has {len(subset):,} observations for {location.get('zone_name', 'all locations')}, with mean next-hour connections {subset['connections_next_hour'].mean():.1f} and HIGH share {subset['demand_level'].eq('HIGH').mean():.1%}.",
                "rows": int(len(subset)),
                "mean_connections": float(subset["connections_next_hour"].mean()),
                "high_share": float(subset["demand_level"].eq("HIGH").mean()),
                "location": location,
            }
        if "historically" in text or "historical" in text or "highest" in text:
            return historical_demand_tool(
                self.frame,
                hour=parsed.hour if parsed.hour is not None else 18,
                zone_type=self._zone_type(parsed.zone),
            )
        if "traffic" in text and "demand" in text:
            rows = dataset_query_tool(
                self.queries,
                DatasetQuery(
                    metric="high_share", column="demand_level", group_by=["traffic_level"]
                ),
            )
            return {
                "answer": "HIGH-demand share by simulated traffic category, from historical rows.",
                "rows": rows,
            }
        if "weekend" in text:
            rows = dataset_query_tool(
                self.queries,
                DatasetQuery(
                    metric="mean", column="connections_next_hour", group_by=["is_weekend"]
                ),
            )
            return {
                "answer": "Mean simulated next-hour connections for weekdays and weekends.",
                "rows": rows,
            }
        if "compare" in text:
            rows = dataset_query_tool(
                self.queries,
                DatasetQuery(metric="mean", column="connections_next_hour", group_by=["zone_type"]),
            )
            return {
                "answer": "Historical mean next-hour connections by synthetic zone type.",
                "rows": rows,
            }
        if "traffic" in text:
            rows = dataset_query_tool(
                self.queries,
                DatasetQuery(
                    metric="mean", column="connections_next_hour", group_by=["traffic_level"]
                ),
            )
            return {
                "answer": "Historical mean connections by simulated traffic category.",
                "rows": rows,
            }
        if "month" in text:
            rows = dataset_query_tool(
                self.queries,
                DatasetQuery(metric="mean", column="connections_next_hour", group_by=["month"]),
            )
            return {"answer": "Historical mean next-hour connections by month.", "rows": rows}
        if "weekday" in text or "day of week" in text:
            rows = dataset_query_tool(
                self.queries,
                DatasetQuery(
                    metric="mean", column="connections_next_hour", group_by=["day_of_week"]
                ),
            )
            return {"answer": "Historical mean next-hour connections by day of week.", "rows": rows}
        if "hour" in text or "hourly" in text:
            rows = dataset_query_tool(
                self.queries,
                DatasetQuery(metric="mean", column="connections_next_hour", group_by=["hour"]),
            )
            if parsed.hour is not None:
                rows = [row for row in rows if row["hour"] == parsed.hour]
            return {"answer": "Historical mean next-hour connections by hour.", "rows": rows}
        query = DatasetQuery(metric="count")
        return dataset_query_tool(self.queries, query)

    @staticmethod
    def _dashboard_help(text: str) -> dict[str, Any]:
        sections = {
            "Overview": "Dataset and model status, synthetic-data warning, and generated evaluation summaries.",
            "Scenario prediction": "Choose a synthetic location and time to run classification, regression, calibrated interval, capacity estimate, and feature sensitivity.",
            "Demand Explorer": "Filter historical synthetic observations and compare hourly, daily, weekday, monthly, and zone patterns.",
            "Geographic Analysis": "Inspect simulated AP points and select a marker to resolve a synthetic location.",
            "AI Assistant": "Ask data and scenario questions answered by validated internal tools.",
            "Advanced Prediction": "Provide the full model-input contract directly for technical experiments.",
        }
        requested = next((name for name in sections if name.casefold() in text), None)
        if "chart" in text:
            answer = "Charts summarize historical simulated outcomes; they are descriptive and do not establish causation. "
            answer += "Use Demand Explorer filters to inspect the date, zone, hour, weather, and event subsets."
        elif requested:
            answer = f"{requested}: {sections[requested]}"
        else:
            answer = "The dashboard supports " + "; ".join(
                f"{name}: {description}" for name, description in sections.items()
            )
        return {"answer": answer, "sections": sections, "dataset": "synthetic"}

    @staticmethod
    def _zone_type(zone: str | None) -> str | None:
        if zone is None:
            return None
        normalized = zone.casefold().replace(" ", "_")
        aliases = {"downtown": "DOWNTOWN", "north": "UNIVERSITY", "south": "UNIVERSITY"}
        return aliases.get(normalized, normalized.upper())
