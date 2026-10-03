"""Deterministic natural-language dispatcher backed only by project tools."""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from threading import RLock
from typing import Any
from uuid import uuid4

import pandas as pd

from wifi_tunja_smart_predictor.assistant.intents import (
    LOCAL_ZONE,
    ParsedMessage,
    normalize_message,
    parse_message,
)
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

_SECTION_ALIASES = {
    "overview": "overview",
    "dashboard overview": "overview",
    "live scenario": "live_scenario",
    "scenario prediction": "live_scenario",
    "scenario": "live_scenario",
    "demand explorer": "demand_explorer",
    "demand analysis": "demand_explorer",
    "geographic analysis": "geographic_analysis",
    "geographic": "geographic_analysis",
    "geography": "geographic_analysis",
    "network analysis": "network_analysis",
    "network": "network_analysis",
    "model performance": "model_performance",
    "performance": "model_performance",
    "ai assistant": "ai_assistant",
    "assistant": "ai_assistant",
    "advanced prediction": "advanced_prediction",
    "advanced": "advanced_prediction",
    "about": "about",
}


def _canonical_section(value: str) -> str | None:
    return _SECTION_ALIASES.get(value.casefold().strip())


def _is_scenario_question(text: str) -> bool:
    return any(
        term in text
        for term in (
            "prediction",
            "demand high",
            "capacity",
            "interval",
            "why",
            "probability",
            "conexiones",
            "prediccion",
            "capacidad",
            "intervalo",
            "demanda es alta",
        )
    )


def _section_explanation(section: str, language: str = "en") -> str:
    answers = {
        "overview": "Overview summarizes the synthetic dataset and model readiness. The prototype classifies next-hour demand as LOW/HIGH and estimates next-hour connections; it does not use live Tunja telemetry.",
        "live_scenario": "Live Scenario combines a selected synthetic location and time with the existing classifier and regressor. It displays LOW/HIGH probability, expected next-hour connections, a validation-calibrated prediction interval, a capacity-use proxy, and local sensitivity factors. These are synthetic estimates, not live telemetry or causal findings.",
        "demand_explorer": "Demand Explorer filters historical synthetic observations by date, zone type, access point, simulated weather, and nearby-event flag. Its distribution, hourly, weekday, zone, daily, monthly, and weekend charts describe those filtered records; they do not establish causation.",
        "geographic_analysis": "Geographic Analysis displays simulated access-point locations and lets you select a point for the scenario builder. Coordinates and points do not represent real public WiFi infrastructure in Tunja.",
        "network_analysis": "Network Analysis plots hourly means of the prototype's synthetic network indicators, such as connected devices, sessions, channel utilization, latency, bandwidth, packet loss, signal, and capacity proxy. They are not live network measurements.",
        "model_performance": "Model Performance reports classification and regression metrics from a temporal synthetic-data test holdout. Accuracy, precision, recall, F1, and ROC-AUC describe classification; MAE, RMSE, and R² describe connection-count regression. These scores do not establish real-world performance.",
        "ai_assistant": "AI Assistant uses deterministic intent rules and fixed project tools for dataset, demand, location, model, and dashboard questions. It can reuse session location, time, dashboard section, and scenario context; no LLM credential is required.",
        "advanced_prediction": "Advanced Prediction accepts the model's explicit feature inputs and returns the existing regression estimate and calibrated prediction interval alongside classification output. The interval summarizes uncertainty for this synthetic prototype; real-world coverage is not guaranteed.",
        "about": "About describes WiFi Tunja Smart Predictor, a Python and Streamlit decision-support prototype with deterministic assistant/API services and scikit-learn models. Its observations, locations, and outcomes are synthetic, not municipal WiFi telemetry.",
    }
    if language == "es":
        spanish = {
            "overview": "Overview resume el conjunto de datos sintético y el estado de los modelos. El prototipo clasifica la demanda de la próxima hora como LOW/HIGH y estima conexiones; no usa telemetría WiFi en vivo de Tunja.",
            "live_scenario": "Live Scenario combina una ubicación y hora sintéticas con los modelos existentes. Muestra probabilidad LOW/HIGH, conexiones estimadas, intervalo calibrado con validación, un indicador aproximado de capacidad y factores de sensibilidad local. Son estimaciones sintéticas, no telemetría en vivo ni causas comprobadas.",
            "demand_explorer": "Demand Explorer filtra observaciones históricas sintéticas por fecha, zona, punto de acceso, clima simulado y eventos cercanos. Sus gráficos describen esos registros y no demuestran causalidad.",
            "geographic_analysis": "Geographic Analysis muestra ubicaciones simuladas de puntos de acceso para seleccionar en el generador de escenarios. Las coordenadas no representan infraestructura WiFi pública real en Tunja.",
            "network_analysis": "Network Analysis grafica promedios horarios de indicadores de red sintéticos, como dispositivos conectados, sesiones, uso de canal, latencia, ancho de banda, pérdida de paquetes, señal y capacidad. No son mediciones en vivo.",
            "model_performance": "Model Performance presenta métricas de clasificación y regresión en una partición temporal de prueba sintética. Accuracy, precision, recall, F1 y ROC-AUC describen clasificación; MAE, RMSE y R² describen regresión. No demuestran rendimiento real.",
            "ai_assistant": "AI Assistant usa reglas deterministas y herramientas internas para preguntas sobre datos, demanda, ubicaciones, modelos y el panel. Reutiliza contexto de sesión y no requiere credenciales de un LLM.",
            "advanced_prediction": "Advanced Prediction recibe las entradas explícitas del modelo y devuelve la estimación de regresión, su intervalo calibrado y la clasificación. La cobertura real no está garantizada.",
            "about": "About describe WiFi Tunja Smart Predictor, un prototipo con Python, Streamlit, servicios deterministas y modelos scikit-learn. Sus observaciones, ubicaciones y resultados son sintéticos, no telemetría municipal.",
        }
        return spanish[section]
    return answers[section]


def _metric_explanation(metric: str, language: str = "en") -> str:
    definitions = {
        "ACCURACY": "Accuracy is the share of classification labels predicted correctly across both LOW and HIGH classes.",
        "PRECISION": "Precision for HIGH is the share of predicted-HIGH cases that are HIGH in the evaluation labels.",
        "RECALL": "Recall for HIGH is the share of evaluation HIGH cases the classifier identifies as HIGH.",
        "F1": "F1 for HIGH is the harmonic mean of HIGH-class precision and recall, balancing false alarms and missed HIGH cases.",
        "ROC AUC": "ROC-AUC measures how well the classifier ranks HIGH cases above LOW cases across probability thresholds.",
        "MAE": "MAE is the mean absolute difference between estimated and observed next-hour connection counts; lower is better.",
        "RMSE": "RMSE is the square root of mean squared connection-count error, so larger errors weigh more; lower is better.",
        "R2": "R² compares regression error with a mean-baseline on the evaluated connection counts; higher is generally better, and it can be negative.",
    }
    if language == "es":
        spanish = {
            "ACCURACY": "Accuracy es la proporción de etiquetas LOW/HIGH clasificadas correctamente.",
            "PRECISION": "Precision para HIGH es la proporción de predicciones HIGH que realmente son HIGH.",
            "RECALL": "Recall para HIGH es la proporción de casos HIGH que el clasificador identifica.",
            "F1": "F1 para HIGH es la media armónica de precision y recall de esa clase.",
            "ROC AUC": "ROC-AUC mide qué tan bien el clasificador ordena casos HIGH sobre LOW entre distintos umbrales.",
            "MAE": "MAE es el error absoluto medio entre conexiones estimadas y observadas; menor es mejor.",
            "RMSE": "RMSE es la raíz del error cuadrático medio y penaliza más los errores grandes.",
            "R2": "R² compara el error de regresión con una referencia basada en la media; puede ser negativo.",
        }
        return (
            spanish.get(metric, "Esta métrica se reporta en el periodo sintético de prueba.")
            + " La evaluación usa datos sintéticos y no demuestra rendimiento real."
        )
    return (
        definitions.get(metric, "This metric is reported on the synthetic test period.")
        + " The project's evaluation uses synthetic data, so it does not establish real-world performance."
    )


def _scenario_explanation(result: Any, language: str = "en") -> str:
    if not isinstance(result, dict) or not {"scenario", "classification", "regression"}.issubset(
        result
    ):
        return (
            "No hay un resultado completo del escenario en esta sesión. Ejecuta un escenario primero; las estimaciones usan datos sintéticos."
            if language == "es"
            else "A complete scenario result is not available in this session. Run a scenario first; estimates use synthetic data."
        )
    classification = result["classification"]
    regression = result["regression"]
    demand = classification.get("predicted_demand_level", "unknown")
    probability = classification.get("probability_high")
    estimate = regression.get("predicted_connections_next_hour")
    lower = regression.get("prediction_interval_lower")
    upper = regression.get("prediction_interval_upper")
    confidence = regression.get("interval_confidence")
    capacity = result.get("capacity", {}).get("predicted_capacity_utilization_pct")
    factors = result.get("explanation", {}).get("top_factors", [])
    factor_text = ", ".join(str(item.get("label")) for item in factors[:3] if item.get("label"))
    if language == "es":
        answer = f"El escenario sintético actual estima demanda {demand}"
        if probability is not None:
            answer += f" con una probabilidad estimada de HIGH del {probability:.0%}"
        if estimate is not None:
            answer += f" y aproximadamente {estimate:.0f} conexiones para la próxima hora"
        answer += "."
        if capacity is not None:
            answer += f" El uso aproximado de capacidad es {capacity:.1f}%."
        if lower is not None and upper is not None and confidence is not None:
            answer += f" El intervalo de predicción nominal del {confidence:.0%} es {lower:.0f}–{upper:.0f} conexiones."
        if factor_text:
            answer += f" {factor_text} son factores de sensibilidad local asociados con la salida, no causas comprobadas."
        answer += " Este resultado usa datos sintéticos, no telemetría en vivo."
        return answer
    answer = f"The current synthetic scenario predicts {demand} demand"
    if probability is not None:
        answer += f" with {probability:.0%} estimated probability of HIGH"
    if estimate is not None:
        answer += f" and about {estimate:.0f} connections next hour"
    answer += "."
    if capacity is not None:
        answer += f" Estimated capacity use is {capacity:.1f}%."
    if lower is not None and upper is not None and confidence is not None:
        answer += f" The nominal {confidence:.0%} prediction interval is {lower:.0f}–{upper:.0f} connections."
    if factor_text:
        answer += f" {factor_text} are local model-sensitivity factors associated with this output, not proven causes."
    answer += " It is an estimate from synthetic data, not live telemetry."
    return answer


def _metric_name(text: str) -> str | None:
    from wifi_tunja_smart_predictor.assistant.intents import _comparison_text

    normalized = _comparison_text(text).replace("-", " ")
    for metric in ("accuracy", "precision", "recall", "roc auc", "f1", "mae", "rmse", "r2"):
        if metric in normalized:
            return metric.upper()
    return None


def _response_language(message: str) -> str:
    from wifi_tunja_smart_predictor.assistant.intents import _comparison_text

    text = _comparison_text(message.casefold())
    return (
        "es"
        if any(
            term in text
            for term in (
                "que ",
                "explica",
                "explicame",
                "seccion",
                "grafico",
                "demanda",
                "por que",
                "prediccion",
            )
        )
        else "en"
    )


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
        if parsed.zone and parsed.intent != "DEMAND_SCENARIO":
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
            if key == "dashboard_section":
                normalized = normalize_message(value)
                section = _canonical_section(normalized)
                if section:
                    state[key] = section
                else:
                    state.pop(key, None)
            else:
                state[key] = value.isoformat() if isinstance(value, datetime) else value

    def _dispatch(
        self, parsed: ParsedMessage, message: str, state: dict[str, Any]
    ) -> tuple[str, dict[str, Any], list[str], dict[str, Any] | None]:
        text = message.casefold()
        if parsed.intent in {"GREETING", "IDENTITY", "CAPABILITIES"}:
            return (
                parsed.intent,
                {"message": normalize_message(message)},
                ["deterministic_conversation_templates"],
                None,
            )
        if parsed.intent in {"DASHBOARD_SECTION_HELP", "METRIC_EXPLANATION"}:
            section = parsed.dashboard_section or state.get("dashboard_section")
            language = _response_language(message)
            metric = _metric_name(normalize_message(message))
            if metric:
                answer = _metric_explanation(metric, language)
            elif _is_scenario_question(normalize_message(message)):
                scenario = state.get("scenario_result") or state.get("last_prediction")
                answer = (
                    _scenario_explanation(scenario, language)
                    if scenario
                    else (
                        "Aún no hay un resultado de escenario en esta sesión. Ejecuta un escenario primero; las estimaciones usan datos sintéticos."
                        if language == "es"
                        else "There is no current scenario result in this session yet. Run a scenario first; all estimates use synthetic data."
                    )
                )
            elif section == "live_scenario" and state.get("scenario_result"):
                answer = _scenario_explanation(state["scenario_result"], language)
            elif section:
                answer = _section_explanation(section, language)
                state["dashboard_section"] = section
            else:
                answer = (
                    "¿Qué sección del panel quieres que explique?"
                    if language == "es"
                    else "Which dashboard section would you like me to explain?"
                )
            return (
                "DASHBOARD_SECTION_HELP",
                {"answer": answer},
                ["dashboard_context", "project_documentation"],
                None,
            )
        if parsed.intent in {"DASHBOARD_SECTION_HELP", "METRIC_EXPLANATION"}:
            section = parsed.dashboard_section or state.get("dashboard_section")
            metric = _metric_name(normalize_message(message))
            if metric:
                answer = _metric_explanation(metric)
            elif _is_scenario_question(normalize_message(message)):
                scenario = state.get("scenario_result") or state.get("last_prediction")
                answer = (
                    _scenario_explanation(scenario)
                    if scenario
                    else (
                        "There is no current scenario result in this session yet. Run a scenario first; all estimates use synthetic data."
                    )
                )
            elif section == "live_scenario" and state.get("scenario_result"):
                answer = _scenario_explanation(state["scenario_result"])
            elif section:
                answer = _section_explanation(section)
                state["dashboard_section"] = section
            else:
                answer = "Which dashboard section would you like me to explain?"
            return (
                "DASHBOARD_SECTION_HELP",
                {"answer": answer},
                ["dashboard_context", "project_documentation"],
                None,
            )
        if parsed.intent == "LOCATION_INFO" and state.get("zone"):
            resolved = location_tool(self.locations, state["zone"])
            state["location"] = resolved
            state["zone"] = resolved["zone_id"]
            state["access_point_id"] = resolved["access_point_id"]
            is_spanish = any(
                term in text for term in ("estoy en", "centro", "norte", "sur", "universidad")
            )
            return (
                "LOCATION_INFO",
                {"location": resolved, "language": "es" if is_spanish else "en"},
                ["location_tool", "synthetic_dataset"],
                None,
            )
        if parsed.intent in {"PREDICTION", "FOLLOW_UP", "EXPLANATION"}:
            return self._predict(parsed, text, state)
        if parsed.intent == "DEMAND_SCENARIO":
            if parsed.invalid_time:
                return (
                    parsed.intent,
                    {"answer": "Please provide a valid time, such as 18:00 or 6 PM."},
                    ["scenario_clarification"],
                    None,
                )
            if parsed.zone is None and not state.get("location"):
                return (
                    parsed.intent,
                    {
                        "answer": "Which synthetic zone should I use? For example, downtown, north, south, or university."
                    },
                    ["scenario_clarification"],
                    None,
                )
            if parsed.zone is not None:
                try:
                    self.locations.resolve(zone=parsed.zone)
                except LocationResolutionError:
                    return (
                        parsed.intent,
                        {
                            "answer": f"I couldn't match '{parsed.zone}' to a synthetic zone. Choose a listed project zone such as downtown, north, south, or university."
                        },
                        ["scenario_clarification"],
                        None,
                    )
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
        if any(term in text for term in ("demanda", "conexiones", "cuál", "habrá")):
            output["response_language"] = "es"
        state["location"] = output["scenario"]["location"]
        state["zone"] = output["scenario"]["location"]["zone_id"]
        state["datetime"] = when.isoformat()
        state["last_prediction"] = output
        state["scenario_result"] = output
        state["scenario_result"] = output
        intent = (
            parsed.intent if parsed.intent in {"DEMAND_SCENARIO", "EXPLANATION"} else "PREDICTION"
        )
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
