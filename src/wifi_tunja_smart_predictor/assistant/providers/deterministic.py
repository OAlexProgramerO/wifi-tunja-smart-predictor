"""No-key answer composer; every metric is copied from validated tool output."""

from __future__ import annotations

from datetime import datetime
from typing import Any


class DeterministicProvider:
    """Compose concise responses without an external language model."""

    def compose(self, intent: str, tool_result: dict[str, Any]) -> str:
        """Return a response template from a structured result."""
        if intent == "GREETING":
            message = tool_result.get("message", "")
            if message == "hello":
                return "Hello! How can I help you with WiFi Tunja Smart Predictor?"
            if message == "hola":
                return "¡Hola! ¿Cómo puedo ayudarte?"
            if message == "buenas":
                return "¡Buenas! ¿Cómo puedo ayudarte?"
            return "Hello! How can I help you?"
        if intent == "IDENTITY":
            message = tool_result.get("message", "")
            if message in {"quién eres", "qué eres", "cómo te llamas"}:
                return "Soy el Asistente de WiFi Tunja Smart Predictor. Puedo ayudarte a explorar datos de demanda WiFi, escenarios de predicción e información del modelo."
            return "I'm the WiFi Tunja Smart Predictor Assistant. I can help you explore WiFi demand data, predictions, and model information."
        if intent == "CAPABILITIES":
            message = tool_result.get("message", "")
            if message in {"qué puedes hacer", "en qué puedes ayudarme"}:
                return "Puedo ayudarte a explorar el conjunto de datos sintético de WiFi, analizar escenarios de demanda y entender los resultados del modelo."
            return "I can help you explore the synthetic WiFi dataset, analyze demand scenarios, and understand model results."
        if "answer" in tool_result:
            return str(tool_result["answer"])
        if intent == "LOCATION_INFO":
            location = tool_result["location"]
            if tool_result.get("language") == "es":
                return f"Entendido. Usaré el escenario sintético de {location['zone_name']}."
            return f"Got it. I'll use the {location['zone_name']} synthetic scenario."
        if intent in {"PREDICTION", "DEMAND_SCENARIO"}:
            return _prediction_answer(tool_result)
        if intent == "FOLLOW_UP":
            return _prediction_answer(tool_result)
        if intent == "EXPLANATION":
            factors = tool_result.get("explanation", {}).get("top_factors", [])
            names = ", ".join(factor["label"] for factor in factors[:3])
            return f"The strongest model-sensitivity factors for this synthetic scenario are {names}. They are associated with the prediction, not proven causes."
        if intent == "MODEL_INFO":
            return _model_answer(tool_result)
        if intent == "DASHBOARD_HELP":
            return str(tool_result["answer"])
        if intent == "LIMITATIONS":
            return (
                "This prototype uses simulated observations and historical analogs. It has no live "
                "municipal telemetry, the regression target is synthetic, and feature sensitivity "
                "is not causal."
            )
        if intent == "LOCATION_REMEMBERED":
            return str(tool_result["answer"])
        if intent == "DATA_QUERY":
            if "hour" in tool_result and "rows" in tool_result:
                return _historical_answer(tool_result)
            return _query_answer(tool_result)
        if intent == "HISTORICAL_DEMAND":
            return _historical_answer(tool_result)
        return "I can help with synthetic scenario predictions, data summaries, model details, or dashboard explanations."


def _prediction_answer(result: dict[str, Any]) -> str:
    scenario = result["scenario"]
    classification = result["classification"]
    regression = result["regression"]
    capacity = result["capacity"]
    location_name = scenario["location"]["zone_name"]
    timestamp = scenario["scenario_time"]
    try:
        timestamp = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        pass
    if result.get("response_language") == "es":
        answer = (
            f"En este escenario sintético para {location_name} a las {timestamp}, el modelo estima "
            f"aproximadamente {regression['predicted_connections_next_hour']:.0f} conexiones para la próxima hora. "
            f"La demanda prevista es {classification['predicted_demand_level']}"
        )
        probability = classification.get("probability_high")
        if probability is not None:
            answer += f" (probabilidad estimada de demanda alta: {probability:.0%})"
        utilization = capacity.get("predicted_capacity_utilization_pct")
        if utilization is not None:
            answer += f", con uso estimado de capacidad del {utilization:.1f}%"
        answer += "."
        lower = regression.get("prediction_interval_lower")
        upper = regression.get("prediction_interval_upper")
        confidence = regression.get("interval_confidence")
        if lower is not None and upper is not None and confidence is not None:
            answer += f" El intervalo de predicción del {confidence:.0%} es {lower:.0f}–{upper:.0f} conexiones."
        return answer + " Es una estimación con datos sintéticos, no telemetría en vivo."
    answer = (
        f"In this synthetic scenario for {location_name} at {timestamp}, the model estimates "
        f"approximately {regression['predicted_connections_next_hour']:.0f} connections for the next hour. "
        f"Predicted demand is {classification['predicted_demand_level']}"
    )
    probability = classification.get("probability_high")
    if probability is not None:
        answer += f" ({probability:.0%} estimated probability of HIGH)"
    utilization = capacity.get("predicted_capacity_utilization_pct")
    if utilization is not None:
        answer += f", with estimated capacity utilization of {utilization:.1f}%"
    answer += "."
    lower = regression.get("prediction_interval_lower")
    upper = regression.get("prediction_interval_upper")
    confidence = regression.get("interval_confidence")
    if lower is not None and upper is not None and confidence is not None:
        answer += (
            f" The {confidence:.0%} prediction interval is {lower:.0f}–{upper:.0f} connections."
        )
    answer += " This estimate uses synthetic data, not live network telemetry."
    return answer


def _model_answer(result: dict[str, Any]) -> str:
    metrics = result.get("classification_metrics", {})
    reg_metrics = result.get("regression_metrics", {})
    f1 = metrics.get("f1")
    mae = reg_metrics.get("mae")
    classifier_score = f"{f1:.3f}" if isinstance(f1, (float, int)) else "unavailable"
    regression_score = f"{mae:.2f} connections" if isinstance(mae, (float, int)) else "unavailable"
    return (
        f"The classifier is {result.get('classifier', 'unavailable')} and the numeric model is "
        f"{result.get('regressor', 'unavailable')}. On the synthetic test period, classification "
        f"F1 for HIGH is {classifier_score}; regression MAE is {regression_score}. "
        "These scores do not establish real-world performance."
    )


def _query_answer(result: dict[str, Any]) -> str:
    if "answer" in result:
        return str(result["answer"])
    rows = result.get("rows")
    if rows:
        first = rows[0]
        dimension = next(
            (key for key in first if key not in {"metric", "column", "value", "rows"}), None
        )
        metric_value = first.get("value", first.get("high_share", first.get("mean_connections")))
        if dimension and metric_value is not None:
            return f"According to historical synthetic observations, {dimension}={first[dimension]} has aggregate value {metric_value:.3f}."
    return f"According to the synthetic dataset, the requested aggregate is {result.get('value')} ({result.get('metric')})."


def _historical_answer(result: dict[str, Any]) -> str:
    if not result.get("rows"):
        return "The synthetic dataset has no matching observations for that historical comparison."
    top = result["rows"][0]
    return (
        f"In matching historical synthetic observations at {result['hour']:02d}:00, "
        f"{top['zone_name']} ({top['zone_type']}) had the highest mean next-hour connections "
        f"({top['mean_connections']:.1f}; {top['rows']} rows). This is a historical aggregate, not a forecast."
    )
