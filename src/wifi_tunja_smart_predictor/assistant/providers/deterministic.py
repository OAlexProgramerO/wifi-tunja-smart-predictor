"""No-key answer composer; every metric is copied from validated tool output."""

from __future__ import annotations

from typing import Any


class DeterministicProvider:
    """Compose concise responses without an external language model."""

    def compose(self, intent: str, tool_result: dict[str, Any]) -> str:
        """Return a response template from a structured result."""
        if "answer" in tool_result:
            return str(tool_result["answer"])
        if intent == "LOCATION_INFO":
            location = tool_result["location"]
            return (
                f"{location['zone_name']} is a synthetic zone represented by AP "
                f"{location['access_point_id']} ({location['zone_type']}). "
                "These are simulated locations, not official boundaries."
            )
        if intent == "PREDICTION":
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
    return (
        f"In this synthetic scenario for {scenario['location']['zone_name']} at "
        f"{scenario['scenario_time']}, expected next-hour connections are "
        f"{regression['predicted_connections_next_hour']:.0f} "
        f"(validation-calibrated {regression['interval_confidence']:.0%} interval "
        f"{regression['prediction_interval_lower']:.0f}–{regression['prediction_interval_upper']:.0f}). "
        f"Demand is {classification['predicted_demand_level']} with "
        f"{classification['probability_high']:.0%} probability of HIGH. Estimated capacity use is "
        f"{capacity['predicted_capacity_utilization_pct']:.1f}%. "
        "This is a model-based estimate from historical simulated patterns, not a live observation."
    )


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
