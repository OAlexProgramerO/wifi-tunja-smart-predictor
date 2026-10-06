"""Dashboard-aware deterministic assistant behavior."""

import pytest
from pydantic import ValidationError

from wifi_tunja_smart_predictor.assistant.intents import parse_message
from wifi_tunja_smart_predictor.assistant.schemas import AssistantContext, ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService


@pytest.mark.parametrize(
    ("message", "section"),
    [
        ("Explain Live Scenario", "live_scenario"),
        ("Explícame Live Scenario", "live_scenario"),
        ("EXPLAIN GEOGRAPHIC ANALYSIS!!!", "geographic_analysis"),
        ("que significa esta seccion?", None),
        ("What does F1 mean?", None),
    ],
)
def test_dashboard_intents_and_normalization(message, section):
    parsed = parse_message(message)

    assert parsed.intent in {"DASHBOARD_SECTION_HELP", "METRIC_EXPLANATION"}
    assert parsed.dashboard_section == section


@pytest.mark.parametrize(
    ("section", "expected"),
    [
        ("Overview", "synthetic dataset"),
        ("Live Scenario", "prediction interval"),
        ("Demand Explorer", "historical synthetic observations"),
        ("Geographic Analysis", "do not represent real"),
        ("Network Analysis", "not live network measurements"),
        ("Model Performance", "temporal synthetic-data test holdout"),
        ("AI Assistant", "deterministic intent rules"),
        ("Advanced Prediction", "prediction interval"),
        ("About", "synthetic"),
    ],
)
def test_each_dashboard_section_has_grounded_explanation(mini_frame, section, expected):
    response = AssistantService(mini_frame).handle(
        ChatRequest(message="Explain this.", context=AssistantContext(dashboard_section=section))
    )

    assert response.intent == "DASHBOARD_SECTION_HELP"
    assert expected in response.answer


def test_explain_this_without_dashboard_context_clarifies(mini_frame):
    response = AssistantService(mini_frame).handle(ChatRequest(message="Explain this."))

    assert "Which dashboard section" in response.answer


def test_section_context_is_stored_and_reused(mini_frame):
    assistant = AssistantService(mini_frame)
    first = assistant.handle(
        ChatRequest(
            message="What does this section mean?",
            session_id="dashboard-session",
            context=AssistantContext(dashboard_section="Network Analysis"),
        )
    )
    second = assistant.handle(ChatRequest(message="Explain this", session_id=first.session_id))

    assert "hourly means" in first.answer
    assert "hourly means" in second.answer


def test_unknown_dashboard_section_is_not_assumed(mini_frame):
    with pytest.raises(ValidationError, match="supported dashboard section"):
        AssistantContext(dashboard_section="Forecast Lab")


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What does F1 mean?", "harmonic mean"),
        ("¿Qué significa F1?", "media armónica"),
        ("What does MAE mean?", "mean absolute difference"),
    ],
)
def test_metric_explanations_are_conceptual_and_synthetic(mini_frame, message, expected):
    response = AssistantService(mini_frame).handle(ChatRequest(message=message))

    assert expected in response.answer
    assert "synthetic" in response.answer or "sintéticos" in response.answer


def test_current_scenario_context_is_explained_without_new_prediction(mini_frame):
    scenario = {
        "scenario": {"scenario_time": "2026-10-03T18:00:00"},
        "classification": {"predicted_demand_level": "HIGH", "probability_high": 0.8},
        "regression": {"predicted_connections_next_hour": 42},
        "explanation": {"top_factors": [{"label": "simulated traffic"}]},
    }
    assistant = AssistantService(mini_frame)
    response = assistant.handle(
        ChatRequest(
            message="Why is demand high?",
            context=AssistantContext(dashboard_section="Live Scenario", scenario_result=scenario),
        )
    )

    assert "HIGH" in response.answer
    assert "42 connections" in response.answer
    assert "not proven causes" in response.answer
    assert "synthetic data" in response.answer


def test_v31_conversation_and_v32_demand_intents_remain_available():
    assert parse_message("hola").intent == "GREETING"
    assert parse_message("who are you?").intent == "IDENTITY"
    assert parse_message("what can you do?").intent == "CAPABILITIES"
    demand = parse_message("What is the demand downtown at 6 PM?")
    assert demand.intent == "DEMAND_SCENARIO"
    assert demand.zone == "downtown"
    assert demand.hour == 18
