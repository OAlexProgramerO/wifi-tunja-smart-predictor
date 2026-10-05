"""Deterministic historical-demand analysis over stable synthetic fixtures."""

import numpy as np
import pandas as pd
import pytest

from wifi_tunja_smart_predictor.assistant.intents import parse_message
from wifi_tunja_smart_predictor.assistant.schemas import AssistantContext, ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService


def _two_zone_frame(mini_frame):
    downtown = mini_frame.copy()
    downtown["connections_next_hour"] = downtown["hour"] * 3 + 20
    downtown["demand_level"] = np.where(downtown["hour"] >= 12, "HIGH", "LOW")
    north = mini_frame.copy()
    north["zone_id"] = "ZONE_02"
    north["zone_name"] = "Synthetic North Zone"
    north["zone_type"] = "UNIVERSITY"
    north["wifi_id"] = "WIFI_03"
    north["connections_next_hour"] = north["hour"] + 15
    north["demand_level"] = np.where(north["hour"] >= 18, "HIGH", "LOW")
    return pd.concat([downtown, north], ignore_index=True)


@pytest.fixture
def historical_frame(mini_frame):
    return _two_zone_frame(mini_frame)


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("How is demand historically downtown?", "HISTORICAL_ZONE_ANALYSIS"),
        ("When is demand usually highest?", "HISTORICAL_PEAK"),
        ("Which zone has higher demand?", "HISTORICAL_COMPARISON"),
        ("Compare center and south.", "HISTORICAL_COMPARISON"),
        ("What day of the week has the highest demand?", "HISTORICAL_TIME_ANALYSIS"),
        ("Is demand higher in the morning or evening?", "HISTORICAL_TIME_ANALYSIS"),
        ("¿Cómo ha cambiado la demanda en el centro?", "HISTORICAL_ZONE_ANALYSIS"),
        ("¿Qué zona tuvo más demanda HIGH?", "HISTORICAL_COMPARISON"),
        ("¿Es mayor en la mañana o en la noche?", "HISTORICAL_TIME_ANALYSIS"),
    ],
)
def test_historical_intent_detection_and_zone_aliases(message, intent):
    parsed = parse_message(message)

    assert parsed.intent == intent
    if "center" in message or "centro" in message:
        assert "downtown" in parsed.comparison_zones or parsed.zone == "downtown"


@pytest.mark.parametrize(
    "message",
    ["¿Qué zona tiene mayor demanda?", "¿Qué día de la semana tiene más demanda?"],
)
def test_spanish_ranking_questions_do_not_treat_question_words_as_zones(message):
    parsed = parse_message(message)

    assert parsed.zone is None
    assert parsed.intent in {"HISTORICAL_COMPARISON", "HISTORICAL_TIME_ANALYSIS"}


def test_historical_zone_analysis_reports_separate_metrics(historical_frame):
    response = AssistantService(historical_frame).handle(
        ChatRequest(message="What was the average demand downtown?")
    )

    assert response.intent == "HISTORICAL_ZONE_ANALYSIS"
    assert "average next-hour connections" in response.answer.casefold()
    assert "HIGH-demand rate" in response.answer
    assert "LOW" in response.answer
    assert "simulated" in response.answer or "synthetic" in response.answer
    assert response.structured_result["rows"][0]["mean_connections"] == pytest.approx(
        historical_frame.loc[historical_frame.zone_id.eq("ZONE_01"), "connections_next_hour"].mean()
    )


def test_zone_comparison_uses_zone_aliases_and_both_measures(historical_frame):
    response = AssistantService(historical_frame).handle(
        ChatRequest(message="Compare center vs north.")
    )

    assert response.intent == "HISTORICAL_COMPARISON"
    assert "Synthetic Downtown Core" in response.answer
    assert "Synthetic North Zone" in response.answer
    assert response.answer.count("Historically") == 3
    assert "connections" in response.answer
    assert "HIGH-demand rate" in response.answer
    assert "LOW-demand rate" in response.answer


def test_spanish_historical_comparison_is_deterministic_and_disclaimed(historical_frame):
    response = AssistantService(historical_frame).handle(
        ChatRequest(message="Compara centro y norte.")
    )

    assert response.intent == "HISTORICAL_COMPARISON"
    assert "registros sintéticos" in response.answer
    assert "Synthetic Downtown Core" in response.answer
    assert "Synthetic North Zone" in response.answer
    assert "simulado" in response.answer


def test_unknown_and_same_zone_comparisons_are_clarified(historical_frame):
    assistant = AssistantService(historical_frame)
    unknown = assistant.handle(ChatRequest(message="Compare downtown and Atlantis."))
    same = assistant.handle(ChatRequest(message="Compare center and downtown."))

    assert "couldn't match 'atlantis'" in unknown.answer
    assert "same synthetic zone" in same.answer


def test_peak_hour_and_weekday_use_historical_records(mini_frame):
    hour_frame = mini_frame.copy()
    hour_frame["connections_next_hour"] = hour_frame["hour"]
    day_frame = mini_frame.copy()
    day_frame["connections_next_hour"] = np.where(day_frame["day_of_week"].eq(6), 200, 10)

    peak_hour = AssistantService(hour_frame).handle(ChatRequest(message="When is demand highest?"))
    peak_day = AssistantService(day_frame).handle(
        ChatRequest(message="What day of the week has the highest demand?")
    )

    assert "23:00" in peak_hour.answer
    assert "SUNDAY" in peak_day.answer
    assert "synthetic" in peak_day.answer


def test_morning_evening_comparison_uses_time_period_aggregates(historical_frame):
    response = AssistantService(historical_frame).handle(
        ChatRequest(message="Is demand usually higher in the morning or evening?")
    )

    assert response.intent == "HISTORICAL_TIME_ANALYSIS"
    assert "MORNING" in response.answer
    assert "EVENING" in response.answer


def test_remembered_zone_and_dashboard_context_are_reused(historical_frame):
    assistant = AssistantService(historical_frame)
    first = assistant.handle(ChatRequest(message="I'm in downtown.", session_id="history-context"))
    response = assistant.handle(
        ChatRequest(
            message="How is demand historically?",
            session_id=first.session_id,
            context=AssistantContext(dashboard_section="Demand Explorer"),
        )
    )

    assert response.intent == "HISTORICAL_ZONE_ANALYSIS"
    assert len(response.structured_result["rows"]) == 1
    assert response.structured_result["rows"][0]["zone_id"] == "ZONE_01"
    assert assistant.sessions.get(first.session_id)["dashboard_section"] == "demand_explorer"


def test_historical_change_compares_months_only_when_available(mini_frame):
    first = mini_frame.copy()
    second = mini_frame.copy()
    for frame, month, amount in ((first, 1, 0), (second, 2, 25)):
        timestamps = pd.date_range(f"2024-{month:02d}-01", periods=len(frame), freq="h")
        frame["timestamp"] = timestamps
        frame["year"] = timestamps.year
        frame["month"] = timestamps.month
        frame["connections_next_hour"] = frame["connections_next_hour"] + amount
    frame = pd.concat([first, second], ignore_index=True)

    response = AssistantService(frame).handle(
        ChatRequest(message="How has demand changed downtown?")
    )

    assert "increased by 25.0" in response.answer
    assert "synthetic" in response.answer


def test_historical_change_explains_insufficient_monthly_coverage(mini_frame):
    response = AssistantService(mini_frame).handle(
        ChatRequest(message="How has demand changed downtown?")
    )

    assert "not enough distinct monthly periods" in response.answer


def test_unknown_zone_and_invalid_time_do_not_crash(historical_frame):
    assistant = AssistantService(historical_frame)
    unknown = assistant.handle(ChatRequest(message="What was average demand in Atlantis?"))
    invalid = assistant.handle(ChatRequest(message="What was demand historically at 25:00?"))

    assert "couldn't match 'atlantis'" in unknown.answer
    assert "18:00" in invalid.answer


def test_v31_v32_v33_intents_remain_compatible():
    assert parse_message("hola").intent == "GREETING"
    assert parse_message("who are you?").intent == "IDENTITY"
    assert parse_message("what can you do?").intent == "CAPABILITIES"
    demand = parse_message("What is the demand downtown at 6 PM?")
    assert demand.intent == "DEMAND_SCENARIO"
    assert (demand.zone, demand.hour) == ("downtown", 18)
    dashboard = parse_message("Explain this.")
    assert dashboard.intent == "DASHBOARD_SECTION_HELP"
