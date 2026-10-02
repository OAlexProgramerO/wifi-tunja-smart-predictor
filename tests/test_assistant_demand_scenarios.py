"""Context-aware deterministic demand questions use the shared scenario service."""

from unittest.mock import Mock

import pytest

from tests.test_v3_services import DummyClassifier, DummyRegressor
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService


def _assistant(mini_frame):
    prediction_service = ScenarioPredictionService(mini_frame, DummyClassifier(), DummyRegressor())
    prediction_service.classification_metadata = {"project_version": "0.3.2"}
    prediction_service.regression_metadata = {
        "calibration_radius": 20.0,
        "interval_confidence": 0.9,
    }
    tracked = Mock(wraps=prediction_service)
    return AssistantService(mini_frame, prediction_service=tracked), tracked


@pytest.mark.parametrize(
    "message",
    [
        "What is the demand downtown?",
        "Will demand be high downtown?",
        "What is the expected demand downtown at 6 PM?",
        "How many connections are expected downtown at 6 PM?",
        "¿Cuál es la demanda en el centro?",
        "¿Habrá alta demanda en el centro?",
        "¿Cuántas conexiones se esperan en el centro a las 6 PM?",
        "WHAT IS THE DEMAND DOWNTOWN AT 18:00?!",
        "What is the demand downtown 18",
        "What is the demand downtown at 6 de la tarde?",
        "What is the demand downtown at 7 de la noche?",
    ],
)
def test_natural_demand_questions_call_existing_prediction_service(mini_frame, message):
    assistant, prediction_service = _assistant(mini_frame)

    response = assistant.handle(ChatRequest(message=message))

    assert response.intent == "DEMAND_SCENARIO"
    prediction_service.predict.assert_called_once()
    request = prediction_service.predict.call_args.args[0]
    assert request.zone == "downtown"
    if "6 PM" in message or "18:00" in message or "a las 6" in message or message.endswith("18"):
        assert request.datetime.hour == 18
    elif "7 de la noche" in message:
        assert request.datetime.hour == 19
    assert "80" in response.answer
    assert "HIGH" in response.answer
    assert "53.3%" in response.answer
    assert "60–100" in response.answer


def test_location_context_is_reused_for_followup_time(mini_frame):
    assistant, prediction_service = _assistant(mini_frame)
    first = assistant.handle(ChatRequest(message="I'm in downtown.", session_id="conversation"))
    second = assistant.handle(
        ChatRequest(message="What is the demand at 6 PM?", session_id=first.session_id)
    )

    assert "Synthetic Downtown Core synthetic scenario" in first.answer
    assert second.intent == "DEMAND_SCENARIO"
    prediction_service.predict.assert_called_once()
    request = prediction_service.predict.call_args.args[0]
    assert request.zone_id == "ZONE_01"
    assert request.datetime.hour == 18
    assert "80" in second.answer


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("What is the demand at 6 PM?", "Which synthetic zone should I use?"),
        ("What is the demand in Atlantis at 6 PM?", "couldn't match 'atlantis'"),
        ("What is the demand downtown at 25:00?", "valid time"),
    ],
)
def test_demand_query_clarifies_missing_or_invalid_entities(mini_frame, message, expected):
    assistant, prediction_service = _assistant(mini_frame)

    response = assistant.handle(ChatRequest(message=message))

    assert expected.casefold() in response.answer.casefold()
    prediction_service.predict.assert_not_called()
