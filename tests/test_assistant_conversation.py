"""Basic deterministic conversation intent coverage for the assistant."""

import pytest

from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService


@pytest.mark.parametrize(
    ("message", "intent", "answer_part"),
    [
        ("hi", "GREETING", "How can I help you?"),
        ("hello", "GREETING", "WiFi Tunja Smart Predictor"),
        ("hola", "GREETING", "¡Hola! ¿Cómo puedo ayudarte?"),
        ("who are you?", "IDENTITY", "I'm the WiFi Tunja Smart Predictor Assistant."),
        ("quién eres?", "IDENTITY", "Soy el Asistente de WiFi Tunja Smart Predictor."),
        ("what can you do?", "CAPABILITIES", "I can help you explore the synthetic WiFi dataset"),
        (
            "qué puedes hacer?",
            "CAPABILITIES",
            "Puedo ayudarte a explorar el conjunto de datos sintético",
        ),
    ],
)
def test_basic_conversation_intents(mini_frame, message, intent, answer_part):
    response = AssistantService(mini_frame).handle(ChatRequest(message=message))

    assert response.intent == intent
    assert answer_part in response.answer


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("  HI!!!  ", "GREETING"),
        ("HeY?", "GREETING"),
        ("  HOLA! ", "GREETING"),
        ("¿CÓMO TE LLAMAS?", "IDENTITY"),
        ("EN QUÉ PUEDES AYUDARME?", "CAPABILITIES"),
    ],
)
def test_basic_conversation_matching_ignores_case_and_edge_punctuation(
    mini_frame, message, expected
):
    response = AssistantService(mini_frame).handle(ChatRequest(message=message))

    assert response.intent == expected


@pytest.mark.parametrize(
    ("message", "intent"),
    [
        ("hey", "GREETING"),
        ("buenas", "GREETING"),
        ("what are you?", "IDENTITY"),
        ("who is this?", "IDENTITY"),
        ("qué eres?", "IDENTITY"),
        ("what can you help me with?", "CAPABILITIES"),
        ("en qué puedes ayudarme?", "CAPABILITIES"),
    ],
)
def test_conversation_intent_aliases(mini_frame, message, intent):
    response = AssistantService(mini_frame).handle(ChatRequest(message=message))

    assert response.intent == intent
