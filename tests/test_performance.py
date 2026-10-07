"""Structural checks that guard measured runtime-resource reuse paths."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from wifi_tunja_smart_predictor.assistant.queries import DatasetQueryEngine
from wifi_tunja_smart_predictor.assistant.schemas import ChatRequest
from wifi_tunja_smart_predictor.assistant.service import AssistantService
from wifi_tunja_smart_predictor.scenarios.builder import ScenarioRequest
from wifi_tunja_smart_predictor.scenarios.service import ScenarioPredictionService


class _Classifier:
    classes_ = np.array(["LOW", "HIGH"])

    def predict(self, frame):
        return np.array(["HIGH"] * len(frame))

    def predict_proba(self, frame):
        return np.tile([0.2, 0.8], (len(frame), 1))


class _Regressor:
    def predict(self, frame):
        return np.full(len(frame), 80.0)


def test_assistant_query_engine_reuses_its_read_only_frame(mini_frame):
    assistant = AssistantService(mini_frame)

    assert assistant.queries.frame is assistant.frame
    answer = assistant.handle(ChatRequest(message="How is demand historically downtown?"))
    assert answer.intent == "HISTORICAL_ZONE_ANALYSIS"
    assert (
        answer.structured_result["rows"][0]["mean_connections"]
        == mini_frame["connections_next_hour"].mean()
    )


def test_dataset_query_engine_keeps_copying_by_default(mini_frame):
    engine = DatasetQueryEngine(mini_frame)

    assert engine.frame is not mini_frame


def test_scenario_prediction_selects_analogs_once_and_preserves_outputs(mini_frame):
    import pandas as pd

    scenario_frame = pd.concat([mini_frame] * 3, ignore_index=True)
    service = ScenarioPredictionService(scenario_frame, _Classifier(), _Regressor())
    original_select = service.builder.analogs.select
    calls = []

    def counted_select(location, when):
        calls.append((location, when))
        return original_select(location, when)

    service.builder.analogs.select = counted_select
    result = service.predict(ScenarioRequest(datetime=datetime(2024, 6, 3, 18), zone="downtown"))

    assert len(calls) == 1
    assert result.classification["predicted_demand_level"] == "HIGH"
    assert result.classification["probability_high"] == 0.8
    assert result.regression["predicted_connections_next_hour"] == 80
    assert result.scenario["analog_strategy"] == "zone_type_hour"
