"""Focused checks for the bounded synthetic temporal evaluation service."""

import json

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    REGRESSION_TARGET_COLUMN,
    TARGET_COLUMN,
)
from wifi_tunja_smart_predictor.models.temporal_evaluation import (
    _distribution_change,
    evaluate_temporal_windows,
    render_temporal_markdown,
)
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline


def _models(frame):
    classifier = build_classifier_pipeline(LogisticRegression(max_iter=200), scale_numeric=True)
    regressor = build_classifier_pipeline(LinearRegression(), scale_numeric=False)
    classifier.fit(frame[MODEL_INPUT_COLUMNS], frame[TARGET_COLUMN])
    regressor.fit(frame[MODEL_INPUT_COLUMNS], frame[REGRESSION_TARGET_COLUMN])
    return classifier, regressor


def _fixture_frame():
    frame = make_mini_frame(400)
    frame[TARGET_COLUMN] = [
        "LOW" if (index // 24) % 2 == 0 else "HIGH" for index in range(len(frame))
    ]
    return frame


def test_windows_are_ordered_and_never_train_on_test_timestamps():
    frame = _fixture_frame()
    report = evaluate_temporal_windows(*[frame, *_models(frame)], window_count=3)
    assert len(report["windows"]) == 3
    previous_test_end = None
    for window in report["windows"]:
        assert window["train_end"] < window["test_start"]
        if previous_test_end:
            assert previous_test_end < window["test_start"]
        previous_test_end = window["test_end"]
    assert report["summary"]["evaluated_windows"] >= 1
    assert report["dataset"]["synthetic"] is True
    markdown = render_temporal_markdown(report)
    assert "Feature distribution diagnostics" in markdown
    assert "Dataset SHA-256" in markdown
    assert "synthetic" in markdown


def test_identical_data_has_deterministic_results_except_generation_time():
    frame = _fixture_frame()
    classifier, regressor = _models(frame)
    first = evaluate_temporal_windows(frame, classifier, regressor, window_count=2)
    second = evaluate_temporal_windows(frame, classifier, regressor, window_count=2)
    first.pop("generated_at_utc")
    second.pop("generated_at_utc")
    assert first == second
    json.dumps(first, allow_nan=False)


def test_distribution_methods_cover_numeric_categorical_and_missing_values():
    numeric = _distribution_change(pd.Series([1.0, 2.0, None]), pd.Series([4.0, 5.0, 6.0]))
    category = _distribution_change(pd.Series(["a", "a", "b"]), pd.Series(["b", "b", "b"]))
    insufficient = _distribution_change(pd.Series([None, 1.0]), pd.Series([2.0, None]))
    assert numeric["method"] == "bounded_normalized_mean_difference"
    assert 0 <= numeric["statistic"] <= 1
    assert numeric["reference_missing_proportion"] > 0
    assert category["method"] == "categorical_total_variation_distance"
    assert 0 <= category["statistic"] <= 1
    assert insufficient["status"] == "insufficient_samples"
    assert insufficient["statistic"] is None


def test_single_class_test_windows_are_evaluated_with_undefined_roc_auc():
    frame = _fixture_frame()
    frame[TARGET_COLUMN] = ["HIGH" if index < 40 else "LOW" for index in range(len(frame))]
    report = evaluate_temporal_windows(frame, *_models(frame), window_count=2)
    evaluated = [item for item in report["windows"] if item["status"] == "evaluated"]
    assert evaluated
    assert all(item["metrics"]["classification"]["roc_auc"] is None for item in evaluated)
    assert all(
        item["metrics"]["classification"]["roc_auc_status"]
        == "undefined: both classes are required"
        for item in evaluated
    )


def test_insufficient_rows_are_reported_as_skipped():
    frame = _fixture_frame()
    report = evaluate_temporal_windows(frame, *_models(frame), min_train_rows=500)
    assert report["summary"]["evaluated_windows"] == 0
    assert report["summary"]["skipped_windows"] == 3
    assert report["summary"]["data_quality_limitations"]


def test_insufficient_distinct_timestamps_are_rejected():
    frame = _fixture_frame()
    classifier, regressor = _models(frame)
    with pytest.raises(ValueError, match="distinct timestamps"):
        evaluate_temporal_windows(frame.head(6), classifier, regressor, window_count=5)


@pytest.mark.parametrize("kwargs", [{"window_count": 6}, {"window_count": 1}, {"min_test_rows": 1}])
def test_invalid_settings_are_rejected(kwargs):
    frame = _fixture_frame()
    classifier, regressor = _models(frame)
    with pytest.raises(ValueError):
        evaluate_temporal_windows(frame, classifier, regressor, **kwargs)


def test_invalid_timestamps_are_rejected():
    frame = _fixture_frame()
    frame["timestamp"] = frame["timestamp"].astype(object)
    frame.loc[0, "timestamp"] = "not-a-date"
    classifier, regressor = _models(_fixture_frame())
    with pytest.raises(ValueError, match="Timestamps"):
        evaluate_temporal_windows(frame, classifier, regressor)


def test_missing_timestamp_is_rejected():
    frame = _fixture_frame()
    frame.loc[0, "timestamp"] = pd.NaT
    classifier, regressor = _models(_fixture_frame())
    with pytest.raises(ValueError, match="Timestamps"):
        evaluate_temporal_windows(frame, classifier, regressor)
