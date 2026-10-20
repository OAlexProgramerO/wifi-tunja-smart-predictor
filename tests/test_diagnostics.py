"""Tests for report metrics, chronological fitting, persistence, and staleness checks."""

import json

import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

from tests.conftest import make_mini_frame
from wifi_tunja_smart_predictor.config import (
    MODEL_INPUT_COLUMNS,
    REGRESSION_TARGET_COLUMN,
    TARGET_COLUMN,
)
from wifi_tunja_smart_predictor.models.diagnostics import (
    build_diagnostics,
    load_valid_diagnostics,
    write_diagnostics,
)
from wifi_tunja_smart_predictor.preprocessing.pipeline import build_classifier_pipeline


def _built_report(frame, tmp_path, monkeypatch):
    import wifi_tunja_smart_predictor.models.diagnostics as diagnostics

    dataset = tmp_path / "dataset.csv"
    frame.to_csv(dataset, index=False)
    classifier_path = tmp_path / "classifier.joblib"
    regressor_path = tmp_path / "regressor.joblib"
    classifier_path.write_bytes(b"classifier artifact")
    regressor_path.write_bytes(b"regressor artifact")
    monkeypatch.setattr(diagnostics, "MODEL_PATH", classifier_path)
    monkeypatch.setattr(diagnostics, "REGRESSION_MODEL_PATH", regressor_path)
    monkeypatch.setattr(diagnostics, "PROCESSED_DATASET_PATH", dataset)
    monkeypatch.setattr(diagnostics, "RAW_DATASET_PATH", tmp_path / "not-present.csv")
    classifier = build_classifier_pipeline(LogisticRegression(max_iter=200), scale_numeric=True)
    regressor = build_classifier_pipeline(LinearRegression(), scale_numeric=False)
    classifier.fit(frame[MODEL_INPUT_COLUMNS], frame[TARGET_COLUMN])
    regressor.fit(frame[MODEL_INPUT_COLUMNS], frame[REGRESSION_TARGET_COLUMN])
    metadata = {
        "project_version": diagnostics.PROJECT_VERSION,
        "selected_model": "test-classifier",
        "test_period": {
            "start": str(frame.timestamp.min()),
            "end": str(frame.timestamp.max()),
        },
    }
    report = build_diagnostics(
        frame, classifier, regressor, metadata, {"selected_model": "test-regressor"}, dataset
    )
    report_path = tmp_path / "diagnostics.json"
    write_diagnostics(report, report_path)
    monkeypatch.setattr(diagnostics, "MODEL_DIAGNOSTICS_PATH", report_path)
    return diagnostics, dataset, report_path, report


def test_report_has_metric_support_and_expanding_windows(mini_frame, tmp_path, monkeypatch):
    frame = make_mini_frame(300)
    _diagnostics, _dataset, _path, report = _built_report(frame, tmp_path, monkeypatch)

    assert report["schema_version"] == "1.0"
    assert "missing_value_count" in report["data_quality"]
    assert report["test"]["classification"]["class_metrics"]["LOW"]["support"] > 0
    assert report["test"]["regression"]["absolute_error_quantiles"].keys() == {"0.5", "0.9", "0.95"}
    assert len(report["rolling_windows"]) == 3
    for window in report["rolling_windows"]:
        if window["status"] == "passed":
            assert window["train_end"] < window["test_start"]


def test_report_round_trip_and_rejects_changed_dataset(mini_frame, tmp_path, monkeypatch):
    diagnostics, dataset, report_path, report = _built_report(
        make_mini_frame(300), tmp_path, monkeypatch
    )
    assert load_valid_diagnostics(report_path) == report

    with dataset.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="stale"):
        diagnostics.load_valid_diagnostics(report_path)


def test_atomic_writer_leaves_valid_json(mini_frame, tmp_path, monkeypatch):
    _diagnostics, _dataset, path, report = _built_report(
        make_mini_frame(300), tmp_path, monkeypatch
    )
    assert json.loads(path.read_text(encoding="utf-8")) == report
    assert not list(tmp_path.glob("*.tmp"))
