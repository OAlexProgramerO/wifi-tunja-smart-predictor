from __future__ import annotations

import hashlib
import json

from fastapi.testclient import TestClient

import api.main
import api.routes.model_runs as model_run_routes
import wifi_tunja_smart_predictor.models.run_comparison as comparison


def _metadata(selected: str = "random_forest") -> dict:
    return {
        "project_version": "0.7.9",
        "selected_model": selected,
        "training_run_id": "run-2026-01",
        "training_timestamp_utc": "2026-01-03T12:00:00+00:00",
        "evaluation_timestamp_utc": "2026-01-03T12:05:00+00:00",
        "dataset_version": "synthetic-seed-test",
        "train_period": {"start": "2025-01-01", "end": "2025-06-30", "rows": 10},
        "validation_period": {"start": "2025-07-01", "end": "2025-08-31", "rows": 4},
        "test_period": {"start": "2025-09-01", "end": "2025-09-30", "rows": 4},
    }


def test_compatible_run_records_compare_metrics_with_deltas():
    windows = {"test": {"start": "2025-09-01", "end": "2025-09-30"}}
    report = comparison.compare_run_records(
        [
            {
                "model": "selected",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.8, "mae": 3},
            },
            {
                "model": "candidate",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.9, "mae": 2},
            },
        ],
        selected_model="selected",
    )
    assert report["status"] == "VALID"
    assert report["runs"][1]["metric_deltas_vs_selected"] == {"f1": 0.1, "mae": -1}


def test_different_dataset_fingerprints_are_incompatible_without_deltas():
    windows = {"test": {"start": "a", "end": "b"}}
    report = comparison.compare_run_records(
        [
            {
                "model": "selected",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.8},
            },
            {
                "model": "candidate",
                "dataset_fingerprint": "b" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.9},
            },
        ],
        selected_model="selected",
    )
    assert report["status"] == "INCOMPATIBLE"
    assert all(run["metric_deltas_vs_selected"] is None for run in report["runs"])


def test_different_evaluation_windows_are_incompatible_without_deltas():
    report = comparison.compare_run_records(
        [
            {
                "model": "selected",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": {"test": "Q1"},
                "metrics": {"f1": 0.8},
            },
            {
                "model": "candidate",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": {"test": "Q2"},
                "metrics": {"f1": 0.9},
            },
        ],
        selected_model="selected",
    )
    assert report["status"] == "INCOMPATIBLE"
    assert all(run["metric_deltas_vs_selected"] is None for run in report["runs"])


def test_different_model_roles_are_not_compared():
    windows = {"test": {"start": "a", "end": "b"}}
    report = comparison.compare_run_records(
        [
            {
                "role": "classifier",
                "model": "selected",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.8},
            },
            {
                "role": "regressor",
                "model": "candidate",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.9},
            },
        ],
        selected_model="selected",
    )
    assert report["status"] == "INCOMPATIBLE"
    assert all(run["metric_deltas_vs_selected"] is None for run in report["runs"])


def test_missing_metadata_returns_partial_comparison(tmp_path, monkeypatch):
    csv_path = tmp_path / "classifier.csv"
    csv_path.write_text("model,accuracy\none,0.8\n", encoding="utf-8")
    monkeypatch.setattr(comparison, "CLASSIFICATION_METADATA_PATH", tmp_path / "missing.json")
    monkeypatch.setattr(comparison, "MODEL_COMPARISON_PATH", csv_path)
    monkeypatch.setattr(
        comparison, "REGRESSION_METADATA_PATH", tmp_path / "missing-regression.json"
    )
    monkeypatch.setattr(
        comparison, "REGRESSION_COMPARISON_PATH", tmp_path / "missing-regression.csv"
    )
    monkeypatch.setattr(comparison, "MODEL_DIAGNOSTICS_PATH", tmp_path / "missing-diagnostics.json")
    report = comparison.build_model_run_comparison()
    assert report["tasks"][0]["status"] == "PARTIAL"
    assert report["tasks"][0]["runs"][0]["dataset_fingerprint"] is None


def test_malformed_metadata_is_reported_as_partial_without_leaking_raw_content(
    tmp_path, monkeypatch
):
    classifier_csv = tmp_path / "classifier.csv"
    classifier_csv.write_text("model,accuracy\none,0.8\n", encoding="utf-8")
    metadata = tmp_path / "classification.json"
    metadata.write_text('{"model_path":"C:/private/model.joblib",', encoding="utf-8")
    monkeypatch.setattr(comparison, "CLASSIFICATION_METADATA_PATH", metadata)
    monkeypatch.setattr(comparison, "MODEL_COMPARISON_PATH", classifier_csv)
    monkeypatch.setattr(
        comparison, "REGRESSION_METADATA_PATH", tmp_path / "missing-regression.json"
    )
    monkeypatch.setattr(
        comparison, "REGRESSION_COMPARISON_PATH", tmp_path / "missing-regression.csv"
    )
    monkeypatch.setattr(comparison, "MODEL_DIAGNOSTICS_PATH", tmp_path / "missing-diagnostics.json")
    response = comparison.build_model_run_comparison()
    assert response["tasks"][0]["status"] == "PARTIAL"
    assert "C:/private" not in json.dumps(response)


def test_partial_metric_sets_keep_shared_deltas_and_mark_partial():
    windows = {"test": {"start": "a", "end": "b"}}
    report = comparison.compare_run_records(
        [
            {
                "model": "selected",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.8},
            },
            {
                "model": "candidate",
                "dataset_fingerprint": "a" * 64,
                "evaluation_windows": windows,
                "metrics": {"f1": 0.9, "roc_auc": 0.95},
            },
        ],
        selected_model="selected",
    )
    assert report["status"] == "PARTIAL"
    assert report["runs"][1]["metric_deltas_vs_selected"] == {"f1": 0.1}


def test_saved_table_comparison_is_read_only_and_uses_task_local_metrics(tmp_path, monkeypatch):
    classifier_csv = tmp_path / "classifier.csv"
    regressor_csv = tmp_path / "regressor.csv"
    selected_model = tmp_path / "classifier.joblib"
    selected_regressor = tmp_path / "regressor.joblib"
    dataset = tmp_path / "dataset.csv"
    selected_model.write_bytes(b"persisted classifier bytes")
    selected_regressor.write_bytes(b"persisted regressor bytes")
    dataset.write_bytes(b"synthetic dataset bytes")
    classifier_csv.write_text(
        "model,accuracy,f1\nrandom_forest,0.9,0.8\nother,0.8,0.7\n", encoding="utf-8"
    )
    regressor_csv.write_text(
        "model,mae,rmse\nrandom_forest_regressor,2,3\nother_regressor,3,4\n", encoding="utf-8"
    )
    classifier_metadata = _metadata()
    regressor_metadata = _metadata("random_forest_regressor")
    classifier_meta_path, regressor_meta_path = (
        tmp_path / "classification.json",
        tmp_path / "regression.json",
    )
    classifier_meta_path.write_text(json.dumps(classifier_metadata), encoding="utf-8")
    regressor_meta_path.write_text(json.dumps(regressor_metadata), encoding="utf-8")
    dataset_digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(json.dumps({"dataset": {"sha256": dataset_digest}}), encoding="utf-8")
    for name, value in {
        "CLASSIFICATION_METADATA_PATH": classifier_meta_path,
        "REGRESSION_METADATA_PATH": regressor_meta_path,
        "MODEL_COMPARISON_PATH": classifier_csv,
        "REGRESSION_COMPARISON_PATH": regressor_csv,
        "MODEL_DIAGNOSTICS_PATH": diagnostics,
        "MODEL_PATH": selected_model,
        "REGRESSION_MODEL_PATH": selected_regressor,
        "PROCESSED_DATASET_PATH": dataset,
        "RAW_DATASET_PATH": tmp_path / "missing-raw.csv",
    }.items():
        monkeypatch.setattr(comparison, name, value)
    inputs = [
        path.read_bytes()
        for path in (
            classifier_csv,
            regressor_csv,
            classifier_meta_path,
            regressor_meta_path,
            diagnostics,
            selected_model,
            selected_regressor,
        )
    ]
    report = comparison.build_model_run_comparison()
    outputs = [
        path.read_bytes()
        for path in (
            classifier_csv,
            regressor_csv,
            classifier_meta_path,
            regressor_meta_path,
            diagnostics,
            selected_model,
            selected_regressor,
        )
    ]
    assert report["tasks"][0]["status"] == "VALID"
    assert report["tasks"][1]["status"] == "VALID"
    assert report["tasks"][0]["runs"][0]["training_timestamp"] == ("2026-01-03T12:00:00+00:00")
    assert report["tasks"][0]["runs"][0]["evaluation_timestamp"] == ("2026-01-03T12:05:00+00:00")
    assert report["tasks"][0]["runs"][1]["metric_deltas_vs_selected"] == {
        "accuracy": -0.1,
        "f1": -0.1,
    }
    assert report["tasks"][1]["runs"][1]["metric_deltas_vs_selected"] == {"mae": 1.0, "rmse": 1.0}
    assert inputs == outputs
    assert "accuracy" not in report["tasks"][1]["runs"][0]["metrics"]
    assert str(tmp_path) not in json.dumps(report)


def test_model_run_comparison_endpoint_is_read_only_and_returns_synthetic_scope(
    monkeypatch,
):
    monkeypatch.setattr(
        model_run_routes,
        "build_model_run_comparison",
        lambda: {"status": "PARTIAL", "synthetic_data_only": True, "tasks": []},
    )
    client = TestClient(api.main.app)
    response = client.get("/model-runs/comparison")
    assert response.status_code == 200
    assert response.json()["synthetic_data_only"] is True
