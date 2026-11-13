from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi.testclient import TestClient

import api.main
import wifi_tunja_smart_predictor.models.inventory as inventory
from wifi_tunja_smart_predictor.config import DATASET_VERSION, MODEL_INPUT_COLUMNS


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model_metadata(model: str, artifact: Path, *, regression: bool = False) -> dict:
    metrics = (
        {"mae": 2.5, "rmse": 3.5, "r2": 0.82}
        if regression
        else {
            "accuracy": 0.9,
            "precision": 0.88,
            "recall": 0.86,
            "f1": 0.87,
        }
    )
    return {
        "project_version": "0.3.0",
        "selected_model": model,
        "model_type": "RandomForestRegressor" if regression else "RandomForestClassifier",
        "training_run_id": "run-inventory-test",
        "training_timestamp_utc": "2026-01-02T03:04:05+00:00",
        "dataset_version": DATASET_VERSION,
        "preprocessing_version": "scenario-features-v1",
        "target_column": "connections_next_hour" if regression else "demand_level",
        "feature_count": len(MODEL_INPUT_COLUMNS),
        "feature_names": MODEL_INPUT_COLUMNS,
        "artifact_sha256": _sha256(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "train_period": {"start": "2025-01-01", "end": "2025-06-30", "rows": 12},
        "validation_period": {"start": "2025-07-01", "end": "2025-08-30", "rows": 4},
        "test_period": {"start": "2025-09-01", "end": "2025-09-30", "rows": 4},
        "validation_metrics": metrics,
        "test_metrics": metrics,
    }


def _inventory_files(tmp_path: Path, monkeypatch) -> dict[str, Path]:
    files = {
        "classifier": tmp_path / "classifier.joblib",
        "regressor": tmp_path / "regressor.joblib",
        "combined": tmp_path / "model.json",
        "classification": tmp_path / "classification.json",
        "regression": tmp_path / "regression.json",
        "diagnostics": tmp_path / "diagnostics.json",
        "dataset": tmp_path / "synthetic.csv",
    }
    files["classifier"].write_bytes(b"read-only classifier bytes")
    files["regressor"].write_bytes(b"read-only regressor bytes")
    files["dataset"].write_bytes(b"synthetic dataset rows")
    classifier = _model_metadata("random_forest", files["classifier"])
    regressor = _model_metadata("random_forest_regressor", files["regressor"], regression=True)
    classifier["regression"] = regressor
    for key, value in (
        ("combined", classifier),
        ("classification", _model_metadata("random_forest", files["classifier"])),
        ("regression", regressor),
    ):
        files[key].write_text(json.dumps(value), encoding="utf-8")
    report = {
        "schema_version": "1.0",
        "project_version": inventory.PROJECT_VERSION,
        "generated_at_utc": "2026-01-03T00:00:00+00:00",
        "dataset": {"sha256": _sha256(files["dataset"])},
        "artifacts": {
            "classifier_sha256": _sha256(files["classifier"]),
            "regressor_sha256": _sha256(files["regressor"]),
        },
    }
    files["diagnostics"].write_text(json.dumps(report), encoding="utf-8")

    monkeypatch.setattr(inventory, "MODEL_PATH", files["classifier"])
    monkeypatch.setattr(inventory, "REGRESSION_MODEL_PATH", files["regressor"])
    monkeypatch.setattr(inventory, "MODEL_METADATA_PATH", files["combined"])
    monkeypatch.setattr(inventory, "CLASSIFICATION_METADATA_PATH", files["classification"])
    monkeypatch.setattr(inventory, "REGRESSION_METADATA_PATH", files["regression"])
    monkeypatch.setattr(inventory, "MODEL_DIAGNOSTICS_PATH", files["diagnostics"])
    monkeypatch.setattr(inventory, "PROCESSED_DATASET_PATH", files["dataset"])
    monkeypatch.setattr(inventory, "RAW_DATASET_PATH", tmp_path / "absent.csv")
    return files


def test_inventory_reports_valid_artifacts_versions_fingerprints_and_metrics(tmp_path, monkeypatch):
    files = _inventory_files(tmp_path, monkeypatch)

    report = inventory.build_artifact_inventory()

    assert report["status"] == "READY"
    assert report["dataset"]["kind"] == "synthetic"
    assert report["evaluation"]["status"] == "READY"
    assert [item["status"] for item in report["artifacts"]] == ["READY", "READY"]
    classifier, regressor = report["artifacts"]
    assert classifier["training_metadata"]["project_version"] == "0.3.0"
    assert classifier["fingerprints"]["sha256"] == _sha256(files["classifier"])
    assert classifier["fingerprints"]["metadata_match"] is True
    assert classifier["training_metadata"]["test_metrics"]["f1"] == 0.87
    assert regressor["training_metadata"]["test_metrics"]["mae"] == 2.5
    assert str(tmp_path) not in json.dumps(report)


def test_inventory_reports_missing_artifact_without_attempting_repair(tmp_path, monkeypatch):
    files = _inventory_files(tmp_path, monkeypatch)
    before = {key: path.read_bytes() for key, path in files.items() if path.exists()}
    files["regressor"].unlink()

    report = inventory.build_artifact_inventory()

    regressor = next(item for item in report["artifacts"] if item["role"] == "regressor")
    assert report["status"] == "MISSING"
    assert regressor["status"] == "MISSING"
    assert regressor["availability"] == "missing"
    assert files["regressor"].exists() is False
    assert {key: path.read_bytes() for key, path in files.items() if path.exists()} == {
        key: value for key, value in before.items() if key != "regressor"
    }


def test_missing_evaluation_report_does_not_mark_present_model_files_missing(tmp_path, monkeypatch):
    files = _inventory_files(tmp_path, monkeypatch)
    files["diagnostics"].unlink()

    report = inventory.build_artifact_inventory()

    assert report["status"] == "INCOMPLETE"
    assert report["evaluation"]["status"] == "MISSING"
    assert [item["status"] for item in report["artifacts"]] == ["READY", "READY"]


def test_inventory_marks_malformed_metadata_incomplete_and_sanitized(tmp_path, monkeypatch):
    files = _inventory_files(tmp_path, monkeypatch)
    files["combined"].write_text("{ malformed", encoding="utf-8")

    report = inventory.build_artifact_inventory()

    assert report["status"] == "INCOMPLETE"
    assert all(item["status"] == "INCOMPLETE" for item in report["artifacts"])
    assert "malformed" in report["artifacts"][0]["message"]
    assert report["artifacts"][0]["fingerprints"]["sha256"] == _sha256(files["classifier"])
    assert str(tmp_path) not in json.dumps(report)


def test_inventory_reports_present_artifact_fingerprint_when_metadata_is_missing(
    tmp_path, monkeypatch
):
    files = _inventory_files(tmp_path, monkeypatch)
    files["regression"].unlink()

    report = inventory.build_artifact_inventory()

    regressor = next(item for item in report["artifacts"] if item["role"] == "regressor")
    assert regressor["status"] == "INCOMPLETE"
    assert regressor["fingerprints"]["sha256"] == _sha256(files["regressor"])
    assert regressor["fingerprints"]["recorded_sha256"] is None


def test_inventory_detects_conflicting_classifier_and_regressor_metadata(tmp_path, monkeypatch):
    files = _inventory_files(tmp_path, monkeypatch)
    combined = json.loads(files["combined"].read_text(encoding="utf-8"))
    combined["regression"]["selected_model"] = "different_regressor"
    files["combined"].write_text(json.dumps(combined), encoding="utf-8")

    report = inventory.build_artifact_inventory()

    assert report["status"] == "INCOMPLETE"
    assert all(item["status"] == "INCOMPLETE" for item in report["artifacts"])


def test_inventory_requires_retraining_for_recorded_artifact_fingerprint_mismatch(
    tmp_path, monkeypatch
):
    files = _inventory_files(tmp_path, monkeypatch)
    files["classifier"].write_bytes(b"replacement classifier bytes")

    report = inventory.build_artifact_inventory()

    classifier = next(item for item in report["artifacts"] if item["role"] == "classifier")
    assert report["status"] == "RETRAINING_REQUIRED"
    assert classifier["status"] == "RETRAINING_REQUIRED"
    assert classifier["fingerprints"]["metadata_match"] is False


def test_inventory_detects_evaluation_dataset_fingerprint_mismatch(tmp_path, monkeypatch):
    files = _inventory_files(tmp_path, monkeypatch)
    files["dataset"].write_bytes(b"changed synthetic data")

    report = inventory.build_artifact_inventory()

    assert report["status"] == "RETRAINING_REQUIRED"
    assert report["evaluation"]["status"] == "RETRAINING_REQUIRED"


def test_inventory_api_is_read_only_and_returns_safe_artifact_status(tmp_path, monkeypatch):
    files = _inventory_files(tmp_path, monkeypatch)
    tracked = [path for path in files.values() if path.exists()]
    before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked}

    response = TestClient(api.main.app).get("/model-artifacts/inventory")

    assert response.status_code == 200
    payload = response.json()
    assert payload["read_only"] is True
    assert payload["status"] == "READY"
    assert {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in tracked} == before
    assert str(tmp_path) not in response.text
