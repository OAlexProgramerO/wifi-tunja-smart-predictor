from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.main
import api.routes.drift as drift_routes
import wifi_tunja_smart_predictor.models.drift as drift


@pytest.fixture
def drift_features(monkeypatch):
    monkeypatch.setattr(drift, "MODEL_INPUT_COLUMNS", ["hour", "zone_type"])
    monkeypatch.setattr(drift, "CATEGORICAL_FEATURES", ["zone_type"])
    monkeypatch.setattr(drift, "TARGET_COLUMNS", ["demand_level", "connections_next_hour"])
    monkeypatch.setattr(drift, "LEAKAGE_COLUMNS", ["demand_level", "connections_next_hour"])


def _frames(rows: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    reference = pd.DataFrame(
        {
            "hour": np.arange(rows, dtype=float),
            "zone_type": (["A", "B"] * ((rows + 1) // 2))[:rows],
        }
    )
    evaluation = reference.copy()
    return reference, evaluation


def _report(reference, evaluation, **kwargs):
    return drift.analyze_feature_drift(
        reference,
        evaluation,
        evaluation_split="test",
        source_features_fingerprint="a" * 64,
        **kwargs,
    )


def test_stable_numeric_and_categorical_distributions_are_stable(drift_features):
    reference, evaluation = _frames()

    report = _report(reference, evaluation)

    assert report["status"] == "STABLE"
    assert [item["status"] for item in report["feature_results"]] == ["STABLE", "STABLE"]
    assert report["feature_results"][0]["psi"] == 0
    assert report["feature_results"][1]["categorical_total_variation"] == 0


def test_shifted_numeric_distribution_triggers_drift(drift_features):
    reference, evaluation = _frames()
    evaluation["hour"] += 500

    report = _report(reference, evaluation)
    numeric = report["feature_results"][0]

    assert numeric["status"] == "DRIFT_DETECTED"
    assert numeric["psi"] >= drift.THRESHOLDS["psi_drift"]
    assert (
        report["dataset_fingerprints"]["reference_features_sha256"]
        != report["dataset_fingerprints"]["evaluation_features_sha256"]
    )


def test_categorical_distribution_change_reports_unseen_categories(drift_features):
    reference, evaluation = _frames()
    evaluation["zone_type"] = ["NEW"] * len(evaluation)

    report = _report(reference, evaluation)
    categorical = report["feature_results"][1]

    assert categorical["status"] == "DRIFT_DETECTED"
    assert categorical["unseen_categories"] == ["NEW"]
    assert categorical["categorical_total_variation"] == 1


def test_missing_values_are_counted_and_change_feature_status(drift_features):
    reference, evaluation = _frames()
    evaluation.loc[:24, "hour"] = np.nan

    report = _report(reference, evaluation)
    numeric = report["feature_results"][0]

    assert numeric["evaluation_missing_count"] == 25
    assert numeric["missing_proportion_difference"] == 0.25
    assert numeric["status"] == "DRIFT_DETECTED"


def test_malformed_numeric_values_are_counted_and_excluded(drift_features):
    reference, evaluation = _frames()
    evaluation["hour"] = evaluation["hour"].astype(object)
    evaluation.loc[0, "hour"] = "not-a-number"

    report = _report(reference, evaluation)
    numeric = report["feature_results"][0]

    assert numeric["kind"] == "numeric"
    assert numeric["evaluation_invalid_numeric_count"] == 1
    assert numeric["evaluation_non_missing_count"] == 99


def test_unseen_categories_with_small_share_are_warning(drift_features):
    reference, evaluation = _frames()
    evaluation.loc[0, "zone_type"] = "NEW"

    report = _report(reference, evaluation)
    categorical = report["feature_results"][1]

    assert categorical["unseen_category_count"] == 1
    assert categorical["status"] == "WARNING"


def test_insufficient_samples_do_not_produce_a_drift_statistic(drift_features):
    reference, evaluation = _frames(rows=5)

    report = _report(reference, evaluation, min_samples=20)

    assert report["status"] == "WARNING"
    assert all(item["status"] == "INSUFFICIENT_SAMPLES" for item in report["feature_results"])
    assert all(item["drift_score"] is None for item in report["feature_results"])


def test_fingerprints_are_recorded_and_targets_never_enter_analysis(drift_features):
    reference, evaluation = _frames()
    reference["demand_level"] = "LOW"
    reference["connections_next_hour"] = -999999
    evaluation["demand_level"] = "HIGH"
    evaluation["connections_next_hour"] = 999999
    reference["future_leak_feature"] = 1_000_000
    evaluation["future_leak_feature"] = -1_000_000

    report = _report(reference, evaluation)

    assert report["target_exclusion"]["analyzed_features"] == ["hour", "zone_type"]
    assert set(report["target_exclusion"]["excluded_target_columns"]) == {
        "demand_level",
        "connections_next_hour",
    }
    assert "future_leak_feature" not in report["target_exclusion"]["analyzed_features"]
    fingerprints = report["dataset_fingerprints"]
    assert fingerprints["source_features_sha256"] == "a" * 64
    assert fingerprints["reference_features_sha256"] == fingerprints["evaluation_features_sha256"]
    assert report["generated_at_utc"]
    assert report["configuration"]["thresholds"] == drift.THRESHOLDS
    assert "future_leak_feature" not in json.dumps(report)


def test_target_changes_do_not_change_feature_fingerprints(drift_features):
    reference, evaluation = _frames()
    first = _report(reference.assign(demand_level="LOW"), evaluation.assign(demand_level="LOW"))
    second = _report(reference.assign(demand_level="HIGH"), evaluation.assign(demand_level="HIGH"))

    assert (
        first["dataset_fingerprints"]["reference_features_sha256"]
        == second["dataset_fingerprints"]["reference_features_sha256"]
    )
    assert (
        first["dataset_fingerprints"]["evaluation_features_sha256"]
        == second["dataset_fingerprints"]["evaluation_features_sha256"]
    )


def test_missing_allowlisted_feature_is_reported_and_warns(drift_features):
    reference, evaluation = _frames()
    evaluation = evaluation.drop(columns=["zone_type"])

    report = _report(reference, evaluation)

    assert report["status"] == "WARNING"
    assert report["feature_coverage"]["missing_from_evaluation"] == ["zone_type"]
    missing = next(item for item in report["feature_results"] if item["feature"] == "zone_type")
    assert missing["status"] == "MISSING_FEATURE"


def test_future_named_features_are_blocked_even_if_added_to_model_allowlist(
    drift_features, monkeypatch
):
    reference, evaluation = _frames()
    reference["expected_next_hour_load"] = 1
    evaluation["expected_next_hour_load"] = 1000
    monkeypatch.setattr(
        drift,
        "MODEL_INPUT_COLUMNS",
        ["hour", "zone_type", "expected_next_hour_load", "connections_next_hour"],
    )
    report = _report(reference, evaluation)

    assert report["target_exclusion"]["approved_features"] == ["hour", "zone_type"]
    assert "expected_next_hour_load" not in json.dumps(report)


def test_current_report_builder_reads_temporal_partitions_without_writing(tmp_path, monkeypatch):
    source = tmp_path / "synthetic.csv"
    pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01T00:00:00",
                "2025-01-01T01:00:00",
                "2025-08-01T00:00:00",
                "2025-08-01T01:00:00",
                "2025-10-01T00:00:00",
                "2025-10-01T01:00:00",
            ],
            "hour": [0, 1, 0, 1, 22, 23],
            "zone_type": ["A", "B", "A", "B", "NEW", "NEW"],
            "demand_level": ["LOW", "HIGH", "LOW", "HIGH", "LOW", "HIGH"],
            "connections_next_hour": [1, 2, 1, 2, 1, 2],
        }
    ).to_csv(source, index=False)
    before = source.read_bytes()
    monkeypatch.setattr(drift, "MODEL_INPUT_COLUMNS", ["hour", "zone_type"])
    monkeypatch.setattr(drift, "CATEGORICAL_FEATURES", ["zone_type"])
    monkeypatch.setattr(drift, "PROCESSED_DATASET_PATH", source)
    monkeypatch.setattr(drift, "RAW_DATASET_PATH", tmp_path / "missing-raw.csv")

    report = drift.build_current_drift_report("test")

    assert report["evaluation_split"] == "test"
    source_frame = pd.read_csv(source, parse_dates=["timestamp"])
    expected_source_features = drift._fingerprint_features(source_frame[["hour", "zone_type"]])
    assert report["dataset_fingerprints"]["source_features_sha256"] == expected_source_features
    assert report["reference"]["rows"] == 2
    assert report["evaluation"]["rows"] == 2
    assert source.read_bytes() == before
    altered = pd.read_csv(source)
    altered["demand_level"] = ["HIGH", "LOW", "HIGH", "LOW", "HIGH", "LOW"]
    altered["connections_next_hour"] = [99, 98, 97, 96, 95, 94]
    altered.to_csv(source, index=False)
    altered_before = source.read_bytes()
    altered_report = drift.build_current_drift_report("test")
    assert (
        altered_report["dataset_fingerprints"]["source_features_sha256"]
        == report["dataset_fingerprints"]["source_features_sha256"]
    )
    assert source.read_bytes() == altered_before


def test_monitor_endpoint_returns_report_and_does_not_write(tmp_path, monkeypatch):
    from wifi_tunja_smart_predictor.models import drift as drift_module

    report = {"schema_version": "1.0", "status": "STABLE", "evaluation_split": "test"}
    monkeypatch.setattr(
        drift_routes,
        "build_current_drift_report",
        lambda split: report | {"evaluation_split": split},
    )
    output = tmp_path / "should-not-exist.json"
    client = TestClient(api.main.app)

    response = client.get("/drift/monitor?evaluation_split=validation")

    assert response.status_code == 200
    assert response.json()["evaluation_split"] == "validation"
    assert not output.exists()
    assert drift_module.REPORT_SCHEMA_VERSION == "1.0"


def test_cli_persisted_report_writer_uses_versioned_path_and_preserves_sources(
    tmp_path, monkeypatch
):
    from scripts import generate_drift_report

    raw = tmp_path / "synthetic.csv"
    raw.write_text("synthetic,source\n1,ok\n", encoding="utf-8")
    source_before = raw.read_bytes()
    report = {
        "schema_version": "1.0",
        "status": "STABLE",
        "evaluation_split": "test",
        "generated_at_utc": "2026-11-15T00:00:00+00:00",
    }
    monkeypatch.setattr(generate_drift_report, "DRIFT_REPORT_DIR", tmp_path / "drift")
    monkeypatch.setattr(generate_drift_report, "build_current_drift_report", lambda _split: report)
    monkeypatch.setattr("sys.argv", ["generate_drift_report.py", "--evaluation-split", "test"])

    assert generate_drift_report.main() == 0
    destination = tmp_path / "drift" / "synthetic_drift_v1.0_test.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == report
    assert raw.read_bytes() == source_before
