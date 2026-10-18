"""Streamlit dashboard smoke coverage using its supported AppTest harness."""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_overview_renders_with_v3_navigation(mini_frame, tmp_path, monkeypatch):
    from wifi_tunja_smart_predictor import config

    fake_processed = tmp_path / "processed.csv"
    mini_frame.to_csv(fake_processed, index=False)
    monkeypatch.setattr(config, "PROCESSED_DATASET_PATH", fake_processed)
    dashboard = Path(__file__).resolve().parents[1] / "app" / "dashboard.py"
    app = AppTest.from_file(str(dashboard), default_timeout=30).run()
    assert not app.exception
    assert [item.value for item in app.title] == ["Overview"]
    assert any(metric.label == "Synthetic access points" for metric in app.metric)


def test_scenario_result_separates_classification_and_regression_outputs():

    result = {
        "scenario": {
            "location": {"zone_name": "Synthetic zone", "access_point_id": "AP-1"},
            "scenario_time": "2025-01-01T12:00:00",
            "scenario_mode": "historical_analog",
        },
        "classification": {"predicted_demand_level": "HIGH", "probability_high": 0.8},
        "regression": {
            "predicted_connections_next_hour": 42.0,
            "prediction_interval_lower": 20.0,
            "prediction_interval_upper": 64.0,
            "interval_confidence": 0.9,
        },
        "capacity": {"predicted_capacity_utilization_pct": 42.0},
        "explanation": {"top_factors": []},
        "traceability": {
            "prediction_id": "test-id",
            "predicted_at_utc": "2026-10-18T12:00:00+00:00",
            "application_version": "0.4.2",
            "classifier": {
                "selected_model": "test-classifier",
                "model_version": "0.4.2",
                "artifact_sha256": "a" * 64,
            },
            "regressor": {
                "selected_model": "test-regressor",
                "model_version": "0.4.2",
                "artifact_sha256": "b" * 64,
            },
        },
    }

    script = (
        "import json\n"
        "from app.dashboard import _render_scenario_result\n"
        f"result = json.loads({json.dumps(json.dumps(result))})\n"
        "_render_scenario_result(result)\n"
    )
    app = AppTest.from_string(script).run()

    assert not app.exception
    assert [item.value for item in app.subheader] == [
        "Demand category · classification",
        "Next-hour connection estimate · regression",
    ]
    assert any(metric.label == "Probability of HIGH demand" for metric in app.metric)
    assert any(metric.label == "Estimated connections" for metric in app.metric)
    assert any("SHA-256" in code.value for code in app.code)
