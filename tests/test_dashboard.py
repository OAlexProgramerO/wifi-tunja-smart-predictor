"""Streamlit dashboard smoke coverage using its supported AppTest harness."""

import json
from pathlib import Path

import pandas as pd
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


def test_geospatial_summary_aggregates_only_selected_synthetic_window(mini_frame):
    from app.dashboard import _geospatial_zone_summary

    frame = mini_frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    summary = _geospatial_zone_summary(
        frame,
        frame["timestamp"].min().date(),
        frame["timestamp"].min().date(),
    )
    assert (
        summary["observations"].sum()
        == frame["timestamp"].dt.date.eq(frame["timestamp"].min().date()).sum()
    )
    assert set(summary["zone_id"]).issubset(set(frame["zone_id"]))


def test_experiment_history_dashboard_requests_selected_page_and_labels_no_overlap(monkeypatch):
    from app import dashboard

    requests = []

    def fake_request(path, payload=None):
        requests.append((path, payload))
        if path.startswith("/experiments?"):
            if "offset=25" in path:
                return {"items": [], "limit": 25, "offset": 25}
            return {
                "items": [
                    {
                        "id": "experiment-a",
                        "name": "Synthetic base",
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                        "integrity": {"status": "verified", "reason_codes": []},
                    },
                    {
                        "id": "experiment-b",
                        "name": "Synthetic alternative",
                        "created_at_utc": "2026-01-02T00:00:00+00:00",
                        "integrity": {"status": "verified", "reason_codes": []},
                    },
                ],
                "limit": 25,
                "offset": 25,
            }
        if path.endswith("/report?format=json"):
            return {"report_schema_version": "1.0"}
        if path == "/experiments/compare":
            return {
                "comparisons": [
                    {
                        "comparison_status": "no_shared_zones",
                        "interpretation_detail": "These experiments share no zone IDs.",
                    }
                ]
            }
        experiment_id = path.rsplit("/", 1)[-1]
        return {
            "id": experiment_id,
            "name": (
                "Synthetic base" if experiment_id == "experiment-a" else "Synthetic alternative"
            ),
            "result": {"capacity_plan": {"zones": []}},
        }

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    script = (
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n"
    )
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert requests[0][0] == "/experiments?limit=25&offset=0"
    assert app.number_input[0].label == "History page"
    assert "Showing records 1–2" in app.caption[2].value

    app.multiselect[0].set_value(["experiment-b"]).run()
    app.button[0].click().run()
    assert any("These experiments share no zone IDs." in item.value for item in app.info)

    app.number_input[0].set_value(2).run()
    assert any(path == "/experiments?limit=25&offset=25" for path, _ in requests)
    assert "No experiments on this page" in app.info[-1].value


def test_experiment_history_warns_for_incomplete_records_and_excludes_comparison(monkeypatch):
    from app import dashboard

    def fake_request(path, payload=None):
        if path.startswith("/experiments?"):
            return {
                "items": [
                    {
                        "id": "legacy",
                        "name": "Legacy synthetic run",
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                        "integrity": {
                            "status": "incomplete",
                            "assessment_status": "not_assessable",
                            "reason_codes": ["MODEL_PROVENANCE_MISSING"],
                        },
                    },
                    {
                        "id": "verified",
                        "name": "Verified synthetic run",
                        "created_at_utc": "2026-01-02T00:00:00+00:00",
                        "integrity": {"status": "verified", "reason_codes": []},
                    },
                ],
                "integrity_warnings": [
                    {"experiment_id": "bad", "reason_code": "INVALID_PERSISTED_RECORD"}
                ],
            }
        if path.endswith("/report?format=json"):
            return {"report_schema_version": "1.1"}
        experiment_id = path.rsplit("/", 1)[-1]
        return {
            "id": experiment_id,
            "name": experiment_id,
            "result": {"capacity_plan": {"zones": []}},
        }

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    script = (
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n"
    )
    app = AppTest.from_string(script, default_timeout=30).run()

    assert not app.exception
    assert any("INVALID_PERSISTED_RECORD" in item.value for item in app.warning)
    assert any("MODEL_PROVENANCE_MISSING" in item.value for item in app.warning)
    assert any("evaluación no posible" in item.value for item in app.warning)
    assert app.multiselect[0].options == []
    assert app.button[0].disabled
