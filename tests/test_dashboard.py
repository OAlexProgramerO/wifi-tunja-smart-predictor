"""Streamlit dashboard smoke coverage using its supported AppTest harness."""

import json
from pathlib import Path

import pandas as pd
import pytest
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


def test_architecture_diagnostics_dashboard_handles_unavailable_api(monkeypatch):
    from app import dashboard

    monkeypatch.setattr(
        dashboard,
        "main_api_request",
        lambda _path: (_ for _ in ()).throw(dashboard.ApiClientError("unavailable")),
    )
    app = AppTest.from_string(
        "from app.dashboard import _render_architecture_diagnostics\n"
        "_render_architecture_diagnostics()\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any("diagnostics are unavailable" in item.value for item in app.info)


def test_architecture_diagnostics_dashboard_renders_component_statuses(monkeypatch):
    from app import dashboard

    monkeypatch.setattr(
        dashboard,
        "main_api_request",
        lambda _path: {
            "components": [
                {
                    "id": "assistant_api",
                    "name": "Assistant API",
                    "responsibility": "Provides deterministic chat.",
                    "status": "unknown",
                    "check_status": "unavailable",
                    "message": "Readiness check unavailable.",
                }
            ]
        },
    )
    app = AppTest.from_string(
        "from app.dashboard import _render_architecture_diagnostics\n"
        "_render_architecture_diagnostics()\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any("Status check unavailable" in item.value for item in app.caption)


def test_operational_observability_dashboard_shows_counts_and_unavailable_status(monkeypatch):
    from app import dashboard

    monkeypatch.setattr(
        dashboard,
        "main_api_request",
        lambda _path: {
            "services": {
                "main_api": {
                    "availability": "available",
                    "readiness": "ready",
                    "metrics": {"requests_total": 12, "errors_total": 2},
                },
                "assistant_api": {
                    "availability": "unknown",
                    "readiness": "unknown",
                    "metrics": None,
                    "message": "Assistant check unavailable.",
                },
            }
        },
    )
    app = AppTest.from_string(
        "from app.dashboard import _render_operational_observability\n"
        "_render_operational_observability()\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert any(metric.value == "12" for metric in app.metric)
    assert any("Assistant check unavailable" in item.value for item in app.warning)


def test_model_artifact_inventory_dashboard_shows_status_and_actionable_details(monkeypatch):
    from app import dashboard

    monkeypatch.setattr(
        dashboard,
        "main_api_request",
        lambda _path: {
            "status": "INCOMPLETE",
            "artifacts": [
                {
                    "role": "classifier",
                    "artifact_name": "WiFi demand classifier",
                    "status": "READY",
                    "message": "Artifact and metadata checks passed.",
                    "training_metadata": {
                        "project_version": "0.3.0",
                        "selected_model": "random_forest",
                        "dataset_version": "synthetic_seed_42_v1",
                        "test_metrics": {"f1": 0.9},
                    },
                    "fingerprints": {"sha256": "a" * 64, "metadata_match": True},
                }
            ],
            "evaluation": {
                "status": "INCOMPLETE",
                "message": "Evaluation metadata is stale.",
            },
            "disclaimer": "Synthetic ML assets only.",
        },
    )
    app = AppTest.from_string(
        "from app.dashboard import _render_model_artifact_inventory\n"
        "_render_model_artifact_inventory()\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any("INCOMPLETE" in item.value for item in app.warning)
    assert any("Model version: 0.3.0" in item.value for item in app.caption)
    assert any("Evaluation metadata is stale" in item.value for item in app.caption)


def test_temporal_evaluation_controls_render_without_running_evaluation():
    app = AppTest.from_string(
        "from app.dashboard import _render_temporal_evaluation_lab\n"
        "_render_temporal_evaluation_lab()\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any("Temporal evaluation and synthetic drift lab" in item.value for item in app.markdown)
    assert any(item.label == "Chronological windows" for item in app.slider)
    assert any(item.label == "Features for distribution comparison" for item in app.multiselect)


def test_model_diagnostics_surface_model_version_date_and_classification_summary():
    script = """
import app.dashboard as dashboard
import wifi_tunja_smart_predictor.models.diagnostics as diagnostics
report = {
    "schema_version": "1.0",
    "model_version": "synthetic-model-1",
    "generated_at_utc": "2026-10-29T12:00:00+00:00",
    "test": {
        "rows": 6,
        "classification": {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "roc_auc": 0.75,
            "class_metrics": {"LOW": {"support": 3}, "HIGH": {"support": 3}},
            "confusion_matrix": [[2, 1], [1, 2]],
            "confusion_labels": ["LOW", "HIGH"],
        },
        "regression": {"mae": 2.0, "rmse": 3.0, "r2": 0.5, "mean_prediction_bias": 0.1},
    },
    "rolling_windows": [],
    "warnings": ["All evaluation data are synthetic."],
}
original_load = diagnostics.load_valid_diagnostics
original_render = dashboard.render_performance
original_json_file = dashboard._json_file
diagnostics.load_valid_diagnostics = lambda: report
dashboard.render_performance = lambda: None
dashboard._json_file = lambda _path: {}
dashboard.render_performance_v3()
diagnostics.load_valid_diagnostics = original_load
dashboard.render_performance = original_render
dashboard._json_file = original_json_file
"""
    app = AppTest.from_string(script, default_timeout=30).run()

    assert not app.exception
    assert any(
        "model synthetic-model-1" in item.value and "2026-10-29T12:00:00" in item.value
        for item in app.caption
    )
    assert {item.label for item in app.metric} >= {
        "Accuracy",
        "Precision (HIGH)",
        "Recall (HIGH)",
        "F1 (HIGH)",
        "ROC-AUC",
        "MAE",
        "RMSE",
        "R²",
        "Mean prediction bias",
    }
    metric_help = {item.label: item.help for item in app.metric}
    assert "higher is generally better" in metric_help["Accuracy"]
    assert "fewer false alarms" in metric_help["Precision (HIGH)"]
    assert "fewer missed cases" in metric_help["Recall (HIGH)"]
    assert "Balances HIGH precision and recall" in metric_help["F1 (HIGH)"]
    assert "lower is better" in metric_help["MAE"]
    assert "lower is better" in metric_help["RMSE"]
    assert "closer to 1" in metric_help["R²"]
    assert any("All evaluation data are synthetic" in item.value for item in app.caption)


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


def test_experiment_history_dashboard_renders_selected_experiment_comparison(monkeypatch):
    from app import dashboard

    requests = []
    markdown_requests = []

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
                        "status": "COMPLETED",
                        "experiment_type": "capacity_scenario",
                        "integrity": {"status": "verified", "reason_codes": []},
                    },
                    {
                        "id": "experiment-b",
                        "name": "Synthetic alternative",
                        "created_at_utc": "2026-01-02T00:00:00+00:00",
                        "status": "COMPLETED",
                        "experiment_type": "capacity_scenario",
                        "integrity": {"status": "verified", "reason_codes": []},
                    },
                ],
                "limit": 25,
                "offset": 25,
                "total_count": 2,
                "analytics": {
                    "total_experiment_count": 2,
                    "experiments_by_zone": [{"zone_id": "zone-1", "experiment_count": 2}],
                    "experiments_by_creation_date_utc": [
                        {"date_utc": "2026-01-01", "experiment_count": 1},
                        {"date_utc": "2026-01-02", "experiment_count": 1},
                    ],
                    "record_integrity_counts": {"verified": 2},
                    "numeric_summaries": {
                        "configured_capacity_connections": {
                            "observation_count": 2,
                            "statistics": {"minimum": 8, "maximum": 8, "mean": 8, "median": 8},
                        },
                        "allocation_budget_connections": {
                            "observation_count": 2,
                            "statistics": {"minimum": 20, "maximum": 20, "mean": 20, "median": 20},
                        },
                        "estimated_next_hour_connections": {
                            "observation_count": 2,
                            "statistics": {"minimum": 10, "maximum": 12, "mean": 11, "median": 11},
                        },
                    },
                    "missing_experiment_counts": {},
                },
            }
        if path.endswith("/report?format=json"):
            return {"report_schema_version": "1.0"}
        if path == "/experiments/compare":
            return {
                "schema_version": "1.0",
                "baseline_experiment_id": "experiment-a",
                "comparisons": [
                    {
                        "baseline_experiment_id": "experiment-a",
                        "experiment_id": "experiment-b",
                        "comparison_status": "comparable",
                        "scientific_comparability": "limited",
                        "shared_zones": ["zone-1"],
                        "changes": {
                            "zone-1": {
                                "demand_connections": 2.0,
                                "capacity_connections": 5.0,
                                "overload_connections": -3.0,
                            }
                        },
                        "interpretation_detail": "Synthetic scenario differences only; not a real-network forecast.",
                    }
                ],
            }
        experiment_id = path.rsplit("/", 1)[-1]
        return {
            "id": experiment_id,
            "name": (
                "Synthetic base" if experiment_id == "experiment-a" else "Synthetic alternative"
            ),
            "configuration": {
                "datetime": f"2026-01-0{1 if experiment_id == 'experiment-a' else 2}T00:00:00+00:00",
                "description": "Synthetic test scenario",
                "resource_budget_connections": 20,
                "zones": [{"zone_id": "zone-1", "capacity_connections": 8}],
            },
            "result": {
                "capacity_plan": {"zones": []},
                "zones": [
                    {
                        "zone_id": "zone-1",
                        "demand": 10.0 if experiment_id == "experiment-a" else 12.0,
                        "effective_capacity": 8.0 if experiment_id == "experiment-a" else 13.0,
                        "overload_connections": 2.0 if experiment_id == "experiment-a" else 0.0,
                    }
                ],
                "provenance": {
                    "dataset_version": "synthetic-v1",
                    "dataset_sha256": "a" * 64,
                    "dataset_fingerprint_algorithm": "sha256-canonical-v1",
                    "models": [
                        {
                            "prediction_id": f"prediction-{experiment_id}",
                            "application_version": "0.7.5",
                            "classifier": {
                                "selected_model": "random_forest",
                                "model_version": "model-v1",
                                "artifact_sha256": "b" * 64,
                            },
                            "regressor": {
                                "selected_model": "linear_regression",
                                "model_version": "model-v1",
                                "artifact_sha256": "c" * 64,
                            },
                        }
                    ],
                },
                "limitations": ["Synthetic scenario only."],
            },
            "status": "COMPLETED",
            "experiment_type": "capacity_scenario",
            "created_at_utc": f"2026-01-0{1 if experiment_id == 'experiment-a' else 2}T00:00:00+00:00",
            "integrity": {
                "status": "verified",
                "assessment_status": "assessable",
                "reason_codes": [],
            },
        }

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)

    def fake_markdown_request(path):
        markdown_requests.append(path)
        return "# Synthetic report"

    monkeypatch.setattr(dashboard, "main_api_text_request", fake_markdown_request)
    script = (
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n"
    )
    app = AppTest.from_string(script, default_timeout=30).run()
    assert not app.exception
    assert requests[0][0].startswith("/experiments?limit=25&offset=0")
    assert app.number_input[0].label == "History page"
    assert any("Showing 2 of 2 matching records" in caption.value for caption in app.caption)
    assert any(
        metric.label == "Experiments in filtered history" and metric.value == "2"
        for metric in app.metric
    )
    assert any(metric.label == "Created (UTC)" for metric in app.metric)
    assert any("Dataset version: synthetic-v1" in caption.value for caption in app.caption)
    assert any("Demand classification: Not recorded" in item.value for item in app.info)
    assert app.text_input[0].label == "Search experiment ID or name"
    assert app.text_input[1].label == "Synthetic zone ID"
    assert app.button[0].label == "Clear filters"

    assert app.selectbox[1].options == ["Select a verified experiment", "Synthetic alternative"]
    app.selectbox[1].set_value("experiment-b").run()
    app.button[1].click().run()
    assert (
        "/experiments/compare",
        {"experiment_ids": ["experiment-a", "experiment-b"]},
    ) in requests
    assert any("Synthetic scenario differences only" in item.value for item in app.info)
    assert "#### Selected experiments" in [item.value for item in app.markdown]
    assert any("Demand difference" in item.value.columns for item in app.dataframe)
    assert app.dataframe[-1].value.iloc[0]["Demand · comparison"] == 12.0
    assert app.dataframe[-1].value.iloc[0]["Demand difference"] == 2.0
    assert [item.label for item in app.download_button] == [
        "Download JSON comparison report",
        "Download Markdown comparison report",
        "Download JSON report",
        "Download Markdown report",
    ]
    assert markdown_requests
    assert set(markdown_requests) == {"/experiments/experiment-a/report?format=markdown"}
    app.download_button[0].click().run()
    assert any("Demand difference" in item.value.columns for item in app.dataframe)
    assert len(app.download_button) == 4

    app.selectbox[0].set_value("experiment-b").run()
    assert ("/experiments/experiment-b", None) in requests
    assert any("experiment-b" in caption.value for caption in app.caption)

    app.number_input[0].set_value(2).run()
    assert any(path == "/experiments?limit=25&offset=25" for path, _ in requests)
    assert "No experiments on this page" in app.info[-1].value
    assert any(path.endswith("/report?format=json") for path, _ in requests)


def test_experiment_history_search_filter_and_clear_restore_history(monkeypatch):
    from app import dashboard

    requests = []

    def fake_request(path, payload=None):
        requests.append(path)
        return {"items": [], "total_count": 0, "integrity_warnings": []}

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    app = AppTest.from_string(
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n",
        default_timeout=30,
    ).run()
    assert not app.exception
    app.text_input[0].set_value("base").run()
    assert "search=base" in requests[-1]
    app.text_input[1].set_value("zone-1").run()
    assert "zone_id=zone-1" in requests[-1]
    app.button[0].click().run()
    assert not app.exception
    assert "search=" not in requests[-1]
    assert "zone_id=" not in requests[-1]


def test_experiment_detail_handles_missing_fields_and_preserves_zero_and_false():
    app = AppTest.from_string(
        "from app.dashboard import _render_experiment_detail, _recorded_value\n"
        "_render_experiment_detail({'id': 'legacy', 'configuration': {}, 'result': {}, 'integrity': {}})\n"
        "assert _recorded_value(0) == '0'\n"
        "assert _recorded_value(False) == 'False'\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any(
        metric.label == "Created (UTC)" and metric.value == "Not recorded" for metric in app.metric
    )
    assert any("Demand classification: Not recorded" in item.value for item in app.info)
    assert any("provenance was not recorded" in item.value for item in app.info)


def test_experiment_history_analytics_renders_supported_filtered_metrics():
    analytics = {
        "total_experiment_count": 3,
        "experiments_by_zone": [
            {"zone_id": "zone-1", "experiment_count": 2},
            {"zone_id": "zone-2", "experiment_count": 1},
        ],
        "experiments_by_creation_date_utc": [{"date_utc": "2026-01-02", "experiment_count": 3}],
        "record_integrity_counts": {"verified": 2, "incomplete": 1},
        "numeric_summaries": {
            "configured_capacity_connections": {
                "observation_count": 3,
                "statistics": {
                    "minimum": 10.0,
                    "maximum": 30.0,
                    "mean": 20.0,
                    "median": 20.0,
                },
            },
            "allocation_budget_connections": {
                "observation_count": 2,
                "statistics": None,
            },
            "estimated_next_hour_connections": {
                "observation_count": 0,
                "statistics": None,
            },
        },
        "missing_experiment_counts": {
            "connection_estimates": 3,
            "dataset_fingerprint": 1,
        },
    }
    app = AppTest.from_string(
        "from app.dashboard import _render_experiment_history_analytics\n"
        f"_render_experiment_history_analytics({analytics!r})\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any(
        metric.label == "Experiments in filtered history" and metric.value == "3"
        for metric in app.metric
    )
    assert len(app.get("plotly_chart")) == 2
    assert any("incomplete: 1" in caption.value for caption in app.caption)
    assert any("no valid stored connection estimate: 3" in caption.value for caption in app.caption)
    assert any(
        "Insufficient observations for summary statistics" in str(frame.value.to_dict())
        for frame in app.dataframe
    )


def test_experiment_history_analytics_has_empty_filtered_state():
    app = AppTest.from_string(
        "from app.dashboard import _render_experiment_history_analytics\n"
        "_render_experiment_history_analytics({'total_experiment_count': 0})\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any("No saved experiments match the current filters" in item.value for item in app.info)
    assert not app.get("plotly_chart")


def test_experiment_history_handles_unknown_selected_experiment(monkeypatch):
    from app import dashboard
    from wifi_tunja_smart_predictor.api_client import ApiClientError

    def fake_request(path, payload=None):
        if path.startswith("/experiments?"):
            return {
                "items": [
                    {
                        "id": "missing-record",
                        "name": "Missing synthetic experiment",
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                        "integrity": {"status": "verified", "reason_codes": []},
                    }
                ],
                "total_count": 1,
            }
        raise ApiClientError("Main API returned HTTP 404.")

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    app = AppTest.from_string(
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any("HTTP 404" in item.value for item in app.error)


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
    monkeypatch.setattr(dashboard, "main_api_text_request", lambda path: "# Synthetic report")
    script = (
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n"
    )
    app = AppTest.from_string(script, default_timeout=30).run()

    assert not app.exception
    assert any("INVALID_PERSISTED_RECORD" in item.value for item in app.warning)
    assert any("MODEL_PROVENANCE_MISSING" in item.value for item in app.warning)
    assert any("evaluación no posible" in item.value for item in app.warning)
    assert app.selectbox[1].options == ["Select a verified experiment"]
    assert app.button[1].disabled


def test_experiment_history_empty_page_is_handled_without_comparison(monkeypatch):
    from app import dashboard

    monkeypatch.setattr(
        dashboard,
        "main_api_request",
        lambda path, payload=None: {"items": [], "integrity_warnings": []},
    )
    app = AppTest.from_string(
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any("No saved experiments yet" in item.value for item in app.info)
    assert not app.selectbox


def test_experiment_comparison_without_shared_zones_is_not_presented_as_equal():
    app = AppTest.from_string(
        "from app.dashboard import _render_experiment_comparison\n"
        "_render_experiment_comparison("
        "{'comparisons': [{'comparison_status': 'no_shared_zones', "
        "'interpretation_detail': 'These experiments share no zone IDs.'}]}, "
        "{'name': 'Baseline', 'configuration': {}, 'result': {'zones': []}}, "
        "{'name': 'Alternative', 'configuration': {}, 'result': {'zones': []}})\n",
        default_timeout=30,
    ).run()

    assert not app.exception
    assert any("share no zone IDs" in item.value for item in app.warning)
    assert not app.dataframe


@pytest.mark.parametrize(
    ("api_error", "expected_message"),
    [
        ("Main API returned HTTP 409.", "These experiments are incompatible"),
        ("Main API is unavailable at the configured MAIN_API_URL endpoint.", "API is unavailable"),
    ],
)
def test_experiment_history_handles_comparison_api_errors(monkeypatch, api_error, expected_message):
    from app import dashboard
    from wifi_tunja_smart_predictor.api_client import ApiClientError

    def fake_request(path, payload=None):
        if path.startswith("/experiments?"):
            return {
                "items": [
                    {
                        "id": experiment_id,
                        "name": experiment_id,
                        "created_at_utc": "2026-01-01T00:00:00+00:00",
                        "integrity": {"status": "verified", "reason_codes": []},
                    }
                    for experiment_id in ("experiment-a", "experiment-b")
                ],
                "integrity_warnings": [],
            }
        if path == "/experiments/compare":
            raise ApiClientError(api_error)
        if path.endswith("/report?format=json"):
            return {"report_schema_version": "1.0"}
        return {
            "result": {"capacity_plan": {"zones": []}},
            "integrity": {"status": "verified", "reason_codes": []},
        }

    monkeypatch.setattr(dashboard, "main_api_request", fake_request)
    monkeypatch.setattr(dashboard, "main_api_text_request", lambda path: "# Synthetic report")
    app = AppTest.from_string(
        "from app.dashboard import _render_page\n"
        "_render_page('Experiment History', 'Experiments')\n",
        default_timeout=30,
    ).run()
    app.selectbox[1].set_value("experiment-b").run()
    app.button[1].click().run()

    assert not app.exception
    assert any(expected_message in item.value for item in [*app.warning, *app.error])
