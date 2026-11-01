"""Focused coverage for synthetic saved-experiment comparison exports."""

import json
from datetime import UTC, datetime

import pytest

from wifi_tunja_smart_predictor.reporting.service import (
    SYNTHETIC_DISCLAIMER,
    build_experiment_comparison_report,
    render_experiment_comparison_markdown,
    serialize_experiment_json,
)


def _record(experiment_id: str, *, demand: float, capacity: float) -> dict:
    return {
        "id": experiment_id,
        "name": f"Synthetic {experiment_id}",
        "created_at_utc": "2026-10-31T12:00:00+00:00",
        "configuration": {
            "datetime": "2026-10-31T10:00:00",
            "zones": [{"zone_id": "zone-a", "capacity_connections": capacity}],
            "resource_budget_connections": 12,
            "description": "Exclude unrelated free-form request data",
        },
        "result": {
            "time_window": "2026-10-31T10:00:00",
            "zones": [
                {
                    "zone_id": "zone-a",
                    "demand": demand,
                    "effective_capacity": capacity,
                    "overload_connections": max(0.0, demand - capacity),
                }
            ],
            "provenance": {
                "dataset_version": "synthetic-v1",
                "dataset_sha256": "d" * 64,
                "dataset_fingerprint_algorithm": "sha256-v1",
                "models": [
                    {
                        "application_version": "0.6.7",
                        "classifier": {
                            "model_version": "classifier-v1",
                            "artifact_sha256": "a" * 64,
                        },
                        "regressor": {
                            "model_version": "regressor-v1",
                            "artifact_sha256": "b" * 64,
                        },
                    }
                ],
            },
        },
    }


def _comparison_payload() -> dict:
    return {
        "schema_version": "1.0",
        "baseline_experiment_id": "exp-a",
        "comparisons": [
            {
                "experiment_id": "exp-b",
                "baseline_experiment_id": "exp-a",
                "shared_zones": ["zone-a"],
                "comparison_status": "comparable",
                "scientific_comparability": "limited",
                "comparability_reasons": [],
                "comparability_reason_codes": [],
                "interpretation_detail": "Synthetic scenario differences only.",
                "changes": {
                    "zone-a": {
                        "demand_connections": 4.0,
                        "capacity_connections": 3.0,
                        "overload_connections": 1.0,
                    }
                },
            }
        ],
    }


def _build_report() -> dict:
    return build_experiment_comparison_report(
        _record("exp-a", demand=18.0, capacity=15.0),
        _record("exp-b", demand=22.0, capacity=18.0),
        _comparison_payload(),
        generated_at_utc=datetime(2026, 10, 31, 12, tzinfo=UTC),
    )


def test_comparison_json_preserves_ids_provenance_compatibility_and_disclaimer():
    report = json.loads(serialize_experiment_json(_build_report()))

    assert report["report_schema_version"] == "1.0"
    assert report["comparison_schema_version"] == "1.0"
    assert report["generated_at_utc"] == "2026-10-31T12:00:00+00:00"
    assert report["baseline_experiment_id"] == "exp-a"
    assert report["comparison_experiment_id"] == "exp-b"
    assert report["experiments"]["baseline"]["dataset_provenance"]["sha256"] == "d" * 64
    assert (
        report["experiments"]["comparison"]["model_provenance"][0]["classifier"]["artifact_sha256"]
        == "a" * 64
    )
    assert report["compatibility"]["status"] == "comparable"
    assert report["comparison"] == _comparison_payload()["comparisons"][0]
    assert report["disclaimer"] == SYNTHETIC_DISCLAIMER
    assert "description" not in report["experiments"]["baseline"]["scenario_context"]


def test_comparison_markdown_contains_both_ids_provenance_differences_and_disclaimer():
    markdown = render_experiment_comparison_markdown(_build_report())

    assert "`exp-a`" in markdown
    assert "`exp-b`" in markdown
    assert "Dataset SHA-256: " + "d" * 64 in markdown
    assert "classifier-v1" in markdown
    assert "Demand difference" in markdown
    assert "| zone-a | 18.0 | 22.0 | 4.0 |" in markdown
    assert SYNTHETIC_DISCLAIMER in markdown


def test_comparison_report_handles_missing_optional_metadata():
    baseline = {"id": "exp-a", "configuration": {}, "result": {}}
    compared = {"id": "exp-b", "configuration": {}, "result": {}}
    payload = {
        "baseline_experiment_id": "exp-a",
        "comparisons": [
            {
                "baseline_experiment_id": "exp-a",
                "experiment_id": "exp-b",
                "comparison_status": "no_shared_zones",
                "shared_zones": [],
                "changes": {},
            }
        ],
    }

    report = build_experiment_comparison_report(baseline, compared, payload)
    markdown = render_experiment_comparison_markdown(report)

    assert report["experiments"]["baseline"]["dataset_provenance"]["sha256"] == "Not recorded"
    assert report["experiments"]["comparison"]["model_provenance"] == []
    assert report["compatibility"]["scientific_comparability"] == "Not recorded"
    assert "Not recorded" in markdown
    assert SYNTHETIC_DISCLAIMER in markdown


@pytest.mark.parametrize(
    ("baseline", "compared", "payload"),
    [
        (_record("exp-a", demand=1, capacity=1), _record("exp-b", demand=1, capacity=1), {}),
        (
            {"id": "exp-a", "configuration": {}},
            _record("exp-b", demand=1, capacity=1),
            _comparison_payload(),
        ),
        (
            _record("exp-a", demand=1, capacity=1),
            _record("exp-b", demand=1, capacity=1),
            {"comparisons": []},
        ),
    ],
)
def test_comparison_report_rejects_invalid_or_missing_results(baseline, compared, payload):
    with pytest.raises(ValueError):
        build_experiment_comparison_report(baseline, compared, payload)
