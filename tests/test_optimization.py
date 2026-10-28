from datetime import datetime
from itertools import product

import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import app
from wifi_tunja_smart_predictor.optimization.schemas import (
    AllocationOptimizationRequest,
)
from wifi_tunja_smart_predictor.optimization.service import AllocationOptimizationService


def make_request(**overrides):
    values = {
        "name": "test optimization",
        "datetime": datetime(2025, 1, 1, 12),
        "total_budget_connections": 3,
        "zones": [
            {"zone_id": "B", "capacity_connections": 0},
            {"zone_id": "A", "capacity_connections": 0},
        ],
    }
    values.update(overrides)
    return AllocationOptimizationRequest.model_validate(values)


def test_optimizer_is_repeatable_and_uses_zone_id_for_ties():
    request = make_request()
    first = AllocationOptimizationService().optimize(request, {"A": 4, "B": 4})
    second = AllocationOptimizationService().optimize(request, {"A": 4, "B": 4})
    assert first.model_dump() == second.model_dump()
    assert [
        (row.zone_id, row.allocation_connections) for row in first.selected_candidate.zones
    ] == [
        ("A", 3),
        ("B", 0),
    ]
    assert first.optimality_proven


def test_weighted_objective_prioritizes_explicit_weight():
    request = make_request(
        objective="MINIMIZE_WEIGHTED_DEFICIT",
        zones=[
            {"zone_id": "low", "capacity_connections": 0, "priority_weight": 1},
            {"zone_id": "high", "capacity_connections": 0, "priority_weight": 5},
        ],
        total_budget_connections=1,
    )
    result = AllocationOptimizationService().optimize(request, {"low": 8, "high": 8})
    allocations = {
        row.zone_id: row.allocation_connections for row in result.selected_candidate.zones
    }
    assert allocations == {"high": 1, "low": 0}


def test_optimizer_returns_infeasible_without_candidate_when_minimums_exceed_budget():
    request = make_request(
        total_budget_connections=1,
        zones=[
            {"zone_id": "A", "capacity_connections": 0, "minimum_allocation_connections": 1},
            {"zone_id": "B", "capacity_connections": 0, "minimum_allocation_connections": 1},
        ],
    )
    result = AllocationOptimizationService().optimize(request, {"A": 5, "B": 5})
    assert result.optimization_status == "infeasible"
    assert result.selected_candidate is None
    assert "INFEASIBLE_CONSTRAINTS" in result.reason_codes


def test_search_limit_returns_feasible_best_so_far_without_proof():
    request = make_request(max_allocation_steps=1)
    result = AllocationOptimizationService().optimize(request, {"A": 8, "B": 8})
    assert result.optimization_status == "search_limit_reached"
    assert result.selected_candidate.feasible
    assert not result.optimality_proven
    assert "OPTIMALITY_NOT_PROVEN" in result.reason_codes


def test_invalid_demand_is_rejected():
    with pytest.raises(ValueError, match="finite non-negative"):
        AllocationOptimizationService().optimize(make_request(), {"A": float("nan"), "B": 2})
    with pytest.raises(ValueError, match="exactly the requested zone IDs"):
        AllocationOptimizationService().optimize(make_request(), {"A": 2})


def test_zero_budget_and_twelve_zone_limit_remain_feasible():
    zones = [{"zone_id": f"zone_{index:02}", "capacity_connections": 0} for index in range(12)]
    request = make_request(zones=zones, total_budget_connections=0)
    result = AllocationOptimizationService().optimize(
        request, {zone["zone_id"]: 0 for zone in zones}
    )
    assert result.optimization_status == "optimal"
    assert result.selected_candidate.feasible
    assert result.selected_candidate.allocated_connections == 0
    assert result.selected_candidate.unallocated_budget_connections == 0


def test_budget_above_zone_maxima_stops_without_unnecessary_allocation():
    request = make_request(
        zones=[
            {
                "zone_id": "A",
                "capacity_connections": 0,
                "maximum_allocation_connections": 1,
            }
        ],
        total_budget_connections=10,
    )
    result = AllocationOptimizationService().optimize(request, {"A": 10})
    assert result.selected_candidate.allocated_connections == 1
    assert result.selected_candidate.unallocated_budget_connections == 9
    assert "ZONE_MAXIMUMS_LIMITED_ALLOCATION" in result.constraints_that_limited_allocation
    assert result.optimality_proven


@pytest.mark.parametrize(
    "objective,weights",
    [
        ("MINIMIZE_TOTAL_DEFICIT", [None, None, None]),
        ("MINIMIZE_WEIGHTED_DEFICIT", [1, 3, 2]),
    ],
)
def test_optimizer_matches_bounded_exhaustive_objective(objective, weights):
    rows = [
        {
            "zone_id": zone_id,
            "capacity_connections": capacity,
            "minimum_allocation_connections": minimum,
            "maximum_allocation_connections": maximum,
            **({"priority_weight": weight} if weight is not None else {}),
        }
        for zone_id, capacity, minimum, maximum, weight in zip(
            ["a", "b", "c"], [1, 0, 2], [0, 1, 0], [3, 3, 2], weights
        )
    ]
    demand = {"a": 6.5, "b": 4.0, "c": 5.25}
    request = make_request(
        objective=objective,
        zones=rows,
        total_budget_connections=5,
    )
    result = AllocationOptimizationService().optimize(request, demand)
    actual = result.selected_candidate.weighted_deficit_score
    feasible_scores = []
    for allocations in product(range(4), range(1, 4), range(3)):
        if sum(allocations) > request.total_budget_connections:
            continue
        score = sum(
            (weights[index] or 1)
            * max(0, demand[zone_id] - rows[index]["capacity_connections"] - allocations[index])
            for index, zone_id in enumerate(("a", "b", "c"))
        )
        feasible_scores.append(score)
    assert actual == min(feasible_scores)
    assert result.optimality_proven


def test_request_rejects_duplicate_zone_and_missing_weight():
    with pytest.raises(ValueError, match="unique"):
        make_request(zones=[{"zone_id": "A", "capacity_connections": 0}] * 2)
    with pytest.raises(ValueError, match="requires a priority weight"):
        make_request(objective="MINIMIZE_WEIGHTED_DEFICIT")


def test_optimize_api_persists_and_exports_report(tmp_path, monkeypatch):
    from api.routes import experiments

    monkeypatch.setattr(
        experiments, "_store", lambda: experiments.ExperimentStore(tmp_path / "experiments.sqlite3")
    )

    class FakePrediction:
        def predict(self, scenario):
            from types import SimpleNamespace

            return SimpleNamespace(
                to_dict=lambda: {
                    "scenario": {
                        "dataset_version": "synthetic-v1",
                        "dataset_sha256": "d" * 64,
                        "dataset_fingerprint_algorithm": "sha256-v1",
                    },
                    "regression": {"predicted_connections_next_hour": 12.0},
                    "traceability": {
                        "application_version": "0.6.3",
                        "classifier": {
                            "artifact_sha256": "a" * 64,
                            "model_version": "model-v1",
                        },
                        "regressor": {
                            "artifact_sha256": "b" * 64,
                            "model_version": "model-v1",
                        },
                    },
                }
            )

    monkeypatch.setattr(experiments, "_scenario_service", lambda: FakePrediction())
    client = TestClient(app)
    response = client.post(
        "/experiments/optimize",
        json={
            "name": "api optimization",
            "datetime": "2025-01-01T12:00:00",
            "objective": "MINIMIZE_TOTAL_DEFICIT",
            "total_budget_connections": 4,
            "zones": [{"zone_id": "zone_1", "capacity_connections": 2}],
        },
    )
    assert response.status_code == 201, response.text
    assert response.headers.get("x-request-id")
    record = response.json()
    assert record["result"]["optimization"]["selected_candidate"]["allocated_connections"] == 4
    assert record["integrity"]["status"] == "verified"
    assert client.get(f"/experiments/{record['id']}").status_code == 200
    assert any(
        row["id"] == record["id"]
        for row in client.get("/experiments?limit=10&offset=0").json()["items"]
    )
    report = client.get(f"/experiments/{record['id']}/report?format=json").json()
    assert report["report_schema_version"] == "1.4"
    assert report["optimization_analysis"]["objective"] == "MINIMIZE_TOTAL_DEFICIT"
    markdown = client.get(f"/experiments/{record['id']}/report?format=markdown")
    assert markdown.status_code == 200
    assert client.get(f"/experiments/{record['id']}").json()["integrity"]["status"] == "verified"
    assert "Allocation optimization" in markdown.text
    assert "Feasible/infeasible candidates" in markdown.text
    assert "/experiments/optimize" in client.get("/openapi.json").json()["paths"]
    invalid = client.post(
        "/experiments/optimize",
        json={
            "name": "missing priority",
            "datetime": "2025-01-01T12:00:00",
            "objective": "MINIMIZE_WEIGHTED_DEFICIT",
            "total_budget_connections": 1,
            "zones": [{"zone_id": "zone_1", "capacity_connections": 2}],
        },
    )
    assert invalid.status_code == 422


def test_dashboard_renders_optimization_comparison_chart():
    app_test = AppTest.from_string(
        "import streamlit as st\n"
        "from app.dashboard import _render_optimization_record\n"
        "_render_optimization_record(st.session_state['record'])\n",
        default_timeout=30,
    )
    app_test.session_state["record"] = {
        "result": {
            "optimization": {
                "optimality_proven": True,
                "selection_explanation": "Deterministic allocation.",
                "objective_improvement": 2.0,
                "baseline": {
                    "total_deficit_connections": 4,
                    "zones": [{"zone_id": "synthetic_a", "allocation_connections": 0}],
                },
                "selected_candidate": {
                    "total_deficit_connections": 2,
                    "zones": [
                        {
                            "zone_id": "synthetic_a",
                            "allocation_connections": 2,
                            "predicted_demand_connections": 8,
                            "effective_capacity_connections": 6,
                            "deficit_connections": 2,
                        }
                    ],
                },
            }
        }
    }
    app_test.run()
    assert not app_test.exception


def test_robustness_sensitivity_and_budget_sweep_are_deterministic():
    from wifi_tunja_smart_predictor.optimization.robustness import (
        RobustnessAnalysisRequest,
        RobustnessAnalysisService,
    )

    request = make_request(
        total_budget_connections=2,
        zones=[
            {"zone_id": "A", "capacity_connections": 1},
            {"zone_id": "B", "capacity_connections": 1},
        ],
    )
    optimized = AllocationOptimizationService().optimize(request, {"A": 4.25, "B": 3.0})
    parent = {
        "id": "parent-id",
        "configuration": {
            "scenario_kind": "allocation_optimization",
            "optimization": request.model_dump(mode="json"),
        },
        "result": {"optimization": optimized.model_dump(mode="json")},
        "integrity": {"status": "verified"},
    }
    analysis_request = RobustnessAnalysisRequest(
        name="sensitivity",
        parent_experiment_id="parent-id",
        demand_multiplier_min=0.5,
        demand_multiplier_max=1.5,
        demand_multiplier_step=0.5,
        budget_min_connections=1,
        budget_max_connections=3,
        budget_step_connections=1,
    )
    service = RobustnessAnalysisService()
    first = service.analyze(parent, analysis_request)
    second = service.analyze(parent, analysis_request)
    assert first.model_dump() == second.model_dump()
    assert first.analysis_status == "completed"
    assert first.assessment_status == "assessable"
    assert first.feasibility_rate_denominator_count == first.scenarios_evaluated
    assert any(row.scenario_type == "demand_stress" for row in first.scenario_results)
    assert any(row.scenario_type == "budget_sweep" for row in first.scenario_results)
    stress = next(row for row in first.scenario_results if row.scenario_type == "demand_stress")
    assert stress.original_allocation_objective_value != stress.reoptimized_objective_value
    assert stress.demand_multiplier in {0.5, 1.5}
    assert all(row.assessable_scenario_count > 0 for row in first.zone_stability)
    assert first.most_sensitive_zones[0] in {"A", "B"}


def test_robustness_reports_hard_minimum_budget_shortfall_and_partial_limit():
    from wifi_tunja_smart_predictor.optimization.robustness import (
        RobustnessAnalysisRequest,
        RobustnessAnalysisService,
    )

    request = make_request(
        total_budget_connections=1,
        zones=[
            {"zone_id": "A", "capacity_connections": 0, "minimum_allocation_connections": 2},
            {"zone_id": "B", "capacity_connections": 0, "minimum_allocation_connections": 2},
        ],
    )
    optimized = AllocationOptimizationService().optimize(request, {"A": 4, "B": 4})
    parent = {
        "id": "parent-id",
        "configuration": {
            "scenario_kind": "allocation_optimization",
            "optimization": request.model_dump(mode="json"),
        },
        "result": {"optimization": optimized.model_dump(mode="json")},
        "integrity": {"status": "verified"},
    }
    result = RobustnessAnalysisService().analyze(
        parent,
        RobustnessAnalysisRequest(
            name="partial",
            parent_experiment_id="parent-id",
            budget_min_connections=0,
            budget_max_connections=2,
            budget_step_connections=1,
            max_evaluations=1,
        ),
    )
    assert result.analysis_status == "partial"
    assert result.scenarios_evaluated == 1 < result.scenarios_defined
    assert result.scenario_results[0].minimum_additional_budget_connections == 3
    assert result.scenario_results[0].feasibility_status == "infeasible"
    assert result.feasibility_rate_denominator_count == 1
    assert result.feasibility_rate_under_tested_scenarios == 0


def test_robustness_request_rejects_unbounded_or_reversed_ranges():
    from pydantic import ValidationError

    from wifi_tunja_smart_predictor.optimization.robustness import RobustnessAnalysisRequest

    with pytest.raises(ValidationError):
        RobustnessAnalysisRequest(
            name="bad",
            parent_experiment_id="parent-id",
            demand_multiplier_min=1.5,
            demand_multiplier_max=1,
        )


def test_robustness_retains_unverified_parent_provenance_without_claiming_verification():
    from wifi_tunja_smart_predictor.optimization.robustness import (
        RobustnessAnalysisRequest,
        RobustnessAnalysisService,
    )

    request = make_request(zones=[{"zone_id": "A", "capacity_connections": 1}])
    optimization = AllocationOptimizationService().optimize(request, {"A": 3})
    parent = {
        "id": "legacy-parent",
        "configuration": {
            "scenario_kind": "allocation_optimization",
            "optimization": request.model_dump(mode="json"),
        },
        "result": {"optimization": optimization.model_dump(mode="json")},
        "integrity": {"status": "incomplete"},
    }
    result = RobustnessAnalysisService().analyze(
        parent,
        RobustnessAnalysisRequest(name="legacy", parent_experiment_id="legacy-parent"),
    )
    assert result.assessment_status == "not_assessable"
    assert result.parent_integrity_status == "incomplete"
    assert "PARENT_PROVENANCE_INCOMPLETE" in result.reason_codes


def test_robustness_api_persists_and_exports_analysis(tmp_path, monkeypatch):
    from api.routes import experiments

    monkeypatch.setattr(
        experiments, "_store", lambda: experiments.ExperimentStore(tmp_path / "history.sqlite3")
    )

    class FakePrediction:
        def predict(self, scenario):
            from types import SimpleNamespace

            return SimpleNamespace(
                to_dict=lambda: {
                    "scenario": {
                        "dataset_version": "synthetic-v1",
                        "dataset_sha256": "d" * 64,
                        "dataset_fingerprint_algorithm": "sha256-v1",
                    },
                    "regression": {"predicted_connections_next_hour": 12.0},
                    "traceability": {
                        "application_version": "0.6.3",
                        "classifier": {"artifact_sha256": "a" * 64, "model_version": "m1"},
                        "regressor": {"artifact_sha256": "b" * 64, "model_version": "m1"},
                    },
                }
            )

    monkeypatch.setattr(experiments, "_scenario_service", lambda: FakePrediction())
    client = TestClient(app)
    parent_response = client.post(
        "/experiments/optimize",
        json={
            "name": "parent optimization",
            "datetime": "2025-01-01T12:00:00",
            "total_budget_connections": 40,
            "zones": [{"zone_id": "zone_1", "capacity_connections": 2}],
        },
    )
    assert parent_response.status_code == 201, parent_response.text
    parent = parent_response.json()
    response = client.post(
        "/experiments/robustness",
        json={
            "name": "bounded robustness",
            "parent_experiment_id": parent["id"],
            "demand_multiplier_min": 0.8,
            "demand_multiplier_max": 1.2,
            "demand_multiplier_step": 0.2,
            "budget_min_connections": 2,
            "budget_max_connections": 6,
            "budget_step_connections": 2,
            "max_evaluations": 10,
            "max_allocation_steps_per_evaluation": 100,
        },
    )
    assert response.status_code == 201, response.text
    robustness = response.json()
    assert robustness["integrity"]["status"] == "verified"
    assert robustness["result"]["robustness_analysis"]["analysis_status"] == "completed"
    assert client.get(f"/experiments/{robustness['id']}").status_code == 200
    assert any(
        row["id"] == robustness["id"]
        for row in client.get("/experiments?limit=10&offset=0").json()["items"]
    )
    report = client.get(f"/experiments/{robustness['id']}/report?format=json").json()
    assert report["report_schema_version"] == "1.4"
    assert report["robustness_analysis"]["parent_experiment_id"] == parent["id"]
    markdown = client.get(f"/experiments/{robustness['id']}/report?format=markdown")
    assert markdown.status_code == 200
    assert "Sensitivity and robustness analysis" in markdown.text
    assert "not a probability" in markdown.text
    assert "/experiments/robustness" in client.get("/openapi.json").json()["paths"]
    missing = client.post(
        "/experiments/robustness",
        json={"name": "missing", "parent_experiment_id": "missing-id"},
    )
    assert missing.status_code == 404
    over_limit = client.post(
        "/experiments/robustness",
        json={
            "name": "parent budget adds twenty-first point",
            "parent_experiment_id": parent["id"],
            "budget_min_connections": 0,
            "budget_max_connections": 19,
            "budget_step_connections": 1,
        },
    )
    assert over_limit.status_code == 422


def test_dashboard_renders_robustness_record_without_api_call():
    app_test = AppTest.from_string(
        "import streamlit as st\n"
        "from app.dashboard import _render_robustness_record\n"
        "_render_robustness_record(st.session_state['record'], spanish=False)\n",
        default_timeout=30,
    )
    app_test.session_state["record"] = {
        "id": "saved",
        "result": {
            "robustness_analysis": {
                "analysis_status": "partial",
                "assessment_status": "not_assessable",
                "feasible_scenario_count": 1,
                "feasibility_rate_denominator_count": 2,
                "scenario_results": [],
                "budget_frontier": [],
                "zone_stability": [],
                "synthetic_data_disclaimer": "Synthetic only.",
            }
        },
    }
    app_test.run()
    assert not app_test.exception
    assert any("Partial analysis" in item.value for item in app_test.warning)
    assert any("Feasible / evaluated cases" == metric.label for metric in app_test.metric)


def test_robustness_workbench_renders_form():
    app_test = AppTest.from_string(
        "from app.dashboard import render_scenario_workbench\n"
        "from tests.conftest import make_mini_frame\n"
        "render_scenario_workbench(make_mini_frame(8))\n",
        default_timeout=30,
    ).run()
    workflow = next(widget for widget in app_test.radio if widget.label == "Workflow")
    workflow.set_value("Analyze robustness").run()
    assert not app_test.exception
    assert any(widget.label == "Saved optimization ID" for widget in app_test.text_input)
    assert any(widget.label == "Maximum evaluated scenarios" for widget in app_test.number_input)
