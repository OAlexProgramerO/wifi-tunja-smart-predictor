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
                        "application_version": "0.6.1",
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
    assert report["report_schema_version"] == "1.3"
    assert report["optimization_analysis"]["objective"] == "MINIMIZE_TOTAL_DEFICIT"
    markdown = client.get(f"/experiments/{record['id']}/report?format=markdown")
    assert markdown.status_code == 200
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
    assert any(metric.label == "Selected deficit" for metric in app_test.metric)
    assert len(app_test.dataframe) == 1
