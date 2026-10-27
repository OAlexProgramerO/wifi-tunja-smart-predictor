"""Validated JSON and readable Markdown export for experiment records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from typing import Any

from wifi_tunja_smart_predictor.experiments.integrity import inspect_experiment_integrity
from wifi_tunja_smart_predictor.simulation.metrics import metric_definitions

SYNTHETIC_DISCLAIMER = (
    "All observations, locations, predictions, and capacity values are synthetic. "
    "This simulation does not represent or recommend configuration for real municipal infrastructure."
)


def build_experiment_report(
    record: dict[str, Any], *, generated_at_utc: datetime | None = None
) -> dict[str, Any]:
    """Build a versioned report envelope from a validated saved experiment."""
    result = record.get("result")
    configuration = record.get("configuration")
    if not isinstance(result, dict) or not isinstance(configuration, dict):
        raise ValueError("Experiment record is missing its configuration or result object.")
    integrity = inspect_experiment_integrity(record)
    if integrity["status"] == "invalid":
        raise ValueError("Experiment record failed integrity checks.")
    generated = generated_at_utc or datetime.now(UTC)
    if generated.tzinfo is None:
        raise ValueError("generated_at_utc must be timezone-aware")
    capacity = result.get("capacity_plan", {})
    provenance = result.get("provenance", {})
    if not isinstance(capacity, dict) or not isinstance(provenance, dict):
        raise ValueError("Experiment report metadata is malformed.")
    models = provenance.get("models", [])
    if not isinstance(models, list) or any(not isinstance(item, dict) for item in models):
        raise ValueError("Experiment model provenance is malformed.")
    what_if = result.get("what_if")
    if what_if is not None and not isinstance(what_if, dict):
        raise ValueError("Experiment what-if results are malformed.")
    optimization = result.get("optimization")
    if optimization is not None and not isinstance(optimization, dict):
        raise ValueError("Experiment optimization results are malformed.")
    robustness = result.get("robustness_analysis")
    if robustness is not None and not isinstance(robustness, dict):
        raise ValueError("Experiment robustness results are malformed.")
    return {
        "report_schema_version": "1.4",
        "generated_at_utc": generated.astimezone(UTC).isoformat(),
        "experiment": record,
        "integrity": integrity,
        "dataset_provenance": provenance.get("dataset_sha256") or "Not recorded",
        "dataset_fingerprint_algorithm": provenance.get("dataset_fingerprint_algorithm")
        or "Not recorded",
        "model_provenance": models,
        "scenario_analysis": what_if,
        "optimization_analysis": optimization,
        "robustness_analysis": robustness,
        "scenario_metric_registry": (
            what_if.get("metric_definitions", metric_definitions()) if what_if else None
        ),
        "application_version": (models[0].get("application_version") if models else None)
        or "Not recorded",
        "metric_definitions": {
            "demand_connections": "Model-estimated synthetic connections in the next hour.",
            "capacity_connections": "User-supplied simulated connection-count capacity, plus any allocated budget.",
            "utilization_pct": "Estimated demand divided by effective simulated capacity, times 100; null when capacity is zero.",
            "overload_connections": "max(0, estimated demand - effective simulated capacity).",
        },
        "units": "estimated connections; utilization is percent",
        "assumptions": (
            capacity.get("assumptions", [])
            if isinstance(capacity.get("assumptions", []), list)
            else []
        ),
        "warnings": (
            result.get("limitations", []) if isinstance(result.get("limitations", []), list) else []
        ),
        "disclaimer": SYNTHETIC_DISCLAIMER,
    }


def render_experiment_markdown(report: dict[str, Any]) -> str:
    """Render a safe human-readable report from the shared report envelope."""
    experiment = report["experiment"]

    def safe_text(value: object) -> str:
        return escape(str(value)).replace("|", r"\|").replace("`", r"\`").replace("\n", " ")

    lines = [
        f"# {safe_text(experiment.get('name', 'Saved experiment'))}",
        "",
        f"Report schema: {report['report_schema_version']}",
        f"Integrity: {report['integrity']['status']} ({report['integrity']['assessment_status']})",
        f"Integrity reason codes: {', '.join(report['integrity']['reason_codes']) or 'None'}",
        f"Generated (UTC): {report['generated_at_utc']}",
        f"Experiment ID: `{safe_text(experiment.get('id', 'unknown'))}`",
        f"Created (UTC): {safe_text(experiment.get('created_at_utc', 'unknown'))}",
        f"Dataset SHA-256: {safe_text(report.get('dataset_provenance'))}",
        f"Dataset fingerprint algorithm: {safe_text(report.get('dataset_fingerprint_algorithm'))}",
        f"Application version: {safe_text(report.get('application_version'))}",
        "",
        "## Simulated capacity by zone",
        "",
        "| Zone | Demand (connections) | Capacity (connections) | Utilization | Overload | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    result = experiment.get("result", {})
    zones = result.get("zones", []) if isinstance(result, dict) else []
    for row in zones if isinstance(zones, list) else []:
        if not isinstance(row, dict):
            continue
        utilization_value = row.get("utilization_pct")
        utilization = (
            "undefined"
            if not isinstance(utilization_value, (int, float))
            else f"{utilization_value:.1f}%"
        )
        demand = row.get("demand", 0)
        effective_capacity = row.get("effective_capacity", 0)
        overload = row.get("overload_connections", 0)
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (demand, effective_capacity, overload)
        ):
            raise ValueError("Experiment report contains malformed zone metrics.")
        lines.append(
            f"| {safe_text(row.get('zone_id', 'unknown'))} | {demand:.1f} | "
            f"{effective_capacity:.1f} | {utilization} | "
            f"{overload:.1f} | {safe_text(row.get('status', 'unknown'))} |"
        )
    lines.extend(["", "## Model artifact provenance", ""])
    for index, trace in enumerate(report.get("model_provenance", []), start=1):
        for role in ("classifier", "regressor"):
            artifact = trace.get(role) or {}
            lines.append(
                f"- Zone run {index} {role}: version "
                f"{safe_text(artifact.get('model_version') or 'Not recorded')}; "
                f"SHA-256 {safe_text(artifact.get('artifact_sha256') or 'Not recorded')}."
            )
    scenario_analysis = report.get("scenario_analysis")
    if isinstance(scenario_analysis, dict):
        baseline = scenario_analysis.get("baseline", {})
        comparisons = scenario_analysis.get("comparisons", [])
        if not isinstance(baseline, dict) or not isinstance(comparisons, list):
            raise ValueError("Experiment what-if report has invalid comparison data.")
        lines.extend(
            [
                "",
                "## What-if scenario comparison",
                "",
                f"Assessment: {safe_text(scenario_analysis.get('assessment_status', 'unavailable'))}",
                f"Baseline: {safe_text(baseline.get('name', 'Not recorded'))}",
                "",
                "| Alternative | Status | Affected zones | Demand delta (connections) | Overload delta (connections) |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for comparison in comparisons:
            if not isinstance(comparison, dict):
                raise ValueError("Experiment what-if report contains an invalid comparison.")
            deltas = comparison.get("aggregate_differences", {})
            if not isinstance(deltas, dict):
                raise ValueError(
                    "Experiment what-if report contains invalid aggregate differences."
                )
            demand_delta = deltas.get("predicted_demand_difference_connections")
            overload_delta = deltas.get("overload_difference_connections")
            demand_text = "Not assessable" if demand_delta is None else f"{demand_delta:.1f}"
            overload_text = "Not assessable" if overload_delta is None else f"{overload_delta:.1f}"
            lines.append(
                f"| {safe_text(comparison.get('alternative_name', 'Not recorded'))} | "
                f"{safe_text(comparison.get('assessment_status', 'unavailable'))} | "
                f"{comparison.get('affected_zone_count', 0)} | {demand_text} | {overload_text} |"
            )
        lines.extend(
            [
                "",
                "Capacity allocation and overload are deterministic scenario calculations. "
                "Changing capacity assumptions does not change predicted demand.",
            ]
        )
    optimization = report.get("optimization_analysis")
    if isinstance(optimization, dict):
        baseline = optimization.get("baseline", {})
        candidate = optimization.get("selected_candidate")
        configuration = experiment.get("configuration", {})
        problem = configuration.get("optimization", {}) if isinstance(configuration, dict) else {}
        if not isinstance(problem, dict):
            problem = {}
        lines.extend(
            [
                "",
                "## Allocation optimization",
                "",
                f"Status: {safe_text(optimization.get('optimization_status', 'unavailable'))}",
                f"Objective: {safe_text(optimization.get('objective', 'unavailable'))}",
                f"Algorithm: {safe_text(optimization.get('algorithm_id', 'unavailable'))} v{safe_text(optimization.get('algorithm_version', 'unavailable'))}",
                f"Search strategy: {safe_text(optimization.get('search_strategy', 'unavailable'))}",
                f"Total budget: {safe_text(problem.get('total_budget_connections', 'unavailable'))} connections",
                f"Stopping limit: {safe_text(optimization.get('max_allocation_steps', 'unavailable'))} allocation steps",
                f"Allocation steps evaluated: {safe_text(optimization.get('allocation_steps_evaluated', 'unknown'))}",
                f"Optimality proven: {safe_text(optimization.get('optimality_proven', False))}",
                f"Selection: {safe_text(optimization.get('selection_explanation', 'Not recorded'))}",
                f"Candidates evaluated: {safe_text(optimization.get('candidates_evaluated', 'unknown'))}",
                f"Feasible/infeasible candidates: {safe_text(optimization.get('feasible_candidates', 'unknown'))}/{safe_text(optimization.get('infeasible_candidates', 'unknown'))}",
                f"Reason codes: {safe_text(', '.join(optimization.get('reason_codes', [])) or 'None')}",
            ]
        )
        if problem.get("zones"):
            lines.append(
                "Constraints: "
                + safe_text(
                    "; ".join(
                        f"{zone.get('zone_id')}: {zone.get('minimum_allocation_connections', 0)}–{zone.get('maximum_allocation_connections', 'unbounded')}"
                        for zone in problem["zones"]
                        if isinstance(zone, dict)
                    )
                )
            )
        if optimization.get("missing_metrics"):
            lines.append(
                "Unassessed metrics: " + safe_text(", ".join(optimization["missing_metrics"]))
            )
        if isinstance(baseline, dict):
            lines.append(
                f"Baseline deficit: {safe_text(baseline.get('total_deficit_connections', 'unavailable'))} connections"
            )
        if isinstance(candidate, dict):
            lines.extend(
                [
                    f"Selected deficit: {safe_text(candidate.get('total_deficit_connections', 'unavailable'))} connections",
                    "",
                    "| Zone | Predicted demand | Allocation | Effective capacity | Deficit |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for row in candidate.get("zones", []):
                if isinstance(row, dict):
                    values = (
                        "zone_id",
                        "predicted_demand_connections",
                        "allocation_connections",
                        "effective_capacity_connections",
                        "deficit_connections",
                    )
                    lines.append(
                        "| "
                        + " | ".join(safe_text(row.get(key, "unavailable")) for key in values)
                        + " |"
                    )
        else:
            lines.append("No feasible candidate was selected.")
    robustness = report.get("robustness_analysis")
    if isinstance(robustness, dict):
        rate = robustness.get("feasibility_rate_under_tested_scenarios")
        rate_text = "Not assessable" if rate is None else f"{rate:.1%}"
        lines.extend(
            [
                "",
                "## Sensitivity and robustness analysis",
                "",
                f"Analysis status: {safe_text(robustness.get('analysis_status', 'unavailable'))}",
                f"Assessment: {safe_text(robustness.get('assessment_status', 'unavailable'))}",
                f"Parent optimization: `{safe_text(robustness.get('parent_experiment_id', 'unavailable'))}`",
                f"Objective: {safe_text(robustness.get('objective', 'unavailable'))}",
                f"Feasibility in tested scenarios: {rate_text} ({robustness.get('feasible_scenario_count', 0)}/{robustness.get('feasibility_rate_denominator_count', 0)} evaluated; not a probability)",
                f"Defined/evaluated: {robustness.get('scenarios_defined', 0)}/{robustness.get('scenarios_evaluated', 0)}",
                f"Feasible/infeasible/not assessable: {robustness.get('feasible_scenario_count', 0)}/{robustness.get('infeasible_scenario_count', 0)}/{robustness.get('not_assessable_scenario_count', 0)}",
                f"Optimizer searches with step limit: {robustness.get('optimizer_incomplete_scenario_count', 0)}",
                f"Reason codes: {safe_text(', '.join(robustness.get('reason_codes', [])) or 'None')}",
                "",
                "### Evaluated scenarios",
                "",
                "| Type | Varied parameter | Value | Budget | Feasibility | Original objective under variation | Reoptimized objective | Change | Search |",
                "|---|---|---:|---:|---|---:|---:|---:|---|",
            ]
        )
        for row in robustness.get("scenario_results", []):
            if not isinstance(row, dict):
                raise ValueError("Experiment robustness report contains an invalid scenario.")
            lines.append(
                "| "
                + " | ".join(
                    safe_text(row.get(key, "Not assessable"))
                    for key in (
                        "scenario_type",
                        "varied_parameter",
                        "varied_value",
                        "budget_connections",
                        "feasibility_status",
                        "original_allocation_objective_value",
                        "reoptimized_objective_value",
                        "objective_degradation",
                        "optimizer_status",
                    )
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "### Budget frontier",
                "",
                "| Budget | Feasibility | Objective | Allocated | Unallocated | Zones with deficit | Extra budget to meet minimums |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for point in robustness.get("budget_frontier", []):
            if isinstance(point, dict):
                lines.append(
                    "| "
                    + " | ".join(
                        safe_text(point.get(key, "Not assessable"))
                        for key in (
                            "budget_connections",
                            "feasibility_status",
                            "objective_value",
                            "total_allocated_connections",
                            "unallocated_budget_connections",
                            "zones_with_deficit",
                            "minimum_additional_budget_connections",
                        )
                    )
                    + " |"
                )
        lines.extend(
            [
                "",
                "### Zone allocation stability",
                "",
                "| Zone | Tested | Unchanged | Stability rate | Mean absolute change | Maximum absolute change | Own-demand stress changes |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for zone in robustness.get("zone_stability", []):
            if isinstance(zone, dict):
                stability = zone.get("allocation_stability_rate")
                stability_text = "Not assessable" if stability is None else f"{stability:.1%}"
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            safe_text(zone.get("zone_id", "unknown")),
                            safe_text(zone.get("assessable_scenario_count", "unknown")),
                            safe_text(zone.get("unchanged_allocation_count", "unknown")),
                            stability_text,
                            safe_text(
                                zone.get("mean_absolute_allocation_change", "Not assessable")
                            ),
                            safe_text(
                                zone.get("maximum_absolute_allocation_change", "Not assessable")
                            ),
                            safe_text(
                                zone.get("own_demand_stress_allocation_change_count", "unknown")
                            ),
                        ]
                    )
                    + " |"
                )
    lines.extend(["", "## Scenario configuration", "", "```json"])
    config_json = json.dumps(experiment.get("configuration", {}), indent=2, sort_keys=True)
    lines.append(config_json.replace("```", r"\u0060\u0060\u0060"))
    lines.extend(["```", "", "## Metric definitions", ""])
    lines.extend(
        f"- **{safe_text(name)}:** {safe_text(value)}"
        for name, value in report["metric_definitions"].items()
    )
    lines.extend(["", "## Assumptions and limitations", ""])
    lines.extend(f"- {safe_text(item)}" for item in [*report["assumptions"], *report["warnings"]])
    lines.extend(["", safe_text(report["disclaimer"])])
    return "\n".join(lines)


def serialize_experiment_json(report: dict[str, Any]) -> str:
    """Serialize a report deterministically other than its generation timestamp."""
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
