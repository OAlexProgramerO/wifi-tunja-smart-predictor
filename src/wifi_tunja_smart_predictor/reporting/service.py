"""Validated JSON and readable Markdown export for experiment records."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from html import escape
from typing import Any

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
    return {
        "report_schema_version": "1.0",
        "generated_at_utc": generated.astimezone(UTC).isoformat(),
        "experiment": record,
        "dataset_provenance": provenance.get("dataset_sha256"),
        "dataset_fingerprint_algorithm": provenance.get("dataset_fingerprint_algorithm"),
        "model_provenance": models,
        "application_version": (models[0].get("application_version") if models else None),
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
    for row in experiment.get("result", {}).get("zones", []):
        utilization_value = row.get("utilization_pct")
        utilization = "undefined" if utilization_value is None else f"{utilization_value:.1f}%"
        lines.append(
            f"| {safe_text(row.get('zone_id', 'unknown'))} | {row.get('demand', 0):.1f} | "
            f"{row.get('effective_capacity', 0):.1f} | {utilization} | "
            f"{row.get('overload_connections', 0):.1f} | {safe_text(row.get('status', 'unknown'))} |"
        )
    lines.extend(["", "## Model artifact provenance", ""])
    for index, trace in enumerate(report.get("model_provenance", []), start=1):
        for role in ("classifier", "regressor"):
            artifact = trace.get(role) or {}
            lines.append(
                f"- Zone run {index} {role}: version {safe_text(artifact.get('model_version'))}; "
                f"SHA-256 {safe_text(artifact.get('artifact_sha256'))}."
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
