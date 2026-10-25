"""Versioned, shared definitions for model and scenario-workbench metrics."""

from __future__ import annotations

METRIC_REGISTRY_VERSION = "1.0"

METRIC_REGISTRY = {
    "predicted_demand_connections": {
        "label_en": "Predicted next-hour connections",
        "label_es": "Conexiones predichas para la próxima hora",
        "definition": "Regressor output for the selected synthetic zone and scenario time.",
        "formula": "existing regression model output",
        "unit": "connections",
        "source": "model_prediction",
        "valid_range": "finite, non-negative",
        "unavailable_when": "required model artifact or scenario context is unavailable",
    },
    "predicted_demand_level": {
        "label_en": "Predicted demand category",
        "label_es": "Categoría de demanda predicha",
        "definition": "Classifier output LOW or HIGH for the same scenario inputs.",
        "formula": "existing classification model output",
        "unit": "category",
        "source": "model_prediction",
        "valid_range": "LOW or HIGH",
        "unavailable_when": "required model artifact or scenario context is unavailable",
    },
    "assumed_capacity_connections": {
        "label_en": "Assumed capacity",
        "label_es": "Capacidad supuesta",
        "definition": "User-supplied simulated connection-count capacity.",
        "formula": "scenario input",
        "unit": "connections",
        "source": "user_assumption",
        "valid_range": "0 to 100000",
        "unavailable_when": "not supplied for the selected zone",
    },
    "effective_capacity_connections": {
        "label_en": "Effective simulated capacity",
        "label_es": "Capacidad simulada efectiva",
        "definition": "Assumed capacity plus deterministic budget allocation.",
        "formula": "assumed capacity + allocated budget",
        "unit": "connections",
        "source": "deterministic_simulation",
        "valid_range": "finite, non-negative",
        "unavailable_when": "capacity or allocation input is invalid",
    },
    "capacity_surplus_connections": {
        "label_en": "Capacity surplus proxy",
        "label_es": "Excedente de capacidad simulado",
        "definition": "Positive amount by which effective simulated capacity exceeds predicted demand.",
        "formula": "max(effective capacity - predicted demand, 0)",
        "unit": "connections",
        "source": "deterministic_simulation",
        "valid_range": "finite, non-negative",
        "unavailable_when": "prediction or effective capacity is unavailable",
    },
    "capacity_deficit_connections": {
        "label_en": "Capacity deficit proxy",
        "label_es": "Déficit de capacidad simulado",
        "definition": "Positive amount by which predicted demand exceeds effective simulated capacity.",
        "formula": "max(predicted demand - effective capacity, 0)",
        "unit": "connections",
        "source": "deterministic_simulation",
        "valid_range": "finite, non-negative",
        "unavailable_when": "prediction or effective capacity is unavailable",
    },
    "utilization_proxy_pct": {
        "label_en": "Simulated utilization proxy",
        "label_es": "Proxy de utilización simulada",
        "definition": "Predicted demand divided by effective simulated capacity.",
        "formula": "predicted demand / effective capacity * 100",
        "unit": "percent",
        "source": "deterministic_simulation",
        "valid_range": "0 or greater; may exceed 100%",
        "unavailable_when": "effective capacity is zero or a required input is missing",
    },
    "allocated_budget_connections": {
        "label_en": "Allocated simulated budget",
        "label_es": "Presupuesto simulado asignado",
        "definition": "Total additional connection-count capacity assigned by the deterministic planner.",
        "formula": "sum(per-zone allocated budget)",
        "unit": "connections",
        "source": "deterministic_simulation",
        "valid_range": "0 to declared total budget",
        "unavailable_when": "budget assumptions are invalid",
    },
    "unallocated_budget_connections": {
        "label_en": "Unallocated simulated budget",
        "label_es": "Presupuesto simulado sin asignar",
        "definition": "Declared additional budget that the planner did not assign.",
        "formula": "declared budget - allocated budget",
        "unit": "connections",
        "source": "deterministic_simulation",
        "valid_range": "0 to declared total budget",
        "unavailable_when": "budget assumptions are invalid",
    },
    "difference_from_baseline_connections": {
        "label_en": "Difference from baseline",
        "label_es": "Diferencia frente a la línea base",
        "definition": "Alternative metric minus the same baseline metric.",
        "formula": "alternative - baseline",
        "unit": "connections when applied to count metrics",
        "source": "deterministic_comparison",
        "valid_range": "finite signed value",
        "unavailable_when": "either scenario metric is unavailable",
    },
    "difference_from_baseline_pct": {
        "label_en": "Relative difference from baseline",
        "label_es": "Diferencia relativa frente a la línea base",
        "definition": "Signed difference relative to a non-zero baseline value.",
        "formula": "(alternative - baseline) / abs(baseline) * 100",
        "unit": "percent",
        "source": "deterministic_comparison",
        "valid_range": "finite signed value",
        "unavailable_when": "baseline is zero or either value is unavailable",
    },
}


def metric_definitions() -> dict:
    """Return a JSON-ready copy of definitions for APIs and persisted reports."""
    return {
        "registry_version": METRIC_REGISTRY_VERSION,
        "metrics": {key: dict(value) for key, value in METRIC_REGISTRY.items()},
    }
