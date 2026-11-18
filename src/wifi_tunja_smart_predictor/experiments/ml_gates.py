"""Deterministic acceptance checks for persisted synthetic ML experiments."""

from __future__ import annotations

from math import isfinite
from typing import Any

from wifi_tunja_smart_predictor.experiments.ml_comparison import (
    METRIC_DIRECTIONS,
    compare_experiment_records,
)


def evaluate_experiment_gates(
    records: list[dict[str, Any]], task: str, criteria: list[dict[str, Any]]
) -> dict[str, Any]:
    """Evaluate requested thresholds only when selected experiments are compatible."""
    compatibility = compare_experiment_records(records)
    comparisons = compatibility.get("comparisons", [])
    incompatible = [
        pair
        for pair in comparisons
        if any(reason != "NO_SHARED_VALID_METRICS" for reason in pair.get("reasons", []))
    ]
    by_id = {record.get("experiment_id"): record for record in records}
    results = []
    for experiment_id in sorted(by_id, key=lambda value: records.index(by_id[value])):
        record = by_id[experiment_id]
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        config = record.get("configuration") if isinstance(record.get("configuration"), dict) else {}
        evaluation = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
        dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
        temporal = (
            result.get("temporal_split")
            if isinstance(result.get("temporal_split"), dict)
            else {}
        )
        selected_split = config.get("evaluation_split")
        period = temporal.get(selected_split) if isinstance(temporal, dict) else None
        provenance = {
            "dataset_kind": dataset.get("kind"),
            "dataset_fingerprint_algorithm": dataset.get("fingerprint_algorithm"),
            "dataset_fingerprint": dataset.get("fingerprint"),
            "evaluation_split": selected_split,
            "evaluation_window": period if isinstance(period, dict) else None,
            "created_at_utc": record.get("created_at_utc"),
            "completed_at_utc": record.get("completed_at_utc"),
        }
        criterion_results = []
        has_missing = False
        has_failure = False
        metrics = evaluation.get("metrics") if isinstance(evaluation.get("metrics"), dict) else {}
        for criterion in criteria:
            value = metrics.get(criterion["metric"])
            valid = (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and isfinite(float(value))
            )
            if not valid:
                has_missing = True
                criterion_results.append(
                    {
                        **criterion,
                        "value": None,
                        "status": "INCONCLUSIVE",
                        "explanation": "A finite recorded metric value is unavailable.",
                    }
                )
                continue
            passed = value >= criterion["threshold"] if criterion["operator"] == "gte" else value <= criterion["threshold"]
            has_failure = has_failure or not passed
            criterion_results.append(
                {
                    **criterion,
                    "value": float(value),
                    "status": "PASSED" if passed else "FAILED",
                    "explanation": (
                        "The recorded metric meets the configured threshold."
                        if passed
                        else "The recorded metric does not meet the configured threshold."
                    ),
                }
            )
        run_reasons = [
            reason
            for pair in incompatible
            if experiment_id in {pair.get("left_experiment_id"), pair.get("right_experiment_id")}
            for reason in pair.get("reasons", [])
        ]
        if config.get("task") != task:
            run_reasons.append("EXPERIMENT_TASK_DOES_NOT_MATCH_GATE_TASK")
        if record.get("status") != "COMPLETED":
            run_reasons.append("EXPERIMENT_NOT_COMPLETED")
        if run_reasons:
            status = "INCONCLUSIVE"
            explanation = "Compatibility or completion evidence is insufficient; thresholds were not judged."
        elif has_missing:
            status = "INCONCLUSIVE"
            explanation = "One or more required metrics are unavailable or invalid."
        elif has_failure:
            status = "FAILED"
            explanation = "At least one valid metric did not meet its configured threshold."
        else:
            status = "PASSED"
            explanation = "All configured criteria have valid evidence and meet their thresholds."
        if run_reasons:
            for item in criterion_results:
                item["status"] = "INCONCLUSIVE"
                item["explanation"] = "Not evaluated because experiment compatibility or completion failed."
        results.append(
            {
                "experiment_id": experiment_id,
                "status": status,
                "explanation": explanation,
                "reasons": sorted(set(run_reasons)),
                "metrics": metrics,
                "criteria": criterion_results,
                "provenance": provenance,
            }
        )
    return {
        "results": results,
        "compatibility": {
            "status": "INCOMPATIBLE" if incompatible else "COMPATIBLE",
            "comparisons": comparisons,
            "incompatible_pair_count": len(incompatible),
        },
    }


def supported_gate_metrics(task: str) -> dict[str, str]:
    """Return the accepted metric directions for API/UI help text."""
    return METRIC_DIRECTIONS[task]
