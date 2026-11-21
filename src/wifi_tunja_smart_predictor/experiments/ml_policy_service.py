"""Reusable policy persistence and evaluation services for synthetic ML runs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from wifi_tunja_smart_predictor.config import PROJECT_VERSION
from wifi_tunja_smart_predictor.experiments.ml_gates import evaluate_experiment_gates
from wifi_tunja_smart_predictor.experiments.ml_policy_audit import (
    AUDIT_SCHEMA_VERSION,
    audit_digest,
    build_policy_audit_event,
)
from wifi_tunja_smart_predictor.experiments.ml_policy_run_comparison import (
    compare_policy_evaluation_run_records,
)
from wifi_tunja_smart_predictor.experiments.ml_store import MLExperimentStore


class MLEvaluationPolicyNotFoundError(LookupError):
    """A requested policy version or experiment was not found."""


class MLEvaluationPolicyService:
    """Create immutable policy versions and apply them to saved experiments."""

    def __init__(self, store: MLExperimentStore) -> None:
        self.store = store

    def create_version(
        self,
        *,
        policy_id: str | None,
        name: str,
        description: str,
        tags: list[str],
        criteria: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        try:
            return self.store.create_policy_version(
                policy_id=policy_id,
                name=name,
                description=description,
                tags=tags,
                criteria=criteria,
            )
        except KeyError as exc:
            raise MLEvaluationPolicyNotFoundError(str(exc)) from exc

    def get_policy(self, policy_id: str, version: int | None = None) -> dict[str, Any]:
        policy = self.store.get_policy(policy_id, version)
        if policy is None:
            raise MLEvaluationPolicyNotFoundError("Evaluation policy version was not found.")
        return policy

    def list_policies(self, *, limit: int, offset: int) -> tuple[list[dict], int]:
        return self.store.list_policies(limit=limit, offset=offset)

    def list_versions(self, policy_id: str, *, limit: int, offset: int) -> tuple[list[dict], int]:
        return self.store.list_policy_versions(policy_id, limit=limit, offset=offset)

    def evaluate(
        self,
        policy_id: str,
        version: int,
        experiment_id: str,
        *,
        execution_source: str = "api",
    ) -> dict[str, Any]:
        policy = self.store.get_policy(policy_id, version)
        record = self.store.get(experiment_id)
        if policy is None or record is None:
            raise MLEvaluationPolicyNotFoundError(
                "Evaluation policy version or experiment was not found."
            )
        configuration = record.get("configuration")
        configuration = configuration if isinstance(configuration, dict) else {}
        task = configuration.get("task")
        task = task if isinstance(task, str) and task in {"classification", "regression"} else None
        configured_criteria = policy.get("criteria", {})
        criteria = configured_criteria.get(task) if isinstance(configured_criteria, dict) else None
        if not criteria:
            run = {
                "status": "INCONCLUSIVE",
                "explanation": "The selected policy has no criteria for the experiment task.",
                "reasons": ["POLICY_TASK_UNSUPPORTED"],
                "metrics": {},
                "criteria": [],
            }
            compatibility = {"status": "INCOMPATIBLE", "reasons": ["POLICY_TASK_UNSUPPORTED"]}
        else:
            # A self-pair reuses V8.3's full single-record provenance and leakage checks.
            evaluated = evaluate_experiment_gates([record, record], task, criteria)
            run = evaluated["results"][0]
            pair = evaluated["compatibility"]["comparisons"][0]
            compatibility = {
                "status": pair["status"],
                "reasons": pair["reasons"],
                "reason_details": pair["reason_details"],
            }
            missing = [
                f"MISSING_REQUIRED_METRIC_{item['metric'].upper()}"
                for item in run["criteria"]
                if item["status"] == "INCONCLUSIVE" and item.get("value") is None
            ]
            run["reasons"] = sorted(set(run["reasons"] + missing))
        result = record.get("result") if isinstance(record.get("result"), dict) else {}
        dataset = result.get("dataset") if isinstance(result.get("dataset"), dict) else {}
        split = configuration.get("evaluation_split")
        temporal = (
            result.get("temporal_split") if isinstance(result.get("temporal_split"), dict) else {}
        )
        evaluation_window = temporal.get(split) if isinstance(temporal, dict) else None
        timestamp = datetime.now(UTC).isoformat()
        report = {
            "report_schema_version": "1.0",
            "run_id": str(uuid4()),
            "created_at_utc": timestamp,
            "generated_at_utc": timestamp,
            "project_version": PROJECT_VERSION,
            "data_boundary": "Synthetic ML experiment only; official MinTIC records are excluded.",
            "policy": policy,
            "experiment": {
                "experiment_id": record.get("experiment_id"),
                "task": task,
                "status": record.get("status"),
                "created_at_utc": record.get("created_at_utc"),
                "completed_at_utc": record.get("completed_at_utc"),
                "evaluation_split": split,
                "evaluation_window": evaluation_window,
                "dataset": {
                    "kind": dataset.get("kind"),
                    "fingerprint_algorithm": dataset.get("fingerprint_algorithm"),
                    "fingerprint": dataset.get("fingerprint"),
                },
            },
            "status": run["status"],
            "explanation": run["explanation"],
            "reasons": run["reasons"],
            "metrics": run["metrics"],
            "criteria": run["criteria"],
            "compatibility": compatibility,
            "limitations": [
                "This result evaluates recorded synthetic metrics only.",
                "A passed policy is not a universal model ranking or real-world validation.",
                "The report is read-only and does not promote models or modify artifacts.",
            ],
        }
        audit_event = build_policy_audit_event(report, execution_source=execution_source)
        self.store.save_policy_evaluation_run(report, audit_event)
        return report

    @staticmethod
    def _audit_verification(audit: dict[str, Any] | None, run_id: str | None) -> dict[str, Any]:
        if audit is None:
            return {
                "status": "UNVERIFIABLE",
                "run_id": run_id,
                "reasons": ["AUDIT_EVENT_MISSING"],
                "explanation": "No audit event is stored for this evaluation run.",
            }
        event = audit.get("event")
        if audit.get("decode_status") != "READABLE" or not isinstance(event, dict):
            return {
                "status": "UNVERIFIABLE",
                "run_id": audit.get("run_id"),
                "audit_id": audit.get("audit_id"),
                "reasons": ["AUDIT_EVENT_UNREADABLE"],
                "explanation": "The stored audit event cannot be parsed as a JSON object.",
            }
        try:
            computed = audit_digest(event)
        except (TypeError, ValueError):
            return {
                "status": "INVALID",
                "run_id": audit.get("run_id"),
                "audit_id": audit.get("audit_id"),
                "reasons": ["AUDIT_EVENT_NOT_CANONICALIZABLE"],
                "explanation": "The event contains values outside the canonical JSON format.",
            }
        if computed != audit.get("digest_sha256"):
            return {
                "status": "INVALID",
                "run_id": audit.get("run_id"),
                "audit_id": audit.get("audit_id"),
                "stored_digest_sha256": audit.get("digest_sha256"),
                "computed_digest_sha256": computed,
                "reasons": ["DIGEST_MISMATCH"],
                "explanation": "The stored audit event does not match its local digest.",
            }
        if event.get("audit_schema_version") != AUDIT_SCHEMA_VERSION:
            return {
                "status": "UNVERIFIABLE",
                "run_id": audit.get("run_id"),
                "audit_id": audit.get("audit_id"),
                "reasons": ["UNSUPPORTED_AUDIT_SCHEMA"],
                "explanation": "This application cannot verify the audit event schema version.",
            }
        reasons = []
        if audit.get("audit_id") != event.get("audit_id"):
            reasons.append("AUDIT_ID_MISMATCH")
        for event_key, audit_key in (
            ("run_id", "run_id"),
            ("timestamp_utc", "timestamp_utc"),
            ("execution_source", "execution_source"),
            ("outcome", "outcome"),
        ):
            if event.get(event_key) != audit.get(audit_key):
                reasons.append(f"INDEX_{event_key.upper()}_MISMATCH")
        return {
            "status": "INVALID" if reasons else "VERIFIED",
            "run_id": audit.get("run_id"),
            "audit_id": audit.get("audit_id"),
            "stored_digest_sha256": audit.get("digest_sha256"),
            "computed_digest_sha256": computed,
            "reasons": sorted(set(reasons)),
            "explanation": (
                "The local digest and indexed event fields are consistent."
                if not reasons
                else "The audit record has a digest or indexed-field inconsistency."
            ),
        }

    def verify_audit(self, run_id: str) -> dict[str, Any]:
        """Verify one audit event and its consistency with the linked evaluation report."""
        audit = self.store.get_policy_audit(run_id)
        result = self._audit_verification(audit, run_id)
        if audit is None or result["status"] == "UNVERIFIABLE":
            return result
        try:
            run = self.store.get_policy_evaluation_run(run_id)
        except (ValueError, TypeError):
            result["status"] = "INVALID"
            result["reasons"].append("LINKED_RUN_REPORT_UNREADABLE")
            result["explanation"] = "The linked evaluation report is malformed or unreadable."
            return result
        if run is None:
            result["status"] = "INVALID"
            result["reasons"].append("LINKED_RUN_MISSING")
        else:
            report = run.get("report", {})
            policy = report.get("policy", {}) if isinstance(report, dict) else {}
            experiment = report.get("experiment", {}) if isinstance(report, dict) else {}
            expected = {
                "run_id": run.get("run_id"),
                "experiment_id": run.get("experiment_id"),
                "policy_id": run.get("policy_id"),
                "policy_version": run.get("policy_version"),
                "timestamp_utc": run.get("created_at_utc"),
                "outcome": run.get("status"),
            }
            event = audit.get("event", {})
            actual = {
                "run_id": event.get("run_id"),
                "experiment_id": event.get("experiment_id"),
                "policy_id": event.get("policy_id"),
                "policy_version": event.get("policy_version"),
                "timestamp_utc": event.get("timestamp_utc"),
                "outcome": event.get("outcome"),
            }
            if (
                policy.get("policy_id") != expected["policy_id"]
                or policy.get("version") != expected["policy_version"]
                or experiment.get("experiment_id") != expected["experiment_id"]
            ):
                result["reasons"].append("RUN_REPORT_INDEX_INCONSISTENT")
            if actual != expected:
                result["reasons"].append("AUDIT_RUN_REFERENCE_MISMATCH")
            if isinstance(report, dict) and event.get("outcome") != report.get("status"):
                result["reasons"].append("AUDIT_OUTCOME_REPORT_MISMATCH")
            if not isinstance(report, dict):
                result["reasons"].append("LINKED_RUN_REPORT_MALFORMED")
            else:
                report_experiment = (
                    report.get("experiment") if isinstance(report.get("experiment"), dict) else {}
                )
                report_dataset = (
                    report_experiment.get("dataset")
                    if isinstance(report_experiment.get("dataset"), dict)
                    else {}
                )
                expected_provenance = {
                    "project_version": report.get("project_version"),
                    "task": report_experiment.get("task"),
                    "dataset_kind": report_dataset.get("kind"),
                    "dataset_fingerprint_algorithm": report_dataset.get("fingerprint_algorithm"),
                    "dataset_fingerprint": report_dataset.get("fingerprint"),
                    "evaluation_split": report_experiment.get("evaluation_split"),
                    "evaluation_window": report_experiment.get("evaluation_window"),
                }
                if event.get("provenance") != expected_provenance:
                    result["reasons"].append("AUDIT_PROVENANCE_REPORT_MISMATCH")
        result["reasons"] = sorted(set(result["reasons"]))
        if result["reasons"]:
            result["status"] = "INVALID"
            result["explanation"] = (
                "The audit event, its index, or linked run data is inconsistent."
            )
        else:
            result["status"] = "VERIFIED"
        return result

    def verify_audit_id(self, audit_id: str) -> dict[str, Any]:
        """Verify an audit record by ID, including whether its linked run still exists."""
        audit = self.store.get_policy_audit_by_id(audit_id)
        if audit is None:
            return {
                "status": "UNVERIFIABLE",
                "audit_id": audit_id,
                "reasons": ["AUDIT_EVENT_MISSING"],
                "explanation": "No audit event is stored with this identifier.",
            }
        result = self.verify_audit(audit["run_id"])
        if audit["audit_id"] != audit_id:
            result["status"] = "INVALID"
            result["reasons"] = sorted(set(result["reasons"] + ["AUDIT_ID_MISMATCH"]))
        return result

    def list_audits(self, *, limit: int, offset: int, **filters: Any) -> tuple[list, int]:
        start = self._normalize_utc(filters.pop("created_from_utc", None))
        end = self._normalize_utc(filters.pop("created_to_utc", None))
        if start is not None and end is not None and start > end:
            raise ValueError("created_from_utc must not be later than created_to_utc.")
        audits, total = self.store.list_policy_audits(
            limit=limit,
            offset=offset,
            created_from_utc=start,
            created_to_utc=end,
            **filters,
        )
        items = []
        for audit in audits:
            verification = self.verify_audit(audit["run_id"])
            items.append({**audit, "verification": verification})
        return items, total

    @staticmethod
    def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
        report = run["report"]
        return {
            "run_id": run["run_id"],
            "created_at_utc": run["created_at_utc"],
            "experiment_id": run["experiment_id"],
            "policy_id": run["policy_id"],
            "policy_version": run["policy_version"],
            "status": run["status"],
            "criteria": report.get("criteria", []),
            "provenance": {
                "policy": report.get("policy"),
                "experiment": report.get("experiment"),
            },
        }

    @staticmethod
    def _normalize_utc(value: datetime | None) -> str | None:
        if value is None:
            return None
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat()

    def list_runs(
        self,
        *,
        limit: int,
        offset: int,
        policy_id: str | None = None,
        policy_version: int | None = None,
        experiment_id: str | None = None,
        status: str | None = None,
        created_from_utc: datetime | None = None,
        created_to_utc: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        start = self._normalize_utc(created_from_utc)
        end = self._normalize_utc(created_to_utc)
        if start is not None and end is not None and start > end:
            raise ValueError("created_from_utc must not be later than created_to_utc.")
        runs, total = self.store.list_policy_evaluation_runs(
            limit=limit,
            offset=offset,
            policy_id=policy_id,
            policy_version=policy_version,
            experiment_id=experiment_id,
            status=status,
            created_from_utc=start,
            created_to_utc=end,
        )
        return [self._run_summary(run) for run in runs], total

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        run = self.store.get_policy_evaluation_run(run_id)
        if run is None:
            return None
        return {**self._run_summary(run), "report": run["report"]}

    def get_run_report(self, run_id: str) -> dict[str, Any] | None:
        run = self.store.get_policy_evaluation_run(run_id)
        return run["report"] if run else None

    def compare_runs(self, left_run_id: str, right_run_id: str) -> dict[str, Any]:
        left = self.store.get_policy_evaluation_run(left_run_id)
        right = self.store.get_policy_evaluation_run(right_run_id)
        if left is None or right is None:
            raise MLEvaluationPolicyNotFoundError(
                "One or more policy evaluation runs were not found."
            )
        return compare_policy_evaluation_run_records(left, right)
