"""Transactional SQLite storage for bounded scenario experiment records."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, date, datetime
from math import isfinite
from pathlib import Path
from typing import Any

from wifi_tunja_smart_predictor.experiments.integrity import inspect_experiment_integrity

EXPERIMENT_SCHEMA_VERSION = "1.2"


class ExperimentStore:
    """Small local experiment store with explicit schema and safe serialization."""

    def __init__(self, path: Path, *, readonly: bool = False) -> None:
        self.path = Path(path)
        self.readonly = readonly
        if readonly:
            if not self.path.is_file():
                raise FileNotFoundError("Experiment history is not initialized.")
            with self._connect() as connection:
                row = connection.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
                if row is None or row[0] != EXPERIMENT_SCHEMA_VERSION:
                    raise ValueError("Experiment history has an unsupported schema.")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS schema_version (version TEXT NOT NULL)")
            row = connection.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
            if row is None:
                existing = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experiments'"
                ).fetchone()
                if existing:
                    raise ValueError("Experiment database has no recorded schema version.")
                connection.execute(
                    "INSERT INTO schema_version(version) VALUES (?)", (EXPERIMENT_SCHEMA_VERSION,)
                )
            elif row[0] in {"1.0", "1.1"}:
                existing = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'experiments'"
                ).fetchone()
                if existing:
                    columns = {
                        item["name"]
                        for item in connection.execute("PRAGMA table_info(experiments)")
                    }
                    if "updated_at_utc" not in columns:
                        connection.execute("ALTER TABLE experiments ADD COLUMN updated_at_utc TEXT")
                    if "status" not in columns:
                        connection.execute("ALTER TABLE experiments ADD COLUMN status TEXT")
                    connection.execute(
                        "UPDATE experiments SET updated_at_utc = created_at_utc WHERE updated_at_utc IS NULL"
                    )
                    connection.execute(
                        "UPDATE experiments SET status = 'COMPLETED' WHERE status IS NULL"
                    )
                    connection.execute(
                        "UPDATE experiments SET schema_version = ?", (EXPERIMENT_SCHEMA_VERSION,)
                    )
                connection.execute(
                    "UPDATE schema_version SET version = ?", (EXPERIMENT_SCHEMA_VERSION,)
                )
            elif row[0] != EXPERIMENT_SCHEMA_VERSION:
                raise ValueError(f"Unsupported experiment database schema: {row[0]}")
            connection.execute("""CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY, created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL, status TEXT NOT NULL,
                    experiment_type TEXT NOT NULL, name TEXT NOT NULL,
                    schema_version TEXT NOT NULL, configuration_json TEXT NOT NULL,
                    result_json TEXT NOT NULL
                )""")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        target = f"file:{self.path.as_posix()}?mode=ro" if self.readonly else self.path
        connection = sqlite3.connect(target, timeout=10, uri=self.readonly)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def save(
        self, name: str, configuration: dict[str, Any], result: dict[str, Any]
    ) -> dict[str, Any]:
        if not name.strip() or len(name) > 120:
            raise ValueError("name must contain 1 to 120 characters")
        experiment = {
            "id": str(uuid.uuid4()),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "updated_at_utc": None,
            "status": "COMPLETED",
            "experiment_type": "capacity_scenario",
            "name": name.strip(),
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "configuration": configuration,
            "result": result,
        }
        experiment["updated_at_utc"] = experiment["created_at_utc"]
        experiment["integrity"] = inspect_experiment_integrity(experiment)
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    experiment["id"],
                    experiment["created_at_utc"],
                    experiment["updated_at_utc"],
                    experiment["status"],
                    experiment["experiment_type"],
                    experiment["name"],
                    experiment["schema_version"],
                    json.dumps(configuration, sort_keys=True, allow_nan=False),
                    json.dumps(result, sort_keys=True, allow_nan=False),
                ),
            )
        return experiment

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM experiments WHERE id = ?", (experiment_id,)
            ).fetchone()
        return self._decode(row) if row else None

    def list_page(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        search: str | None = None,
        zone_id: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]], int, dict[str, Any]]:
        """Return one filtered page and complete filtered-history analytics."""
        records: list[dict[str, Any]] = []
        warnings: list[dict[str, str]] = []
        total_count = 0
        zones_by_experiment: Counter[str] = Counter()
        created_by_utc_date: Counter[str] = Counter()
        integrity_counts: Counter[str] = Counter()
        capacities: list[float] = []
        budgets: list[float] = []
        estimates: list[float] = []
        missing_counts: Counter[str] = Counter()

        with self._connect() as connection:
            cursor = connection.execute(
                "SELECT * FROM experiments ORDER BY created_at_utc DESC, id"
            )
            for row in cursor:
                try:
                    record = self._decode(row)
                except ValueError:
                    warnings.append(
                        {
                            "experiment_id": str(row["id"]),
                            "reason_code": "INVALID_PERSISTED_RECORD",
                            "message": "This saved record failed integrity checks and was omitted.",
                        }
                    )
                    continue

                created = datetime.fromisoformat(
                    record["created_at_utc"].replace("Z", "+00:00")
                ).astimezone(UTC)
                if start_date and created.date() < start_date:
                    continue
                if end_date and created.date() > end_date:
                    continue
                if (
                    search
                    and search.casefold() not in f"{record['id']} {record['name']}".casefold()
                ):
                    continue

                configuration = record["configuration"]
                result = record["result"]
                configured_zones = configuration.get("zones")
                configured_zones = configured_zones if isinstance(configured_zones, list) else []
                zone_ids = {
                    zone["zone_id"]
                    for zone in configured_zones
                    if isinstance(zone, dict)
                    and isinstance(zone.get("zone_id"), str)
                    and bool(zone["zone_id"])
                }
                if zone_id and zone_id not in zone_ids:
                    continue

                total_count += 1
                if total_count > offset and len(records) < limit:
                    records.append(record)
                created_by_utc_date[created.date().isoformat()] += 1
                integrity_status = record["integrity"].get("status", "unknown")
                integrity_counts[integrity_status] += 1
                if integrity_status == "invalid":
                    missing_counts["invalid_records_excluded_from_field_analytics"] += 1
                    continue

                for identifier in zone_ids:
                    zones_by_experiment[identifier] += 1

                capacity_values = [
                    self._finite_number(zone.get("capacity_connections"))
                    for zone in configured_zones
                    if isinstance(zone, dict)
                ]
                valid_capacities = [value for value in capacity_values if value is not None]
                capacities.extend(valid_capacities)
                if not zone_ids:
                    missing_counts["zone_configuration"] += 1
                if not valid_capacities:
                    missing_counts["configured_capacity"] += 1
                if not capacity_values or any(value is None for value in capacity_values):
                    missing_counts["configured_capacity_values"] += (
                        sum(value is None for value in capacity_values) or 1
                    )

                budget = self._finite_number(configuration.get("resource_budget_connections"))
                if budget is None:
                    missing_counts["allocation_budget"] += 1
                else:
                    budgets.append(budget)

                result_zones = result.get("zones")
                result_zones = result_zones if isinstance(result_zones, list) else []
                valid_estimates = [
                    value
                    for zone in result_zones
                    if isinstance(zone, dict)
                    for value in [self._finite_number(zone.get("demand"))]
                    if value is not None
                ]
                estimates.extend(valid_estimates)
                if not valid_estimates:
                    missing_counts["connection_estimates"] += 1

                provenance = result.get("provenance")
                provenance = provenance if isinstance(provenance, dict) else {}
                if not isinstance(provenance.get("dataset_sha256"), str):
                    missing_counts["dataset_fingerprint"] += 1
                model_traces = provenance.get("models")
                if not isinstance(model_traces, list) or not model_traces:
                    missing_counts["model_provenance"] += 1

        analytics = {
            "total_experiment_count": total_count,
            "experiments_by_zone": [
                {"zone_id": identifier, "experiment_count": count}
                for identifier, count in sorted(zones_by_experiment.items())
            ],
            "experiments_by_creation_date_utc": [
                {"date_utc": day, "experiment_count": count}
                for day, count in sorted(created_by_utc_date.items())
            ],
            "record_integrity_counts": dict(sorted(integrity_counts.items())),
            "numeric_summaries": {
                "configured_capacity_connections": self._numeric_summary(capacities),
                "allocation_budget_connections": self._numeric_summary(budgets),
                "estimated_next_hour_connections": self._numeric_summary(estimates),
            },
            "missing_experiment_counts": dict(sorted(missing_counts.items())),
        }
        return records, warnings, total_count, analytics

    @staticmethod
    def _finite_number(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        try:
            number = float(value)
        except OverflowError:
            return None
        return number if isfinite(number) else None

    @staticmethod
    def _numeric_summary(values: list[float]) -> dict[str, Any]:
        summary: dict[str, Any] = {"observation_count": len(values), "statistics": None}
        if len(values) >= 2:
            ordered = sorted(values)
            middle = len(ordered) // 2
            middle_value = (
                ordered[middle]
                if len(ordered) % 2
                else ordered[middle - 1] / 2 + ordered[middle] / 2
            )
            summary["statistics"] = {
                "minimum": ordered[0],
                "maximum": ordered[-1],
                "mean": sum(value / len(values) for value in values),
                "median": middle_value,
            }
        return summary

    def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        """Return readable records, omitting malformed rows without hiding valid history."""
        records, _warnings, _total, _analytics = self.list_page(limit, offset)
        return records

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        try:
            configuration = json.loads(row["configuration_json"])
            result = json.loads(row["result_json"])
        except json.JSONDecodeError as exc:
            raise ValueError("Experiment record contains invalid JSON.") from exc
        if (
            row["schema_version"] != EXPERIMENT_SCHEMA_VERSION
            or row["experiment_type"] != "capacity_scenario"
            or row["status"] != "COMPLETED"
            or not isinstance(configuration, dict)
            or not isinstance(result, dict)
        ):
            raise ValueError("Experiment record is malformed or uses an unsupported schema.")
        for field in ("created_at_utc", "updated_at_utc"):
            value = row[field]
            try:
                timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError("Experiment record has an invalid UTC timestamp.") from exc
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("Experiment record has a UTC timestamp without an offset.")
        record = {
            "id": row["id"],
            "created_at_utc": row["created_at_utc"],
            "updated_at_utc": row["updated_at_utc"],
            "status": row["status"],
            "experiment_type": row["experiment_type"],
            "name": row["name"],
            "schema_version": row["schema_version"],
            "configuration": configuration,
            "result": result,
        }
        if not isinstance(record["id"], str) or not isinstance(record["name"], str):
            raise ValueError("Experiment record has invalid identifying fields.")
        try:
            record["integrity"] = inspect_experiment_integrity(record)
        except (TypeError, ValueError, OverflowError):
            record["integrity"] = {
                "status": "invalid",
                "assessment_status": "not_assessable",
                "reason_codes": ["INTEGRITY_CHECK_FAILED"],
            }
        return record
