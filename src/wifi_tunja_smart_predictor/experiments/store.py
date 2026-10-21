"""Transactional SQLite storage for bounded scenario experiment records."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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

    def _connect(self) -> sqlite3.Connection:
        target = f"file:{self.path.as_posix()}?mode=ro" if self.readonly else self.path
        connection = sqlite3.connect(target, timeout=10, uri=self.readonly)
        connection.row_factory = sqlite3.Row
        return connection

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

    def list(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM experiments ORDER BY created_at_utc DESC, id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._decode(row) for row in rows]

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
        return {
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
