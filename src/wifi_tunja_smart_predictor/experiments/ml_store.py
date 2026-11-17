"""Additive SQLite persistence for reproducible synthetic ML experiments."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class IdempotencyConflict(ValueError):
    """An idempotency key was reused with a different validated configuration."""


class MLExperimentStore:
    """Persist ML experiment lifecycle without changing scenario experiment records."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection, connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS ml_experiments (
                    experiment_id TEXT PRIMARY KEY,
                    idempotency_sha256 TEXT UNIQUE,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT,
                    status TEXT NOT NULL,
                    configuration_json TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT
                )""")
            columns = {row[1] for row in connection.execute("PRAGMA table_info(ml_experiments)")}
            if "completed_at_utc" not in columns:
                connection.execute("ALTER TABLE ml_experiments ADD COLUMN completed_at_utc TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "experiment_id": row["experiment_id"],
            "created_at_utc": row["created_at_utc"],
            "updated_at_utc": row["updated_at_utc"],
            "completed_at_utc": row["completed_at_utc"],
            "status": row["status"],
            "configuration": json.loads(row["configuration_json"]),
            "result": json.loads(row["result_json"]) if row["result_json"] else None,
            "error": json.loads(row["error_json"]) if row["error_json"] else None,
        }

    def create_pending(
        self, configuration: dict[str, Any], idempotency_key: str
    ) -> tuple[dict[str, Any], bool]:
        config_json = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
        key_hash = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        now = datetime.now(UTC).isoformat()
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM ml_experiments WHERE idempotency_sha256 = ?", (key_hash,)
            ).fetchone()
            if row:
                if row["configuration_json"] != config_json:
                    raise IdempotencyConflict("Idempotency key conflicts with saved configuration.")
                record = self._decode(row)
                record["replayed"] = True
                return record, False
            experiment_id = str(uuid.uuid4())
            connection.execute(
                """INSERT INTO ml_experiments (
                    experiment_id, idempotency_sha256, created_at_utc, updated_at_utc,
                    completed_at_utc, status, configuration_json, result_json, error_json
                ) VALUES (?, ?, ?, ?, NULL, 'PENDING', ?, NULL, NULL)""",
                (experiment_id, key_hash, now, now, config_json),
            )
            row = connection.execute(
                "SELECT * FROM ml_experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode(row), True

    def transition(
        self,
        experiment_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        allowed = {"PENDING": {"RUNNING"}, "RUNNING": {"COMPLETED", "FAILED"}}
        result_json = json.dumps(result, sort_keys=True, allow_nan=False) if result else None
        error_json = json.dumps(error, sort_keys=True) if error else None
        with closing(self._connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM ml_experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
            if row is None:
                raise KeyError("Experiment not found.")
            if status not in allowed.get(row["status"], set()):
                raise ValueError("Invalid experiment state transition.")
            connection.execute(
                """UPDATE ml_experiments SET status = ?, updated_at_utc = ?,
                    completed_at_utc = CASE WHEN ? IN ('COMPLETED', 'FAILED') THEN ? ELSE completed_at_utc END,
                    result_json = COALESCE(?, result_json), error_json = COALESCE(?, error_json)
                    WHERE experiment_id = ?""",
                (
                    status,
                    datetime.now(UTC).isoformat(),
                    status,
                    datetime.now(UTC).isoformat() if status in {"COMPLETED", "FAILED"} else None,
                    result_json,
                    error_json,
                    experiment_id,
                ),
            )
            updated = connection.execute(
                "SELECT * FROM ml_experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode(updated)

    def get(self, experiment_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection, connection:
            row = connection.execute(
                "SELECT * FROM ml_experiments WHERE experiment_id = ?", (experiment_id,)
            ).fetchone()
        return self._decode(row) if row else None

    def list_page(self, *, limit: int, offset: int) -> tuple[list[dict[str, Any]], int]:
        with closing(self._connect()) as connection, connection:
            total = int(connection.execute("SELECT COUNT(*) FROM ml_experiments").fetchone()[0])
            rows = connection.execute(
                "SELECT * FROM ml_experiments ORDER BY created_at_utc DESC, experiment_id LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [self._decode(row) for row in rows], total
