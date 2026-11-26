"""Opt-in local history for offline audit export verification results."""

from __future__ import annotations

import csv
import io
import json
import re
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HISTORY_SCHEMA_VERSION = 1
HISTORY_RECORD_FIELDS = {
    "verification_id",
    "timestamp_utc",
    "export_filename",
    "export_format",
    "manifest_schema_version",
    "record_count",
    "digest_sha256",
    "result",
}
_RESULTS = {"VALID", "INVALID", "UNSUPPORTED"}
_RESULT_ORDER = ("VALID", "INVALID", "UNSUPPORTED")
_FORMATS = {"json", "csv", "unsupported"}
SUMMARY_CSV_FIELDS = (
    "row_type",
    "history_schema_version",
    "filter_result",
    "filter_from_utc",
    "filter_to_utc",
    "total_records",
    "result",
    "result_count",
    "export_format",
    "format_count",
    "missing_format_count",
    "date_range_from_utc",
    "date_range_to_utc",
)
_SCHEMA_VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,39}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COLUMNS = (
    "verification_id",
    "timestamp_utc",
    "export_filename",
    "export_format",
    "manifest_schema_version",
    "record_count",
    "digest_sha256",
    "result",
)


class VerificationHistoryError(ValueError):
    """Raised when the local verification history cannot be read or updated safely."""


def _timestamp(value: str | None, field: str = "timestamp") -> str:
    raw = value or datetime.now(UTC).isoformat()
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise VerificationHistoryError(f"{field} must be an ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None:
        raise VerificationHistoryError(f"{field} must include a timezone.")
    return parsed.astimezone(UTC).isoformat()


def build_verification_history_record(
    export_path: str | Path,
    manifest_path: str | Path,
    result: dict[str, Any],
    *,
    timestamp_utc: str | None = None,
) -> dict[str, Any]:
    """Select essential metadata only; never copy records from either evidence file."""
    manifest: dict[str, Any] = {}
    try:
        candidate = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if isinstance(candidate, dict):
            manifest = candidate
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        pass

    export_file = Path(export_path)
    export_format = manifest.get("export_format")
    if export_format not in {"json", "csv"}:
        export_format = (
            export_file.suffix.lower().lstrip(".")
            if export_file.suffix.lower() in {".json", ".csv"}
            else "unsupported"
        )
    record_count = result.get("record_count", manifest.get("record_count"))
    if not isinstance(record_count, int) or isinstance(record_count, bool) or record_count < 0:
        record_count = None
    digest = manifest.get("records_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        digest = result.get("manifest_sha256") or result.get("computed_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        digest = None
    schema_version = manifest.get("manifest_schema_version")
    if not isinstance(schema_version, str) or not _SCHEMA_VERSION_RE.fullmatch(schema_version):
        schema_version = "unknown"

    record = {
        "verification_id": uuid.uuid4().hex,
        "timestamp_utc": _timestamp(timestamp_utc),
        "export_filename": export_file.name,
        "export_format": export_format,
        "manifest_schema_version": schema_version,
        "record_count": record_count,
        "digest_sha256": digest,
        "result": result.get("status"),
    }
    _validate_record(record)
    return record


def _validate_record(record: dict[str, Any]) -> None:
    if set(record) != HISTORY_RECORD_FIELDS:
        raise VerificationHistoryError("Verification history record fields are invalid.")
    if record["result"] not in _RESULTS:
        raise VerificationHistoryError(
            "Verification result must be VALID, INVALID, or UNSUPPORTED."
        )
    if not isinstance(record["verification_id"], str) or not record["verification_id"]:
        raise VerificationHistoryError("verification_id must be a non-empty string.")
    _timestamp(record["timestamp_utc"])
    for key in ("export_filename",):
        if (
            not isinstance(record[key], str)
            or not record[key]
            or Path(record[key]).name != record[key]
        ):
            raise VerificationHistoryError(f"{key} must contain only a filename.")
    for key in ("export_format", "manifest_schema_version", "digest_sha256"):
        if record[key] is not None and not isinstance(record[key], str):
            raise VerificationHistoryError(f"{key} must be text or null.")
    if record["export_format"] is not None and record["export_format"] not in _FORMATS:
        raise VerificationHistoryError("export_format is not supported for the history catalog.")
    schema_version = record["manifest_schema_version"]
    if schema_version is not None and not _SCHEMA_VERSION_RE.fullmatch(schema_version):
        raise VerificationHistoryError("manifest_schema_version is not a safe version label.")
    digest = record["digest_sha256"]
    if digest is not None and not _SHA256_RE.fullmatch(digest):
        raise VerificationHistoryError("digest_sha256 must be a lowercase SHA-256 value.")
    count = record["record_count"]
    if count is not None and (not isinstance(count, int) or isinstance(count, bool) or count < 0):
        raise VerificationHistoryError("record_count must be a non-negative integer or null.")


def _open_store(path: str | Path, *, create: bool) -> sqlite3.Connection:
    store = Path(path)
    if not create and not store.is_file():
        raise VerificationHistoryError(f"Verification history store does not exist: {store}")
    if create:
        store.parent.mkdir(parents=True, exist_ok=True)
    try:
        if create:
            connection = sqlite3.connect(str(store))
        else:
            connection = sqlite3.connect(f"{store.resolve().as_uri()}?mode=ro", uri=True)
        check = connection.execute("PRAGMA quick_check").fetchone()
        if not check or check[0] != "ok":
            raise VerificationHistoryError("Verification history database is corrupted.")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if version == 0 and not tables and create:
            connection.execute("""CREATE TABLE verification_history (
                    verification_id TEXT PRIMARY KEY,
                    timestamp_utc TEXT NOT NULL,
                    export_filename TEXT NOT NULL,
                    export_format TEXT,
                    manifest_schema_version TEXT,
                    record_count INTEGER,
                    digest_sha256 TEXT,
                    result TEXT NOT NULL CHECK(result IN ('VALID', 'INVALID', 'UNSUPPORTED'))
                )""")
            connection.execute(
                "CREATE INDEX idx_verification_history_time ON verification_history(timestamp_utc, verification_id)"
            )
            connection.execute(f"PRAGMA user_version = {HISTORY_SCHEMA_VERSION}")
            connection.commit()
            tables = {"verification_history"}
            version = HISTORY_SCHEMA_VERSION
        if version != HISTORY_SCHEMA_VERSION or "verification_history" not in tables:
            raise VerificationHistoryError(
                f"Unsupported or incomplete verification history store (schema {version})."
            )
        actual = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(verification_history)")
        )
        if actual != _COLUMNS:
            raise VerificationHistoryError("Verification history table structure is unsupported.")
        return connection
    except VerificationHistoryError:
        if "connection" in locals():
            connection.close()
        raise
    except (sqlite3.DatabaseError, OSError) as exc:
        if "connection" in locals():
            connection.close()
        raise VerificationHistoryError(f"Cannot open verification history store: {exc}") from exc


def save_verification_history(path: str | Path, record: dict[str, Any]) -> None:
    """Persist only the validated metadata allowlist in a local SQLite database."""
    _validate_record(record)
    connection = _open_store(path, create=True)
    try:
        connection.execute(
            "INSERT INTO verification_history VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(record[field] for field in _COLUMNS),
        )
        connection.commit()
    except sqlite3.DatabaseError as exc:
        connection.rollback()
        raise VerificationHistoryError(f"Cannot save verification history: {exc}") from exc
    finally:
        connection.close()


def _date_filter(value: str | None, field: str) -> str | None:
    return _timestamp(value, field) if value is not None else None


def list_verification_history(
    path: str | Path,
    *,
    result: str | None = None,
    from_utc: str | None = None,
    to_utc: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Return a stable page and deterministic JSON-ready report of saved metadata."""
    if result is not None and result not in _RESULTS:
        raise VerificationHistoryError("result filter must be VALID, INVALID, or UNSUPPORTED.")
    if not 1 <= limit <= 500 or offset < 0:
        raise VerificationHistoryError("limit must be 1..500 and offset must be non-negative.")
    start = _date_filter(from_utc, "from_utc")
    end = _date_filter(to_utc, "to_utc")
    if start and end and start > end:
        raise VerificationHistoryError("from_utc must not be later than to_utc.")
    where: list[str] = []
    params: list[Any] = []
    if result:
        where.append("result = ?")
        params.append(result)
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    connection = _open_store(path, create=False)
    try:
        total = connection.execute(
            f"SELECT COUNT(*) FROM verification_history{clause}", params
        ).fetchone()[0]
        rows = connection.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM verification_history{clause} "
            "ORDER BY timestamp_utc DESC, verification_id ASC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise VerificationHistoryError(f"Cannot list verification history: {exc}") from exc
    finally:
        connection.close()
    filters = {"result": result, "from_utc": start, "to_utc": end}
    return {
        "history_schema_version": HISTORY_SCHEMA_VERSION,
        "filters": filters,
        "pagination": {"limit": limit, "offset": offset, "total": total},
        "records": [dict(zip(_COLUMNS, row, strict=True)) for row in rows],
    }


def summarize_verification_history(
    path: str | Path,
    *,
    result: str | None = None,
    from_utc: str | None = None,
    to_utc: str | None = None,
) -> dict[str, Any]:
    """Aggregate metadata for all matching rows without returning individual records."""
    if result is not None and result not in _RESULTS:
        raise VerificationHistoryError("result filter must be VALID, INVALID, or UNSUPPORTED.")
    start = _date_filter(from_utc, "from_utc")
    end = _date_filter(to_utc, "to_utc")
    if start and end and start > end:
        raise VerificationHistoryError("from_utc must not be later than to_utc.")
    where: list[str] = []
    params: list[Any] = []
    if result:
        where.append("result = ?")
        params.append(result)
    if start:
        where.append("timestamp_utc >= ?")
        params.append(start)
    if end:
        where.append("timestamp_utc <= ?")
        params.append(end)
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    connection = _open_store(path, create=False)
    try:
        total, earliest, latest, missing_format = connection.execute(
            "SELECT COUNT(*), MIN(timestamp_utc), MAX(timestamp_utc), "
            f"SUM(CASE WHEN export_format IS NULL THEN 1 ELSE 0 END) FROM verification_history{clause}",
            params,
        ).fetchone()
        result_rows = connection.execute(
            f"SELECT result, COUNT(*) FROM verification_history{clause} GROUP BY result",
            params,
        ).fetchall()
        format_rows = connection.execute(
            (
                f"SELECT export_format, COUNT(*) FROM verification_history{clause} "
                "AND export_format IS NOT NULL GROUP BY export_format ORDER BY export_format"
                if clause
                else "SELECT export_format, COUNT(*) FROM verification_history "
                "WHERE export_format IS NOT NULL GROUP BY export_format ORDER BY export_format"
            ),
            params,
        ).fetchall()
    except sqlite3.DatabaseError as exc:
        raise VerificationHistoryError(f"Cannot summarize verification history: {exc}") from exc
    finally:
        connection.close()
    result_counts = {status: 0 for status in sorted(_RESULTS)}
    result_counts.update({status: count for status, count in result_rows})
    return {
        "history_schema_version": HISTORY_SCHEMA_VERSION,
        "filters": {"result": result, "from_utc": start, "to_utc": end},
        "total_records": total,
        "result_counts": result_counts,
        "format_counts": {export_format: count for export_format, count in format_rows},
        "missing_format_count": missing_format or 0,
        "date_range_utc": {"from": earliest, "to": latest} if total else None,
    }


def inspect_verification_history(path: str | Path, verification_id: str) -> dict[str, Any]:
    """Retrieve one metadata-only verification result by ID."""
    connection = _open_store(path, create=False)
    try:
        row = connection.execute(
            f"SELECT {', '.join(_COLUMNS)} FROM verification_history WHERE verification_id = ?",
            (verification_id,),
        ).fetchone()
    except sqlite3.DatabaseError as exc:
        raise VerificationHistoryError(f"Cannot read verification history: {exc}") from exc
    finally:
        connection.close()
    if row is None:
        raise VerificationHistoryError(f"Verification record not found: {verification_id}")
    return {
        "history_schema_version": HISTORY_SCHEMA_VERSION,
        "record": dict(zip(_COLUMNS, row, strict=True)),
    }


def render_history_json(report: dict[str, Any]) -> str:
    """Serialize a report deterministically, independent of dictionary insertion order."""
    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def render_history_csv(report: dict[str, Any]) -> str:
    """Render deterministic metadata-only summary rows using stable columns and LF endings."""
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=SUMMARY_CSV_FIELDS, lineterminator="\n", extrasaction="raise"
    )
    writer.writeheader()
    filters = report["filters"]
    date_range = report["date_range_utc"] or {}
    shared = {
        "history_schema_version": report["history_schema_version"],
        "filter_result": filters["result"] or "",
        "filter_from_utc": filters["from_utc"] or "",
        "filter_to_utc": filters["to_utc"] or "",
        "total_records": report["total_records"],
        "missing_format_count": report["missing_format_count"],
        "date_range_from_utc": date_range.get("from") or "",
        "date_range_to_utc": date_range.get("to") or "",
    }
    writer.writerow({**shared, "row_type": "summary"})
    for status in _RESULT_ORDER:
        writer.writerow(
            {
                "row_type": "result",
                "result": status,
                "result_count": report["result_counts"][status],
            }
        )
    for export_format, count in sorted(report["format_counts"].items()):
        writer.writerow(
            {"row_type": "format", "export_format": export_format, "format_count": count}
        )
    return output.getvalue()


def history_cli_main(argv: list[str]) -> int:
    """CLI for listing and inspecting local verification history."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(prog="verify_audit_export.py history")
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List saved verification results")
    listing.add_argument("--store", required=True, type=Path)
    listing.add_argument("--result", choices=sorted(_RESULTS))
    listing.add_argument("--from-utc")
    listing.add_argument("--to-utc")
    listing.add_argument("--limit", type=int, default=50)
    listing.add_argument("--offset", type=int, default=0)
    summary = commands.add_parser("summary", help="Summarize matching verification results")
    summary.add_argument("--store", required=True, type=Path)
    summary.add_argument("--result", choices=sorted(_RESULTS))
    summary.add_argument("--from-utc")
    summary.add_argument("--to-utc")
    summary.add_argument("--format", choices=("json", "csv"), default="json")
    summary.add_argument("--output", type=Path, help="Write the summary to a UTF-8 file")
    detail = commands.add_parser("show", help="Inspect one saved verification result")
    detail.add_argument("verification_id")
    detail.add_argument("--store", required=True, type=Path)
    try:
        args = parser.parse_args(argv)
        if args.command == "list":
            report = list_verification_history(
                args.store,
                result=args.result,
                from_utc=args.from_utc,
                to_utc=args.to_utc,
                limit=args.limit,
                offset=args.offset,
            )
        elif args.command == "summary":
            report = summarize_verification_history(
                args.store,
                result=args.result,
                from_utc=args.from_utc,
                to_utc=args.to_utc,
            )
        else:
            report = inspect_verification_history(args.store, args.verification_id)
        output = (
            render_history_csv(report)
            if args.command == "summary" and args.format == "csv"
            else render_history_json(report)
        )
        if args.command == "summary" and args.output is not None:
            args.output.write_text(output, encoding="utf-8", newline="")
            print(f"Summary written to: {args.output}")
        elif args.command == "summary" and args.format == "csv":
            sys.stdout.buffer.write(output.encode("utf-8"))
        else:
            print(output, end="")
        return 0
    except VerificationHistoryError as exc:
        print(f"History error: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"History output error: {exc}", file=sys.stderr)
        return 3
