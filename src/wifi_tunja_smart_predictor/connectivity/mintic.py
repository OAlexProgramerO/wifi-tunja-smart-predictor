"""MinTIC Socrata ingestion and auditable local snapshot persistence."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

LOGGER = logging.getLogger(__name__)
SOURCE_ID = "mintic_fixed_accesses"
DATASET_ID = "hwrp-3su3"
API_URL = f"https://www.datos.gov.co/resource/{DATASET_ID}.json"
METADATA_URL = f"https://www.datos.gov.co/api/views/{DATASET_ID}.json"
TUNJA_DANE_CODE = "15001"
PAGE_SIZE = 50000
REQUIRED_FIELDS = {
    "anno",
    "trimestre",
    "proveedor",
    "codigo_dane_departamento",
    "departamento",
    "codigo_dane_municipio",
    "municipio",
    "segmento",
    "tecnologia",
    "velocidad_bajada",
    "velocidad_subida",
    "accesos_fijos_internet",
}


class SourceUnavailableError(RuntimeError):
    """Raised when online refresh fails and no valid cached snapshot is available."""


def _request_json(url: str, params: dict[str, Any] | None = None) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "wifi-tunja-smart-predictor/0.7.0"}
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    if app_token:
        headers["X-App-Token"] = app_token
    for attempt in range(4):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=(5, 30))
            if response.status_code == 429 or response.status_code >= 500:
                if attempt < 3:
                    time.sleep(min(2**attempt, 8))
                    continue
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError, ValueError):
            if attempt == 3:
                raise
            time.sleep(min(2**attempt, 8))
    raise SourceUnavailableError("Official source request failed after retries.")


def _metadata() -> dict[str, Any]:
    payload = _request_json(METADATA_URL)
    if not isinstance(payload, dict) or payload.get("id") != DATASET_ID:
        raise ValueError("Official source metadata did not match the expected dataset.")
    columns = {column.get("fieldName") for column in payload.get("columns", [])}
    if not REQUIRED_FIELDS <= columns:
        raise ValueError("Official source schema is missing expected fields.")
    rows_updated_at = payload.get("rowsUpdatedAt")
    source_updated = (
        datetime.fromtimestamp(int(rows_updated_at), UTC).isoformat()
        if rows_updated_at is not None
        else None
    )
    return {"source_updated_at_utc": source_updated, "license": payload.get("licenseId")}


def _fetch_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    offset = 0
    while True:
        page = _request_json(
            API_URL,
            {
                "$limit": PAGE_SIZE,
                "$offset": offset,
                "$order": "anno,trimestre,proveedor,segmento,tecnologia,velocidad_bajada,velocidad_subida,accesos_fijos_internet",
                "$where": f"codigo_dane_municipio = '{TUNJA_DANE_CODE}'",
            },
        )
        if not isinstance(page, list) or any(not isinstance(row, dict) for row in page):
            raise ValueError("Official source response was not a list of JSON records.")
        if any(not REQUIRED_FIELDS <= row.keys() for row in page):
            raise ValueError("Official source data response is missing expected fields.")
        records.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += len(page)
    return records


def _normalize(row: dict[str, Any]) -> dict[str, Any]:
    def numeric(field: str, *, integral: bool = False) -> int | float:
        value = row.get(field)
        if value in (None, ""):
            raise ValueError(f"Missing required numeric field: {field}")
        try:
            parsed = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(f"Invalid numeric field: {field}") from exc
        if (
            not parsed.is_finite()
            or parsed < 0
            or (integral and parsed != parsed.to_integral_value())
        ):
            raise ValueError(f"Invalid numeric value: {field}")
        return int(parsed) if parsed == parsed.to_integral_value() else float(parsed)

    def integer(field: str) -> int:
        return int(numeric(field, integral=True))

    municipality_code = str(row.get("codigo_dane_municipio", "")).zfill(5)
    if municipality_code != TUNJA_DANE_CODE:
        raise ValueError("record_outside_tunja")
    year = integer("anno")
    quarter = integer("trimestre")
    if quarter not in {1, 2, 3, 4}:
        raise ValueError("invalid_quarter")
    access_count = integer("accesos_fijos_internet")

    def normalized_text(field: str) -> str:
        value = " ".join(str(row[field]).split())
        return unicodedata.normalize("NFC", value)

    return {
        "year": year,
        "quarter": quarter,
        "period": f"{year}-Q{quarter}",
        "provider": normalized_text("proveedor"),
        "department_code": str(row["codigo_dane_departamento"]).zfill(2),
        "department": normalized_text("departamento"),
        "municipality_code": municipality_code,
        "municipality": normalized_text("municipio"),
        "segment": normalized_text("segmento"),
        "technology": normalized_text("tecnologia"),
        "downstream_mbps": numeric("velocidad_bajada"),
        "upstream_mbps": numeric("velocidad_subida"),
        "reported_accesses": access_count,
        "raw": row,
    }


def _database_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    configured = os.getenv("OFFICIAL_DATA_DB_PATH")
    return (
        Path(configured)
        if configured
        else Path(__file__).resolve().parents[3] / "data" / "official_connectivity.sqlite3"
    )


def load_snapshot(path: Path | None = None) -> dict[str, Any] | None:
    database = _database_path(path)
    if not database.is_file():
        return None
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        batch = connection.execute(
            "SELECT * FROM connectivity_batches WHERE source_id = ? AND status = 'valid' ORDER BY retrieved_at_utc DESC LIMIT 1",
            (SOURCE_ID,),
        ).fetchone()
        if batch is None:
            return None
        records = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT normalized_json FROM connectivity_records WHERE batch_id = ? ORDER BY row_number",
                (batch["batch_id"],),
            )
        ]
    return {
        "batch": dict(batch),
        "records": records,
        "stale": (datetime.now(UTC) - datetime.fromisoformat(batch["retrieved_at_utc"])).days >= 30,
    }


def refresh_official_data(
    path: Path | None = None,
    *,
    fetch_records=_fetch_records,
    fetch_metadata=_metadata,
) -> dict[str, Any]:
    """Fetch, validate and atomically persist a new raw + processed Tunja snapshot."""
    database = _database_path(path)
    database.parent.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(UTC).isoformat()
    try:
        metadata = fetch_metadata()
        cached = load_snapshot(database)
        source_update = metadata.get("source_updated_at_utc")
        if (
            cached
            and source_update
            and cached["batch"].get("source_updated_at_utc") == source_update
        ):
            try:
                with sqlite3.connect(database) as cached_connection:
                    raw_count = cached_connection.execute(
                        "SELECT COUNT(*) FROM connectivity_source_raw WHERE batch_id = ?",
                        (cached["batch"]["batch_id"],),
                    ).fetchone()[0]
                if raw_count == cached["batch"]["retrieved_count"]:
                    return {**cached, "unchanged_source": True, "checked_at_utc": retrieved_at}
            except sqlite3.DatabaseError:
                pass
        raw_records = fetch_records()
        if not raw_records:
            raise ValueError("The source returned no records; existing snapshot was kept.")
        normalized: list[dict[str, Any]] = []
        rejected = 0
        rejection_reasons: Counter[str] = Counter()
        seen: set[str] = set()
        for raw in raw_records:
            key = json.dumps(raw, sort_keys=True, ensure_ascii=False)
            if key in seen:
                rejected += 1
                rejection_reasons["duplicate_source_row"] += 1
                continue
            seen.add(key)
            try:
                normalized.append(_normalize(raw))
            except (KeyError, TypeError, ValueError) as exc:
                rejected += 1
                reason = "record_outside_tunja" if str(exc) == "record_outside_tunja" else str(exc)
                rejection_reasons[reason] += 1
        if not normalized:
            raise ValueError("No valid Tunja records found; existing snapshot was kept.")
        raw_bytes = json.dumps(raw_records, sort_keys=True, ensure_ascii=False).encode("utf-8")
        fingerprint = hashlib.sha256(raw_bytes).hexdigest()
        source_query = urlencode({"$where": f"codigo_dane_municipio = '{TUNJA_DANE_CODE}'"})
        batch_id = hashlib.sha256(f"{SOURCE_ID}:{retrieved_at}:{fingerprint}".encode()).hexdigest()[
            :32
        ]
        connection = sqlite3.connect(database, timeout=10)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript("""CREATE TABLE IF NOT EXISTS connectivity_batches (
                    batch_id TEXT PRIMARY KEY, source_id TEXT NOT NULL, original_url TEXT NOT NULL,
                    dataset_id TEXT NOT NULL, retrieved_at_utc TEXT NOT NULL,
                    source_updated_at_utc TEXT, content_sha256 TEXT NOT NULL,
                    retrieved_count INTEGER NOT NULL, accepted_count INTEGER NOT NULL,
                    rejected_count INTEGER NOT NULL, validation_status TEXT NOT NULL,
                    transformation_version TEXT NOT NULL, status TEXT NOT NULL,
                    error_summary TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS connectivity_records (
                    batch_id TEXT NOT NULL REFERENCES connectivity_batches(batch_id),
                    row_number INTEGER NOT NULL, raw_json TEXT NOT NULL, normalized_json TEXT NOT NULL,
                    PRIMARY KEY(batch_id, row_number));""")
            connection.execute("""CREATE TABLE IF NOT EXISTS connectivity_source_raw (
                    batch_id TEXT NOT NULL REFERENCES connectivity_batches(batch_id),
                    row_number INTEGER NOT NULL, raw_json TEXT NOT NULL,
                    PRIMARY KEY(batch_id, row_number))""")
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO connectivity_batches VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    batch_id,
                    SOURCE_ID,
                    f"{API_URL}?{source_query}",
                    DATASET_ID,
                    retrieved_at,
                    metadata.get("source_updated_at_utc"),
                    fingerprint,
                    len(raw_records),
                    len(normalized),
                    rejected,
                    "valid",
                    "mintic-tunja-fixed-access-v1",
                    "valid",
                    json.dumps(dict(rejection_reasons), ensure_ascii=False),
                ),
            )
            connection.executemany(
                "INSERT INTO connectivity_source_raw VALUES (?, ?, ?)",
                [
                    (batch_id, index, json.dumps(row, sort_keys=True, ensure_ascii=False))
                    for index, row in enumerate(raw_records)
                ],
            )
            for index, record in enumerate(normalized):
                connection.execute(
                    "INSERT INTO connectivity_records VALUES (?, ?, ?, ?)",
                    (
                        batch_id,
                        index,
                        json.dumps(record["raw"], sort_keys=True, ensure_ascii=False),
                        json.dumps(record, sort_keys=True, ensure_ascii=False),
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
        LOGGER.info(
            "Official MinTIC snapshot stored: batch=%s retrieved=%d accepted=%d rejected=%d",
            batch_id,
            len(raw_records),
            len(normalized),
            rejected,
        )
        return load_snapshot(database) or {}
    except Exception as exc:
        cached = load_snapshot(database)
        if cached:
            LOGGER.warning(
                "MinTIC refresh failed; retaining valid local snapshot (%s)", type(exc).__name__
            )
            return {**cached, "refresh_error": type(exc).__name__}
        if isinstance(exc, (SourceUnavailableError, requests.RequestException)):
            raise SourceUnavailableError(
                "MinTIC refresh failed and no valid local snapshot exists."
            ) from exc
        raise
