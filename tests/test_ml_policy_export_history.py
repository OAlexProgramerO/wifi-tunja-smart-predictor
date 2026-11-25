"""Metadata-only history for offline audit export checks."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from wifi_tunja_smart_predictor.experiments.ml_policy_export import (
    build_policy_audit_export,
    render_policy_audit_csv,
)
from wifi_tunja_smart_predictor.experiments.ml_policy_export_history import (
    VerificationHistoryError,
    build_verification_history_record,
    inspect_verification_history,
    list_verification_history,
    render_history_json,
    save_verification_history,
    summarize_verification_history,
)
from wifi_tunja_smart_predictor.experiments.ml_policy_export_manifest import (
    create_policy_audit_manifest,
    verify_policy_audit_export_files,
)


def _write_pair(tmp_path: Path, export_format: str = "json"):
    exported = build_policy_audit_export(
        [
            {
                "audit_id": "audit-1",
                "run_id": "run-1",
                "timestamp_utc": "2026-11-24T10:00:00+00:00",
                "execution_source": "api",
                "outcome": "PASSED",
                "digest_sha256": "a" * 64,
                "event": {
                    "run_id": "run-1",
                    "experiment_id": "experiment-1",
                    "policy_id": "policy-1",
                    "policy_version": 1,
                    "provenance": {
                        "project_version": "0.8.10",
                        "task": "classification",
                        "dataset_kind": "synthetic",
                        "dataset_fingerprint_algorithm": "sha256",
                        "dataset_fingerprint": "b" * 64,
                        "evaluation_split": "validation",
                        "evaluation_window": {"start": "2026-01-01", "end": "2026-01-31"},
                    },
                },
                "verification": {"status": "VERIFIED", "reasons": []},
            }
        ],
        {},
        generated_at_utc="2026-11-24T10:00:00+00:00",
    )
    export_path = tmp_path / f"audit.{export_format}"
    if export_format == "json":
        export_path.write_text(json.dumps(exported), encoding="utf-8")
    else:
        export_path.write_text(render_policy_audit_csv(exported), encoding="utf-8")
    manifest = create_policy_audit_manifest(
        export_format, exported, generated_at_utc="2026-11-24T10:05:00+00:00"
    )
    manifest_path = tmp_path / f"audit.{export_format}.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return export_path, manifest_path, manifest


def _record(tmp_path: Path, result: str, timestamp: str):
    export_path, manifest_path, manifest = _write_pair(tmp_path)
    if result == "INVALID":
        export_path.write_text(export_path.read_text(encoding="utf-8").replace("PASSED", "FAILED"))
    elif result == "UNSUPPORTED":
        manifest["manifest_schema_version"] = "future"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    outcome = verify_policy_audit_export_files(export_path, manifest_path)
    return build_verification_history_record(
        export_path,
        manifest_path,
        outcome,
        timestamp_utc=timestamp,
    )


@pytest.mark.parametrize("result", ["VALID", "INVALID", "UNSUPPORTED"])
def test_saving_verification_is_opt_in_and_metadata_only(tmp_path, result):
    record = _record(tmp_path, result, "2026-11-24T10:00:00+00:00")
    store = tmp_path / "history.sqlite"
    assert not store.exists()
    save_verification_history(store, record)
    report = list_verification_history(store)
    assert report["pagination"]["total"] == 1
    assert report["records"][0] == record
    serialized = render_history_json(report)
    assert result in serialized
    assert "audit-1" not in serialized
    assert "experiment-1" not in serialized
    assert "verification_reasons" not in serialized
    assert "request_body" not in serialized


def test_history_lists_filters_stably_and_supports_pagination(tmp_path):
    store = tmp_path / "history.sqlite"
    records = [
        _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"),
        _record(tmp_path, "INVALID", "2026-11-24T11:00:00+00:00"),
        _record(tmp_path, "VALID", "2026-11-24T12:00:00+00:00"),
    ]
    for record in records:
        save_verification_history(store, record)
    first = list_verification_history(store, result="VALID", limit=1, offset=0)
    second = list_verification_history(store, result="VALID", limit=1, offset=1)
    assert first["pagination"]["total"] == 2
    assert first["records"][0]["timestamp_utc"] > second["records"][0]["timestamp_utc"]
    filtered = list_verification_history(
        store,
        from_utc="2026-11-24T10:30:00Z",
        to_utc="2026-11-24T11:30:00Z",
    )
    assert [row["result"] for row in filtered["records"]] == ["INVALID"]
    assert render_history_json(filtered) == render_history_json(filtered)
    detail = inspect_verification_history(store, records[0]["verification_id"])
    assert detail["record"] == records[0]


def test_summary_aggregates_result_format_missing_metadata_and_date_range(tmp_path):
    store = tmp_path / "history.sqlite"
    records = [
        _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"),
        _record(tmp_path, "INVALID", "2026-11-24T11:00:00+00:00"),
        _record(tmp_path, "UNSUPPORTED", "2026-11-24T12:00:00+00:00"),
    ]
    records[1]["export_format"] = "csv"
    records[2]["export_format"] = None
    for record in records:
        save_verification_history(store, record)
    stored_bytes = store.read_bytes()
    report = summarize_verification_history(store)
    assert store.read_bytes() == stored_bytes
    assert report["total_records"] == 3
    assert report["result_counts"] == {"INVALID": 1, "UNSUPPORTED": 1, "VALID": 1}
    assert report["format_counts"] == {"csv": 1, "json": 1}
    assert report["missing_format_count"] == 1
    assert report["date_range_utc"] == {
        "from": "2026-11-24T10:00:00+00:00",
        "to": "2026-11-24T12:00:00+00:00",
    }
    valid_only = summarize_verification_history(store, result="VALID")
    assert valid_only["total_records"] == 1
    assert valid_only["result_counts"] == {"INVALID": 0, "UNSUPPORTED": 0, "VALID": 1}
    date_only = summarize_verification_history(
        store,
        from_utc="2026-11-24T10:30:00Z",
        to_utc="2026-11-24T11:30:00Z",
    )
    assert date_only["result_counts"]["INVALID"] == 1
    serialized = render_history_json(report)
    assert serialized == render_history_json(report)
    assert "audit-1" not in serialized
    assert "experiment-1" not in serialized
    assert "verification_reasons" not in serialized


def test_summary_of_empty_history_and_no_matching_rows(tmp_path):
    store = tmp_path / "history.sqlite"
    save_verification_history(store, _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"))
    empty = summarize_verification_history(store, result="UNSUPPORTED")
    assert empty["total_records"] == 0
    assert empty["result_counts"] == {"INVALID": 0, "UNSUPPORTED": 0, "VALID": 0}
    assert empty["format_counts"] == {}
    assert empty["missing_format_count"] == 0
    assert empty["date_range_utc"] is None


def test_history_reports_missing_corrupted_and_unsupported_stores(tmp_path):
    missing = tmp_path / "missing.sqlite"
    with pytest.raises(VerificationHistoryError, match="does not exist"):
        list_verification_history(missing)
    corrupted = tmp_path / "corrupted.sqlite"
    corrupted.write_bytes(b"not a database")
    with pytest.raises(VerificationHistoryError, match="Cannot open|corrupted"):
        list_verification_history(corrupted)
    with pytest.raises(VerificationHistoryError, match="Cannot open|corrupted"):
        summarize_verification_history(corrupted)
    unsupported = tmp_path / "unsupported.sqlite"
    with sqlite3.connect(unsupported) as connection:
        connection.execute("PRAGMA user_version = 99")
        connection.execute("CREATE TABLE retained(value TEXT)")
    with pytest.raises(VerificationHistoryError, match="Unsupported"):
        save_verification_history(
            unsupported,
            _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"),
        )
    with sqlite3.connect(unsupported) as connection:
        assert connection.execute("SELECT count(*) FROM retained").fetchone()[0] == 0


def test_history_rejects_bad_filters_and_missing_inspection_ids(tmp_path):
    store = tmp_path / "history.sqlite"
    save_verification_history(store, _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"))
    with pytest.raises(VerificationHistoryError, match="result filter"):
        list_verification_history(store, result="UNKNOWN")
    with pytest.raises(VerificationHistoryError, match="from_utc"):
        list_verification_history(
            store,
            from_utc="2026-11-25T00:00:00Z",
            to_utc="2026-11-24T00:00:00Z",
        )
    with pytest.raises(VerificationHistoryError, match="not found"):
        inspect_verification_history(store, "absent")


def test_cli_saves_lists_inspects_and_reports_history_errors(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_audit_export.py"
    export, manifest, _ = _write_pair(tmp_path, "json")
    store = tmp_path / "verification-history.sqlite"
    saved = subprocess.run(
        [sys.executable, str(script), str(export), str(manifest), "--save-history", str(store)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert saved.returncode == 0
    assert "VALID:" in saved.stdout
    verification_id = saved.stdout.split("Saved verification ID: ", 1)[1].strip()
    export.write_text(export.read_text(encoding="utf-8").replace("PASSED", "FAILED"))
    invalid_saved = subprocess.run(
        [sys.executable, str(script), str(export), str(manifest), "--save-history", str(store)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid_saved.returncode == 1
    assert "Saved verification ID:" in invalid_saved.stdout
    listed = subprocess.run(
        [sys.executable, str(script), "history", "list", "--store", str(store)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert listed.returncode == 0
    listed_records = json.loads(listed.stdout)["records"]
    assert any(row["verification_id"] == verification_id for row in listed_records)
    assert any(row["result"] == "INVALID" for row in listed_records)
    summary = subprocess.run(
        [sys.executable, str(script), "history", "summary", "--store", str(store)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert summary.returncode == 0
    assert json.loads(summary.stdout)["total_records"] == 2
    shown = subprocess.run(
        [sys.executable, str(script), "history", "show", verification_id, "--store", str(store)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert shown.returncode == 0
    assert json.loads(shown.stdout)["record"]["result"] == "VALID"
    failed = subprocess.run(
        [sys.executable, str(script), "history", "list", "--store", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 3
    assert "History error:" in failed.stderr
