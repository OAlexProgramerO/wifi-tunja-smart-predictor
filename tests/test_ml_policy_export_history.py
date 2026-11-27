"""Metadata-only history for offline audit export checks."""

from __future__ import annotations

import csv
import io
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
    SUMMARY_CSV_FIELDS,
    VerificationHistoryError,
    build_verification_history_record,
    check_verification_history_integrity,
    inspect_verification_history,
    list_verification_history,
    render_history_csv,
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
                        "project_version": "0.8.12",
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
    with sqlite3.connect(store) as connection:
        connection.execute("DELETE FROM verification_history")
    empty = summarize_verification_history(store)
    assert empty["total_records"] == 0
    assert empty["result_counts"] == {"INVALID": 0, "UNSUPPORTED": 0, "VALID": 0}
    assert empty["format_counts"] == {}
    assert empty["missing_format_count"] == 0
    assert empty["date_range_utc"] is None
    csv_rows = list(csv.DictReader(io.StringIO(render_history_csv(empty), newline="")))
    assert [row["row_type"] for row in csv_rows] == ["summary", "result", "result", "result"]
    assert csv_rows[0]["date_range_from_utc"] == ""
    assert csv_rows[0]["missing_format_count"] == "0"


def test_integrity_check_is_consistent_read_only_filtered_and_deterministic(tmp_path):
    store = tmp_path / "history.sqlite"
    records = [
        _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"),
        _record(tmp_path, "INVALID", "2026-11-24T11:00:00+00:00"),
    ]
    for record in records:
        save_verification_history(store, record)
    before = store.read_bytes()
    filters = {
        "result": "VALID",
        "from_utc": "2026-11-24T09:00:00Z",
        "to_utc": "2026-11-24T10:30:00Z",
    }
    report = check_verification_history_integrity(store, **filters)
    assert report["status"] == "CONSISTENT"
    assert report["records_checked"] == 1
    assert report["summary"]["result_counts"]["VALID"] == 1
    assert report["recomputed_summary"] == report["summary"]
    assert report["filters"]["result"] == "VALID"
    assert render_history_json(report) == render_history_json(
        check_verification_history_integrity(store, **filters)
    )
    assert store.read_bytes() == before
    encoded = render_history_json(report)
    assert "audit-1" not in encoded
    assert "experiment-1" not in encoded
    assert "verification_reasons" not in encoded


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("total_records", 99),
        ("result_counts", {"VALID": 0, "INVALID": 1, "UNSUPPORTED": 0}),
        ("format_counts", {"csv": 99}),
        (
            "date_range_utc",
            {"from": "2000-01-01T00:00:00+00:00", "to": "2000-01-01T00:00:00+00:00"},
        ),
    ],
)
def test_integrity_check_detects_mismatched_aggregates_and_boundaries(
    tmp_path, monkeypatch, field, replacement
):
    import wifi_tunja_smart_predictor.experiments.ml_policy_export_history as history

    store = tmp_path / "history.sqlite"
    save_verification_history(store, _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"))
    original = history.summarize_verification_history

    def corrupted_summary(*args, **kwargs):
        summary = original(*args, **kwargs)
        summary[field] = replacement
        return summary

    monkeypatch.setattr(history, "summarize_verification_history", corrupted_summary)
    report = history.check_verification_history_integrity(store)
    assert report["status"] == "INCONSISTENT"
    assert any(finding["field"] == field for finding in report["findings"])


def test_integrity_check_is_indeterminate_for_unavailable_or_incomplete_evidence(tmp_path):
    missing = check_verification_history_integrity(tmp_path / "missing.sqlite")
    assert missing["status"] == "INDETERMINATE"
    assert missing["findings"][0]["code"] == "HISTORY_UNAVAILABLE"
    corrupted_store = tmp_path / "corrupted.sqlite"
    corrupted_store.write_bytes(b"not a sqlite database")
    corrupted = check_verification_history_integrity(corrupted_store)
    assert corrupted["status"] == "INDETERMINATE"
    assert corrupted["findings"][0]["code"] == "HISTORY_UNAVAILABLE"

    store = tmp_path / "incomplete.sqlite"
    save_verification_history(store, _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"))
    with sqlite3.connect(store) as connection:
        connection.execute("UPDATE verification_history SET timestamp_utc = 'not-a-timestamp'")
    report = check_verification_history_integrity(store)
    assert report["status"] == "INDETERMINATE"
    assert report["findings"][0]["code"] == "INSUFFICIENT_METADATA"
    assert "timestamp_utc" in report["findings"][0]["fields"]


def test_empty_history_integrity_is_consistent(tmp_path):
    store = tmp_path / "history.sqlite"
    save_verification_history(store, _record(tmp_path, "VALID", "2026-11-24T10:00:00+00:00"))
    with sqlite3.connect(store) as connection:
        connection.execute("DELETE FROM verification_history")
    report = check_verification_history_integrity(store)
    assert report["status"] == "CONSISTENT"
    assert report["records_checked"] == 0
    assert report["summary"]["date_range_utc"] is None


def test_csv_summary_has_stable_columns_rows_and_no_evidence_contents(tmp_path):
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
    report = summarize_verification_history(store)
    first = render_history_csv(report)
    assert first == render_history_csv(report)
    assert "\r" not in first
    assert "audit-1" not in first
    assert "experiment-1" not in first
    assert "verification_reasons" not in first
    assert "audit_id" not in first
    assert "request_body" not in first
    reader = csv.DictReader(io.StringIO(first, newline=""))
    assert tuple(reader.fieldnames or ()) == SUMMARY_CSV_FIELDS
    rows = list(reader)
    assert [row["row_type"] for row in rows] == [
        "summary",
        "result",
        "result",
        "result",
        "format",
        "format",
    ]
    assert [row["result"] for row in rows[1:4]] == ["VALID", "INVALID", "UNSUPPORTED"]
    assert [row["export_format"] for row in rows[4:]] == ["csv", "json"]
    assert rows[0]["total_records"] == str(report["total_records"])
    assert rows[0]["date_range_from_utc"] == report["date_range_utc"]["from"]


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
    checked = subprocess.run(
        [sys.executable, str(script), "history", "check", "--store", str(store)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checked.returncode == 0
    assert json.loads(checked.stdout)["status"] == "CONSISTENT"
    csv_output = subprocess.run(
        [
            sys.executable,
            str(script),
            "history",
            "summary",
            "--store",
            str(store),
            "--result",
            "VALID",
            "--from-utc",
            "2000-01-01T00:00:00Z",
            "--to-utc",
            "2099-12-31T23:59:59Z",
            "--format",
            "csv",
        ],
        capture_output=True,
        text=False,
        check=False,
    )
    assert csv_output.returncode == 0
    assert b"\r" not in csv_output.stdout
    csv_text = csv_output.stdout.decode("utf-8")
    parsed_csv = list(csv.DictReader(io.StringIO(csv_text, newline="")))
    assert parsed_csv[0]["total_records"] == "1"
    assert parsed_csv[0]["filter_result"] == "VALID"
    assert [row["result_count"] for row in parsed_csv[1:4]] == ["1", "0", "0"]
    filtered_json = subprocess.run(
        [
            sys.executable,
            str(script),
            "history",
            "summary",
            "--store",
            str(store),
            "--result",
            "VALID",
            "--from-utc",
            "2000-01-01T00:00:00Z",
            "--to-utc",
            "2099-12-31T23:59:59Z",
            "--format",
            "json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    filtered_json_report = json.loads(filtered_json.stdout)
    assert parsed_csv[0]["total_records"] == str(filtered_json_report["total_records"])
    assert {
        row["result"]: int(row["result_count"]) for row in parsed_csv[1:4]
    } == filtered_json_report["result_counts"]
    repeated_csv = subprocess.run(
        [
            sys.executable,
            str(script),
            "history",
            "summary",
            "--store",
            str(store),
            "--result",
            "VALID",
            "--from-utc",
            "2000-01-01T00:00:00Z",
            "--to-utc",
            "2099-12-31T23:59:59Z",
            "--format",
            "csv",
        ],
        capture_output=True,
        text=False,
        check=False,
    )
    assert repeated_csv.stdout == csv_output.stdout
    output_path = tmp_path / "summary.csv"
    file_output = subprocess.run(
        [
            sys.executable,
            str(script),
            "history",
            "summary",
            "--store",
            str(store),
            "--format",
            "csv",
            "--output",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert file_output.returncode == 0
    assert "Summary written to:" in file_output.stdout
    assert output_path.read_bytes() == render_history_csv(
        summarize_verification_history(store)
    ).encode("utf-8")
    invalid_format = subprocess.run(
        [
            sys.executable,
            str(script),
            "history",
            "summary",
            "--store",
            str(store),
            "--format",
            "xml",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid_format.returncode == 2
    output_error = subprocess.run(
        [
            sys.executable,
            str(script),
            "history",
            "summary",
            "--store",
            str(store),
            "--output",
            str(tmp_path / "absent" / "summary.csv"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert output_error.returncode == 3
    assert "History output error:" in output_error.stderr
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
    unavailable = subprocess.run(
        [sys.executable, str(script), "history", "check", "--store", str(tmp_path / "missing")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert unavailable.returncode == 2
    assert json.loads(unavailable.stdout)["status"] == "INDETERMINATE"


def test_integrity_cli_returns_nonzero_for_inconsistent_results(monkeypatch, capsys, tmp_path):
    import wifi_tunja_smart_predictor.experiments.ml_policy_export_history as history

    monkeypatch.setattr(
        history,
        "check_verification_history_integrity",
        lambda *_args, **_kwargs: {
            "integrity_report_schema_version": "1.0",
            "status": "INCONSISTENT",
            "findings": [{"code": "TOTAL_RECORDS_MISMATCH"}],
        },
    )
    result = history.history_cli_main(["check", "--store", str(tmp_path / "history.sqlite")])
    assert result == 1
    assert json.loads(capsys.readouterr().out)["status"] == "INCONSISTENT"
