"""Offline audit export manifest, parser, and CLI verification tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from wifi_tunja_smart_predictor.experiments.ml_policy_export import (
    build_policy_audit_export,
    render_policy_audit_csv,
)
from wifi_tunja_smart_predictor.experiments.ml_policy_export_manifest import (
    AUDIT_MANIFEST_SCHEMA_VERSION,
    AUDIT_RECORD_CANONICALIZATION,
    audit_records_sha256,
    canonical_audit_records_json,
    create_policy_audit_manifest,
    create_policy_audit_manifest_from_bytes,
    verify_policy_audit_export_files,
)


def _export():
    audit = {
        "audit_id": "audit-1",
        "run_id": "run-1",
        "timestamp_utc": "2026-11-22T10:00:00+00:00",
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
    return build_policy_audit_export(
        [audit], {"policy_id": "policy-1"}, generated_at_utc="2026-11-22T10:00:00+00:00"
    )


def _write_pair(tmp_path: Path, export_format: str, export=None):
    exported = export or _export()
    export_path = tmp_path / f"audit.{export_format}"
    if export_format == "json":
        export_path.write_text(json.dumps(exported, ensure_ascii=False), encoding="utf-8")
    else:
        export_path.write_text(render_policy_audit_csv(exported), encoding="utf-8", newline="")
    manifest = create_policy_audit_manifest(
        export_format, exported, generated_at_utc="2026-11-22T11:00:00+00:00"
    )
    manifest_path = tmp_path / f"audit.{export_format}.manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return export_path, manifest_path, manifest


@pytest.mark.parametrize("export_format", ["json", "csv"])
def test_json_and_csv_exports_verify_offline(tmp_path, export_format):
    export_path, manifest_path, manifest = _write_pair(tmp_path, export_format)
    result = verify_policy_audit_export_files(export_path, manifest_path)
    assert result["status"] == "VALID"
    assert result["record_count"] == 1
    assert manifest["manifest_schema_version"] == AUDIT_MANIFEST_SCHEMA_VERSION
    assert manifest["export_format"] == export_format
    assert manifest["generated_at_utc"] == "2026-11-22T11:00:00+00:00"
    assert manifest["applied_filters"] == {"policy_id": "policy-1"}
    assert manifest["records_sha256"] == result["computed_sha256"]


@pytest.mark.parametrize("export_format", ["json", "csv"])
def test_altered_export_records_return_invalid(tmp_path, export_format):
    export_path, manifest_path, _ = _write_pair(tmp_path, export_format)
    if export_format == "json":
        payload = json.loads(export_path.read_text(encoding="utf-8"))
        payload["records"][0]["outcome"] = "FAILED"
        export_path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        contents = export_path.read_text(encoding="utf-8").replace("PASSED", "FAILED")
        export_path.write_text(contents, encoding="utf-8", newline="")
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "INVALID"


def test_malformed_manifest_and_malformed_export_are_invalid(tmp_path):
    export_path, manifest_path, _ = _write_pair(tmp_path, "json")
    manifest_path.write_text("{bad json", encoding="utf-8")
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "INVALID"

    _, manifest_path, _ = _write_pair(tmp_path, "json")
    export_path.write_text("not json", encoding="utf-8")
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "INVALID"


def test_exports_with_unallowlisted_fields_or_malformed_csv_rows_are_invalid(tmp_path):
    export_path, manifest_path, _ = _write_pair(tmp_path, "json")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    payload["chat_message"] = "private"
    export_path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "INVALID"

    export_path, manifest_path, _ = _write_pair(tmp_path, "csv")
    export_path.write_text(
        export_path.read_text(encoding="utf-8") + ",extra-field\n", encoding="utf-8"
    )
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "INVALID"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("manifest_schema_version", "99.0"),
        ("export_format", "xml"),
        ("canonicalization", "other-v2"),
    ],
)
def test_unknown_manifest_versions_and_formats_are_unsupported(tmp_path, field, value):
    export_path, manifest_path, manifest = _write_pair(tmp_path, "json")
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "UNSUPPORTED"


def test_unknown_export_schema_and_mismatched_format_are_reported(tmp_path):
    export_path, manifest_path, manifest = _write_pair(tmp_path, "json")
    payload = json.loads(export_path.read_text(encoding="utf-8"))
    payload["export_schema_version"] = "99.0"
    export_path.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "UNSUPPORTED"

    export_path, manifest_path, manifest = _write_pair(tmp_path, "json")
    manifest["export_format"] = "csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "INVALID"


@pytest.mark.parametrize("export_format", ["json", "csv"])
def test_empty_exports_have_valid_manifests(tmp_path, export_format):
    empty = build_policy_audit_export([], {}, generated_at_utc="2026-11-22T10:00:00+00:00")
    export_path, manifest_path, _ = _write_pair(tmp_path, export_format, empty)
    assert verify_policy_audit_export_files(export_path, manifest_path)["status"] == "VALID"


def test_canonical_record_encoding_is_stable_but_preserves_array_order():
    first_record = _export()["records"][0]
    first_record["policy_id"] = "pñ"
    second_record = {**first_record, "audit_id": "audit-2", "run_id": "run-2"}
    first = [first_record, second_record]
    reordered_keys = [dict(reversed(list(record.items()))) for record in first]
    reordered_records = list(reversed(first))
    assert canonical_audit_records_json(first) == canonical_audit_records_json(reordered_keys)
    assert audit_records_sha256(first) == audit_records_sha256(reordered_keys)
    assert audit_records_sha256(first) != audit_records_sha256(reordered_records)
    with pytest.raises(ValueError, match="field set"):
        audit_records_sha256([{"request_body": "secret"}])
    assert AUDIT_RECORD_CANONICALIZATION == "json-sort-keys-compact-utf8-v1"


def test_json_and_csv_files_share_one_canonical_records_digest():
    export = _export()
    json_data = json.dumps(export, ensure_ascii=False).encode("utf-8")
    csv_data = render_policy_audit_csv(export).encode("utf-8")
    json_manifest = create_policy_audit_manifest_from_bytes(json_data, "json")
    csv_manifest = create_policy_audit_manifest_from_bytes(csv_data, "csv")
    assert json_manifest["records_sha256"] == csv_manifest["records_sha256"]


def test_cli_exit_codes_are_valid_invalid_and_unsupported(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_audit_export.py"
    export_path, manifest_path, manifest = _write_pair(tmp_path, "json")

    def invoke():
        return subprocess.run(
            [sys.executable, str(script), str(export_path), str(manifest_path)],
            capture_output=True,
            text=True,
            check=False,
        )

    valid = invoke()
    assert valid.returncode == 0
    assert valid.stdout.startswith("VALID:")

    export_path.write_text("{}", encoding="utf-8")
    invalid = invoke()
    assert invalid.returncode == 1
    assert invalid.stdout.startswith("INVALID:")

    manifest["manifest_schema_version"] = "88.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    unsupported = invoke()
    assert unsupported.returncode == 2
    assert unsupported.stdout.startswith("UNSUPPORTED:")
