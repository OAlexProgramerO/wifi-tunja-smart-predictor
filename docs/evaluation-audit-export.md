# Evaluation audit evidence export

V8.7 adds read-only exports for the V8.6 local policy-evaluation audit trail. Exports reuse the same exact-match and inclusive UTC date filters as audit history, and include every matching row rather than the current dashboard page.

## API

- `GET /ml-experiments/policy-audit/export.json`
- `GET /ml-experiments/policy-audit/export.csv`
- `GET /ml-experiments/policy-audit/export.manifest.json?export_format=json|csv`

Both endpoints accept optional `policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, `created_from_utc`, and `created_to_utc` filters. Date bounds must be ordered, UTC-aware ISO timestamps. Invalid ranges return HTTP 422. Results sort by timestamp descending and audit ID ascending for timestamp ties. JSON uses a stable record ordering; its generation timestamp naturally differs between export requests.

The JSON document includes `export_schema_version`, `generated_at_utc`, `applied_filters`, `record_count`, and `records`. Each record contains allowlisted audit/run/event identifiers, experiment and policy references when available, timestamp, execution source, outcome, local digest, verification status/reasons, and selected synthetic provenance fields. Unknown event properties are excluded. Missing/malformed event JSON is represented with available linked-run metadata and `UNVERIFIABLE` status rather than copied verbatim.

The CSV uses fixed columns. Its first data row has `row_type=metadata` and contains the export schema version, generation time, canonical filter JSON, and record count. Remaining rows have `row_type=audit_record`; an empty export still contains the header and metadata row.

The existing dashboard audit filters control both downloads. “Prepare complete filtered audit exports” requests the JSON and CSV files with filters only; the visible history `limit` and `offset` are deliberately omitted.

## Offline manifest and CLI verification (V8.8)

The dashboard prepares a separate manifest for each exact JSON/CSV response it downloads, then offers all four files: export JSON, JSON manifest, export CSV, and CSV manifest. The API manifest endpoint uses the same filters and returns a downloadable manifest for `export_format=json` or `export_format=csv`. When requesting an API export and manifest separately, use the same filters; if matching records change between requests, verification reports a mismatch. The dashboard-generated manifests are based directly on the downloaded bytes.

Example file pairs and commands:

```powershell
python scripts/verify_audit_export.py policy-evaluation-audit.json policy-evaluation-audit.json.manifest.json
python scripts/verify_audit_export.py policy-evaluation-audit.csv policy-evaluation-audit.csv.manifest.json
```

The verifier reads only local files and uses the standard library. Without `--save-history`, it does not start FastAPI/Streamlit, open SQLite, or access the network. It exits `0` for `VALID`, `1` for `INVALID`, and `2` for `UNSUPPORTED`; it prints the status and a concise reason. Malformed files/manifests, mismatched filters/counts, and record digest changes are invalid. Unknown manifest/export schema versions, format, or canonicalization versions are unsupported.

### Canonical records digest

JSON and CSV records are first parsed into the same privacy-allowlisted record objects. The ordered `records` array is serialized as UTF-8 with recursively sorted object keys, compact separators, unescaped Unicode, and non-finite numbers rejected. This is equivalent to `json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")`. Array order is preserved. The manifest stores lowercase SHA-256 of those bytes and identifies this rule as `json-sort-keys-compact-utf8-v1`. CSV quoting, line endings, and JSON indentation do not affect the digest; changing a record or record order does.

The manifest includes `manifest_schema_version`, `export_format`, `canonicalization`, `generated_at_utc`, `record_count`, `applied_filters`, and `records_sha256`. Its timestamp is not included in the records digest. Verification also checks the manifest format, filter metadata, schema, and count against the export.

## Integrity and data boundaries

Exported digest strings preserve V8.6 verification semantics and values; exports do not regenerate, update, or repair stored events or digests. `VERIFIED`, `INVALID`, and `UNVERIFIABLE` remain distinct. A local digest supports consistency checks but is neither cryptographic authenticity nor tamper-proof storage; a privileged database operator could rewrite both an event and its digest.

The exporter emits an explicit field allowlist and does not serialize entire run reports or raw event dictionaries. It does not export chat messages, request bodies, credentials, personal data, or sensitive headers. The reports cover synthetic ML policy evaluations only; official MinTIC descriptive connectivity data is not included.

An offline `VALID` result checks only that the current export records match the adjacent manifest under the documented canonicalization. It is not a signature, proof of origin, or tamper-proof guarantee: someone able to replace both files can also replace the manifest digest.

## Local verification history (V8.9)

Saving is opt-in. Add `--save-history PATH.sqlite` to a verification to persist one local history row:

```powershell
python scripts/verify_audit_export.py policy-evaluation-audit.json policy-evaluation-audit.json.manifest.json --save-history verification-history.sqlite
python scripts/verify_audit_export.py history list --store verification-history.sqlite --result VALID --limit 25 --offset 0
python scripts/verify_audit_export.py history list --store verification-history.sqlite --from-utc 2026-11-01T00:00:00Z --to-utc 2026-11-30T23:59:59Z
python scripts/verify_audit_export.py history summary --store verification-history.sqlite --from-utc 2026-11-01T00:00:00Z --result VALID
python scripts/verify_audit_export.py history show VERIFICATION_ID --store verification-history.sqlite
```

`history summary` accepts `--result`, `--from-utc`, and `--to-utc`; it aggregates every matching row without pagination. Its deterministic JSON fields are `history_schema_version`, `filters`, `total_records`, `result_counts`, `format_counts`, `missing_format_count`, and `date_range_utc` (`null` when no rows match). Result counts always include all three statuses. Format counts omit unknown/null formats, which are reported separately. This summary is metadata only: counts do not measure model quality, WiFi performance, or service reliability. Summary/list/detail use SQLite read-only mode.

History JSON has a fixed schema, sorted object keys, and stable ordering by verification time descending then ID ascending; `limit`/`offset` provide stable pagination. Rows contain only the verification ID and time, export basename and format, manifest schema version, record count, digest, and result. Exported record contents are never copied into SQLite or history reports. The normal verification command remains stateless unless `--save-history` is supplied. History inspection errors return exit code `3`. Listing requires an existing store; a missing, corrupted, or unsupported SQLite store produces a clear error and is never silently recreated, reset, or migrated. History is local metadata and is not an additional audit-event system.

The export history relates to synthetic ML policy evaluation evidence only and does not include official MinTIC descriptive records.
