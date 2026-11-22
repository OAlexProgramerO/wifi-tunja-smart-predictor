# Evaluation audit evidence export

V8.7 adds read-only exports for the V8.6 local policy-evaluation audit trail. Exports reuse the same exact-match and inclusive UTC date filters as audit history, and include every matching row rather than the current dashboard page.

## API

- `GET /ml-experiments/policy-audit/export.json`
- `GET /ml-experiments/policy-audit/export.csv`

Both endpoints accept optional `policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, `created_from_utc`, and `created_to_utc` filters. Date bounds must be ordered, UTC-aware ISO timestamps. Invalid ranges return HTTP 422. Results sort by timestamp descending and audit ID ascending for timestamp ties. JSON uses a stable record ordering; its generation timestamp naturally differs between export requests.

The JSON document includes `export_schema_version`, `generated_at_utc`, `applied_filters`, `record_count`, and `records`. Each record contains allowlisted audit/run/event identifiers, experiment and policy references when available, timestamp, execution source, outcome, local digest, verification status/reasons, and selected synthetic provenance fields. Unknown event properties are excluded. Missing/malformed event JSON is represented with available linked-run metadata and `UNVERIFIABLE` status rather than copied verbatim.

The CSV uses fixed columns. Its first data row has `row_type=metadata` and contains the export schema version, generation time, canonical filter JSON, and record count. Remaining rows have `row_type=audit_record`; an empty export still contains the header and metadata row.

The existing dashboard audit filters control both downloads. “Prepare complete filtered audit exports” requests the JSON and CSV files with filters only; the visible history `limit` and `offset` are deliberately omitted.

## Integrity and data boundaries

Exported digest strings preserve V8.6 verification semantics and values; exports do not regenerate, update, or repair stored events or digests. `VERIFIED`, `INVALID`, and `UNVERIFIABLE` remain distinct. A local digest supports consistency checks but is neither cryptographic authenticity nor tamper-proof storage; a privileged database operator could rewrite both an event and its digest.

The exporter emits an explicit field allowlist and does not serialize entire run reports or raw event dictionaries. It does not export chat messages, request bodies, credentials, personal data, or sensitive headers. The reports cover synthetic ML policy evaluations only; official MinTIC descriptive connectivity data is not included.
