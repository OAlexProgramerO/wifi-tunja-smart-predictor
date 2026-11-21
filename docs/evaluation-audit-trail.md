# Evaluation audit trail and integrity verification

V8.6 records one local audit event in the same SQLite transaction as each newly created V8.6 policy evaluation run. The event links the audit ID to the run and records the experiment, exact policy ID/version, UTC timestamp, application execution source (`api`), outcome, project version, task, synthetic dataset fingerprint, and evaluation split/window. It does not copy chat messages, request bodies, credentials, personal data, or sensitive headers. It describes synthetic ML evaluations only and is separate from official MinTIC records.

## Canonical encoding and digest

The event JSON is serialized as UTF-8 without a byte-order mark, with recursively sorted object keys, no insignificant whitespace, Unicode characters left unescaped, and non-finite numbers rejected. In Python this is equivalent to:

```python
json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
```

The digest is the lowercase hexadecimal SHA-256 of those UTF-8 bytes. The stored digest covers the event object, not its database row or the linked run report. Verification also compares indexed columns and event references with the saved evaluation run and report.

## Read APIs

- `GET /ml-experiments/policy-runs/{run_id}/audit/verify` verifies the audit event for a run. Missing audit events are `UNVERIFIABLE`; the endpoint does not backfill historical rows.
- `GET /ml-experiments/policy-audit/{audit_id}/verify` verifies by audit ID and also checks the linked run.
- `GET /ml-experiments/policy-audit` returns events with `limit` (1–100), `offset` (up to 100000), and optional `policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, `created_from_utc`, and `created_to_utc` filters. Date bounds are inclusive UTC-aware ISO timestamps. Each item includes its verification result.

Verification states:

- `VERIFIED`: supported event schema, local digest matches, indexed fields agree, and the linked run/report references are consistent.
- `INVALID`: a digest mismatch or a modified/missing/inconsistent linked record was detected.
- `UNVERIFIABLE`: the event is missing or unreadable, or its schema is unsupported, so a reliable verification cannot be completed.

The dashboard exposes these states, event details, filters, and a run-ID verification control. Older V8.5 runs predate audit events and therefore return `UNVERIFIABLE`; their evaluation reports and policy versions are not changed.

## Security and limitations

This is a local consistency check, not a digital signature or proof of origin. It does not make SQLite tamper-proof. A privileged database operator can rewrite both an event and its digest, or replace the whole database. No external audit service, key, or paid dependency is used. A `VERIFIED` result means only that the current local event digest and linked data are consistent under the documented encoding.
