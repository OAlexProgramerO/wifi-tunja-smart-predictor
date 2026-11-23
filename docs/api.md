# API reference

## V5 experiment routes

V6.1 adds `POST /experiments/optimize`. The typed request supports up to 12 zones, integer per-zone minimum/maximum and baseline allocation, an integer total connection budget (0–100,000), total or explicitly weighted deficit objectives, and a bounded maximum of 100,000 allocation steps. A successful request returns HTTP 201 with the persisted optimization in the existing experiment record. An unknown synthetic zone returns 422; unavailable model artifacts return 503. Use the existing bounded list, retrieval, and JSON/Markdown report routes. The existing optimization report exposes `optimization_analysis` in schema 1.3. Current report schema 1.4 also adds optional `robustness_analysis`; prior records remain readable with additive fields null.

V6.2.0 adds `POST /experiments/robustness` for a saved optimization. It accepts inclusive demand multiplier bounds (0.25–2, up to 6 total values including the inserted 1.0 reference), budget bounds (0–100,000, up to 20 total values including the parent-budget reference), up to 100 evaluations, and up to 10,000 optimizer steps per case. It changes one synthetic zone demand at a time, then separately sweeps budget, including original reference assumptions if omitted. Results distinguish the original allocation objective under each assumption from the reoptimized objective. Incomplete parent integrity is not marked verified. Retrieve and export through the existing experiment routes. See [sensitivity and robustness](robustness.md). This feature was released in V6.2.0.

Example request (illustrative synthetic inputs only):

```json
{
  "name": "Illustrative allocation",
  "datetime": "2025-01-01T12:00:00",
  "objective": "MINIMIZE_WEIGHTED_DEFICIT",
  "total_budget_connections": 3,
  "max_allocation_steps": 100,
  "zones": [
    {"zone_id": "synthetic_zone_a", "capacity_connections": 3, "priority_weight": 2},
    {"zone_id": "synthetic_zone_b", "capacity_connections": 4, "priority_weight": 1}
  ]
}
```

The optimizer obtains fixed next-hour synthetic connection estimates from the current regressor, then allocates integer units against the declared objective. It does not retrain or alter the prediction. A proven optimum applies only to the separable deficit objective and declared independent zone bounds plus shared budget. See the [optimization method](optimization.md).

| Method and path | Behavior |
|---|---|
| `POST /experiments/scenarios` | Validate and run 1–12 synthetic zone estimates, apply deterministic capacity planning, and persist an experiment (201). |
| `POST /experiments/what-if` | Validate and execute one baseline and 1–5 alternatives over the same 1–12 synthetic zones at one timestamp; save a reproducible comparison (201). |
| `GET /experiments?limit=50&offset=0` | Return a bounded history page (limit 1–100). |
| `GET /experiments/{experiment_id}` | Retrieve one saved experiment (404 when absent). |
| `POST /experiments/compare` | Compare 2–10 records only when time window, dataset identity, and model artifact fingerprints agree (409 otherwise). |
| `GET /experiments/{experiment_id}/report?format=json` | Export a structured report; `format=markdown` returns a readable report. |

The comparison is relative to the first supplied experiment and reports only shared zone IDs. Its additive `comparison_status`, `scientific_comparability`, `comparability_reasons`, and `interpretation_detail` fields distinguish a technically compatible comparison from one with no shared zones; the latter returns no zone deltas and is marked `not_assessable`. Shared-zone results remain limited synthetic comparisons, not evidence of real-world equivalence. All demand and capacity values are synthetic connection counts, not throughput or real-network advice.

`POST /experiments/what-if` uses the typed contract in OpenAPI: `name`, a common `datetime`, a `baseline`, and one to five `alternatives`. Each scenario declares a unique bounded list of zone IDs, nonnegative assumed connection capacity per zone, and an optional bounded simulated additional-connection budget. The existing predictor runs once per zone, so predicted demand is fixed across scenarios; only capacity/allocation calculations differ. The response is the saved experiment record. Use existing retrieval and report routes. See [scenario workbench formulas and limitations](scenario-workbench.md).

## Main API (port 8000)

V8.1 adds synchronous synthetic ML experiment routes: `POST /ml-experiments`, `GET /ml-experiments?limit=20&offset=0`, `GET /ml-experiments/{experiment_id}`, and `GET /ml-experiments/{experiment_id}/report`. Create requests define task, allowlisted candidate, validation/test split, and bounded parameters; an optional `X-Idempotency-Key` makes retries safe. Pages are bounded to 100 records. Only the additive experiment-history table is written, never model artifacts. See the [experiment orchestration guide](experiment-orchestration.md).

V8.2 adds read-only `POST /ml-experiments/compare` with two to ten unique saved experiment IDs. It checks synthetic dataset fingerprint, task, target/features, runtime, split, and chronological windows, then returns every pair with metric directions, deltas, and incompatibility reasons. It never ranks a universal best model or writes experiment history. See [Advanced ML experiment comparison](experiment-comparison.md).

V8.3 adds read-only `POST /ml-experiments/gates`. Provide 2–10 unique saved IDs, one task, and one or more unique `{metric, operator, threshold}` criteria. Classification accepts accuracy/precision/recall/F1/ROC-AUC with `gte`; regression accepts MAE/RMSE with `lte` and R² with `gte`. Each run returns `PASSED`, `FAILED`, or `INCONCLUSIVE`, criterion evidence, and dataset/evaluation provenance. Runs must be compatible under the V8.2 task, synthetic fingerprint, runtime, configuration, and chronological-window contract; incompatible runs are inconclusive and thresholds are not judged. See [Automated experiment evaluation gates](experiment-evaluation-gates.md).

V8.4 adds an immutable reusable policy registry: `POST /ml-experiments/policies` creates version 1, and posting with an existing `policy_id` appends the next version. `GET /ml-experiments/policies?limit=20&offset=0` lists each profile's latest version; `GET /ml-experiments/policies/{policy_id}` retrieves its latest version, `?version=n` retrieves an exact version, and `GET /ml-experiments/policies/{policy_id}/versions` lists its versions. `POST /ml-experiments/policies/{policy_id}/versions/{version}/evaluate` accepts `{"experiment_id":"..."}` and returns a versioned report naming the exact policy and criteria. Evaluation reuses the V8.3 compatibility and threshold engine. Unsupported tasks, incomplete provenance, or missing metrics are explicitly inconclusive. Policy writes affect only the additive policy table; experiment records and models are unchanged. See [Reusable evaluation policy registry](evaluation-policy-registry.md).

V8.5 persists each response from the policy evaluation route above as an immutable run. `GET /ml-experiments/policy-runs` returns bounded pagination (`limit` 1–100, `offset` up to 100000) and supports exact-match filters `policy_id`, `policy_version`, `experiment_id`, `status`, plus inclusive `created_from_utc` and `created_to_utc` ISO timestamps. `GET /ml-experiments/policy-runs/{run_id}` returns run metadata, criteria evidence, provenance, and the saved report; `/report` returns the JSON report as a download. `POST /ml-experiments/policy-runs/compare` accepts `left_run_id` and `right_run_id`. Comparisons require the same experiment and policy ID, task, synthetic dataset fingerprint, split/window, evaluator version, and metric/operator criteria; threshold versions may differ. Incompatible contexts include reasons and no criterion comparisons; missing metric evidence yields `PARTIAL`. See [policy run history and comparison](evaluation-policy-registry.md).

V8.6 creates one privacy-safe audit event atomically with each new policy evaluation run. `GET /ml-experiments/policy-audit` supports bounded pagination and optional filters `policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, `created_from_utc`, and `created_to_utc`; each item includes verification status. `GET /ml-experiments/policy-runs/{run_id}/audit/verify` verifies by evaluation run ID, including missing events. `GET /ml-experiments/policy-audit/{audit_id}/verify` verifies by audit ID. Status is `VERIFIED`, `INVALID`, or `UNVERIFIABLE`. The SHA-256 digest covers canonical event JSON and the verifier checks indexed fields plus the linked run/report. It detects local inconsistencies but does not provide cryptographic authenticity or prevent a privileged database operator from rewriting both records and digests. V8.5 runs are not backfilled and verify as `UNVERIFIABLE`. See the [evaluation audit trail guide](evaluation-audit-trail.md).

V8.7 adds `GET /ml-experiments/policy-audit/export.json` and `/export.csv`. Both reuse the V8.6 audit filters (`policy_id`, `policy_version`, `experiment_id`, `outcome`, `execution_source`, and inclusive UTC date bounds) but deliberately omit history pagination and export every matching event in deterministic timestamp-descending/audit-ID-ascending order. JSON includes schema version, generation time, applied filters, record count, and allowlisted event/provenance fields. CSV uses a metadata row followed by audit rows, including for empty results. Responses set attachment filenames and appropriate content types. Invalid date ranges return HTTP 422. Exports include verification statuses and original digest values without changing stored data; unknown event properties are never serialized. The digest limitation described above still applies. See [Audit evidence export](evaluation-audit-export.md).

V8.8 adds `GET /ml-experiments/policy-audit/export.manifest.json?export_format=json|csv`, returning an attachment manifest with schema version, format, generation timestamp, record count, applied filters, canonicalization identifier, and SHA-256 over the canonical parsed `records` array. The dashboard creates each manifest from the exact downloaded file bytes and provides separate downloads for JSON and CSV manifests. Offline CLI verification is available as `python scripts/verify_audit_export.py <export-file> <manifest-file>`. It uses no database or network, checks metadata/structure and digest, and exits 0=`VALID`, 1=`INVALID`, 2=`UNSUPPORTED`. Canonicalization sorts object keys recursively and uses compact UTF-8 JSON while preserving array order; CSV punctuation and JSON whitespace are not hashed. A manifest digest is a local consistency check, not proof of authenticity or tamper-proof storage. See [Offline audit export verification](evaluation-audit-export.md).

Start with `uvicorn api.main:app --reload --port 8000`; interactive OpenAPI is at `/docs`.

- `GET /health` reports that the API process is responding and returns the installed application version. It does not claim models are usable.
- `GET /ready` returns `200` only when runtime files and complete, mutually consistent classifier/regressor metadata exist and both model artifacts can be deserialized with the expected prediction methods; otherwise it returns a sanitized `503`.
- `GET /model-info` distinguishes `project_version` (running application package) from `model_version` (classifier training metadata). `model_loaded` also requires the supported `LOW`/`HIGH` classifier labels; `regression_model_loaded` requires a deserializable regressor and valid, consistent regression metadata. These fields do not describe scientific model quality.
- `GET /model-artifacts/inventory` returns read-only classifier/regressor availability, recorded training/evaluation metadata, current and recorded SHA-256 fingerprints, and compatibility status. It never retrains or modifies artifacts; see the [ML artifact inventory guide](model-artifact-inventory.md).
- `GET /model-runs/comparison` compares persisted candidate evaluation rows within the classifier or regressor task, reports selected-model metric deltas, model/dataset/artifact provenance and shared temporal windows, and marks evidence `VALID`, `PARTIAL`, `INCOMPATIBLE`, or `UNAVAILABLE`. It never compares classification metrics with regression metrics. Only the most recent candidate tables are retained, so it does not provide an archive of historical training runs. See the [model run comparison guide](model-run-comparison.md).
- `GET /drift/monitor?evaluation_split=validation|test` compares allowlisted synthetic training features with the selected chronological synthetic evaluation partition and returns per-feature PSI/total-variation, missingness, unseen-category, fingerprint, and threshold data. The endpoint computes without writing. `scripts/generate_drift_report.py --evaluation-split validation|test` explicitly persists a versioned JSON report. Targets/future columns and official MinTIC records are excluded; see [Synthetic drift monitoring](synthetic-drift-monitoring.md).
- `GET /architecture/diagnostics` returns component responsibilities and sanitized status checks. Unconfigured/unreachable services and unavailable data are distinguished from verified failures. It does not inspect Docker daemon state; see the [architecture guide](architecture.md).
- `GET /observability/diagnostics` returns process-local main API request/error totals and per-route HTTP statuses/durations, plus an assistant availability/readiness check and aggregate metrics when reachable. The assistant exposes `GET /observability/metrics` for that bounded internal check. Counts reset when each API process restarts; request content and sensitive headers are not stored. See [Observability](observability.md).
- `GET /model-diagnostics` returns the versioned holdout and expanding-window report. It returns `503` when the report is absent, invalid, or stale against the current package, dataset, or model artifact fingerprints. See [Model diagnostics](model-diagnostics.md).
- `POST /evaluation/temporal` runs 2–5 bounded expanding chronological fits and selected feature distribution comparisons on synthetic data. Optional settings are `min_train_rows` (2–50,000), `min_test_rows` (2–10,000), and 1–12 unique model input `features`; defaults are 100, 20, and the first 12 features. Use `?report_format=markdown` for a readable summary. The JSON response is versioned and includes dataset/model fingerprints and explicit limitations. See [Temporal evaluation](temporal-evaluation.md).
- `GET /locations` returns the compact synthetic AP/zone/coordinate catalog.
- `POST /predict` preserves the V2 40-field classification contract. Extra fields and either target are rejected. Its `model_version` and `selected_model` identify the loaded classifier artifact using its training metadata; `/health` and `/model-info.project_version` identify the running application package.
- `POST /scenario/predict` accepts a date/time and one synthetic location, then builds context and returns classification, regression, conformal interval, capacity proxy, and sensitivity factors.

Scenario example:

```json
{
  "datetime": "2026-09-30T18:00:00-05:00",
  "location": {"zone": "downtown"}
}
```

Location may identify `zone`, `zone_id`, `access_point_id`, or paired `latitude` and `longitude`. Optional advanced context is typed and allowlisted. Invalid locations or missing earlier rows for a replay are rejected with structured errors.

## Assistant API (port 8001)

Start with `uvicorn assistant_api.main:app --reload --port 8001`; OpenAPI is at `/docs`.

- `GET /health` reports deterministic provider process status; `llm_enabled` is false. `/ready` checks runtime files and model-bundle loadability.
- `GET /suggestions` returns example questions.
- `POST /chat` accepts `message`, optional `session_id`, and structured scenario `context`.
- `POST /dataset/query` accepts allowlisted metrics, columns, filters, and groupings.

See [Assistant](assistant.md), [Scenario Prediction](scenario_prediction.md), and [Limitations](limitations.md) for behavior and interpretation.

## Official connectivity (V7.5.0)

- `GET /connectivity/sources` returns registered official sources and their integration status.
- `GET /connectivity/tunja` returns the latest valid locally stored MinTIC snapshot's municipal analysis, freshness, period, and provenance. It returns 503 when no valid snapshot exists.
- `GET /connectivity/refresh/latest` returns the latest attempt, latest successful attempt, and local snapshot state as separate fields. Snapshot state is always labeled `cached_local`, `stale_cached`, or `unavailable`; it is never a live-data claim.
- `GET /connectivity/refresh/history?limit=20&offset=0` returns newest-first refresh attempts with total count and offset pagination (`limit` 1–100, `offset` 0 or greater). Each row includes timestamp, source, success/failure, retrieved row count when known, reporting period, validation result, sanitized error type, and associated snapshot batch ID.
- `GET /connectivity/tunja/compare?baseline_period=2025-Q1&comparison_period=2026-Q1` compares two periods present in the validated snapshot. Its result includes reported-access total, provider and technology totals, and downstream/upstream band categories with absolute and mathematically valid percentage differences. Missing periods return `period_unavailable`; missing categories are not zero-filled and zero baselines have no percentage difference. The response includes the snapshot freshness and source provenance.
- `GET /connectivity/tunja/quality` reports retrieved and accepted row counts, missing required source fields, exact duplicate rows, observed reporting-period coverage, incomplete periods, per-indicator availability, ingestion validation status, and source provenance. Empty values are reported as missing rather than zero; an unavailable snapshot returns 503. Snapshot/cache status is included and is never represented as a live source check.
- `GET /connectivity/tunja/concentration?reporting_period=2025-Q1` reports provider shares, leading provider and share, provider count, and the 0–10,000 HHI for one available period. It uses preserved unique MinTIC source rows, includes snapshot provenance, and returns explicit unavailable states for incomplete/invalid or zero-total inputs. The measure describes fixed-internet subscription counts only.
- Refresh is controlled by running `python scripts/refresh_official_data.py`; no public mutation endpoint was added.

Only MinTIC fixed-access records from `hwrp-3su3` are integrated. Refresh attempts are stored in an additive table in the existing local SQLite snapshot database; legacy databases without history return an empty history until a refresh is attempted. A failed refresh preserves the last valid snapshot. See [official connectivity data](official-connectivity.md) for source methodology, limitations, and synthetic/real separation.

Both FastAPI applications reject request bodies larger than their documented service limit (256 KiB for the main API and 64 KiB for the assistant API), return sanitized validation errors, and add `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and `Permissions-Policy`. CORS allows the local Streamlit origins by default; set `CORS_ALLOWED_ORIGINS` to a comma-separated list of explicit browser origins for another deployment. Credentials are disabled. See [Security](security.md) for the baseline and its limitations.
