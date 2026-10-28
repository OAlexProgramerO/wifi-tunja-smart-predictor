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

Start with `uvicorn api.main:app --reload --port 8000`; interactive OpenAPI is at `/docs`.

- `GET /health` reports that the API process is responding and returns the installed application version. It does not claim models are usable.
- `GET /ready` returns `200` only when runtime files and complete, mutually consistent classifier/regressor metadata exist and both model artifacts can be deserialized with the expected prediction methods; otherwise it returns a sanitized `503`.
- `GET /model-info` distinguishes `project_version` (running application package) from `model_version` (classifier training metadata). `model_loaded` also requires the supported `LOW`/`HIGH` classifier labels; `regression_model_loaded` requires a deserializable regressor and valid, consistent regression metadata. These fields do not describe scientific model quality.
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

Both FastAPI applications reject request bodies larger than their documented service limit (256 KiB for the main API and 64 KiB for the assistant API), return sanitized validation errors, and add `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and `Permissions-Policy`. CORS allows the local Streamlit origins by default; set `CORS_ALLOWED_ORIGINS` to a comma-separated list of explicit browser origins for another deployment. Credentials are disabled. See [Security](security.md) for the baseline and its limitations.
