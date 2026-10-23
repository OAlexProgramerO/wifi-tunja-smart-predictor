# API reference

## V5 experiment routes

| Method and path | Behavior |
|---|---|
| `POST /experiments/scenarios` | Validate and run 1–12 synthetic zone estimates, apply deterministic capacity planning, and persist an experiment (201). |
| `GET /experiments?limit=50&offset=0` | Return a bounded history page (limit 1–100). |
| `GET /experiments/{experiment_id}` | Retrieve one saved experiment (404 when absent). |
| `POST /experiments/compare` | Compare 2–10 records only when time window, dataset identity, and model artifact fingerprints agree (409 otherwise). |
| `GET /experiments/{experiment_id}/report?format=json` | Export a structured report; `format=markdown` returns a readable report. |

The comparison is relative to the first supplied experiment and reports only shared zone IDs. Its additive `comparison_status`, `scientific_comparability`, `comparability_reasons`, and `interpretation_detail` fields distinguish a technically compatible comparison from one with no shared zones; the latter returns no zone deltas and is marked `not_assessable`. Shared-zone results remain limited synthetic comparisons, not evidence of real-world equivalence. All demand and capacity values are synthetic connection counts, not throughput or real-network advice.

## Main API (port 8000)

Start with `uvicorn api.main:app --reload --port 8000`; interactive OpenAPI is at `/docs`.

- `GET /health` reports that the API process is responding and returns the installed application version. It does not claim models are usable.
- `GET /ready` returns `200` only when runtime files and complete, mutually consistent classifier/regressor metadata exist and both model artifacts can be deserialized with the expected prediction methods; otherwise it returns a sanitized `503`.
- `GET /model-info` distinguishes `project_version` (running application package) from `model_version` (classifier training metadata). `model_loaded` also requires the supported `LOW`/`HIGH` classifier labels; `regression_model_loaded` requires a deserializable regressor and valid, consistent regression metadata. These fields do not describe scientific model quality.
- `GET /model-diagnostics` returns the versioned holdout and expanding-window report. It returns `503` when the report is absent, invalid, or stale against the current package, dataset, or model artifact fingerprints. See [Model diagnostics](model-diagnostics.md).
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
