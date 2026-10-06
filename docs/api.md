# API reference

## Main API (port 8000)

Start with `uvicorn api.main:app --reload --port 8000`; interactive OpenAPI is at `/docs`.

- `GET /health` returns project/version status.
- `GET /model-info` reports classifier and regressor artifacts plus temporal periods.
- `GET /locations` returns the compact synthetic AP/zone/coordinate catalog.
- `POST /predict` preserves the V2 40-field classification contract. Extra fields and either target are rejected.
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

- `GET /health` reports deterministic provider status; `llm_enabled` is false.
- `GET /suggestions` returns example questions.
- `POST /chat` accepts `message`, optional `session_id`, and structured scenario `context`.
- `POST /dataset/query` accepts allowlisted metrics, columns, filters, and groupings.

See [Assistant](assistant.md), [Scenario Prediction](scenario_prediction.md), and [Limitations](limitations.md) for behavior and interpretation.

Both FastAPI applications reject request bodies larger than their documented service limit (256 KiB for the main API and 64 KiB for the assistant API), return sanitized validation errors, and add `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, and `Permissions-Policy`. CORS allows the local Streamlit origins by default; set `CORS_ALLOWED_ORIGINS` to a comma-separated list of explicit browser origins for another deployment. Credentials are disabled. See [Security](security.md) for the baseline and its limitations.
