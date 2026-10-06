# V3.5 security baseline

This is a practical hardening baseline for a synthetic-data portfolio application. It reduces accidental exposure and malformed-input risk; it is not a production security certification.

## Implemented controls

- The main API accepts request bodies up to 256 KiB; the assistant API accepts up to 64 KiB. Oversized bodies receive HTTP 413.
- Prediction contracts reject extra fields and constrain feature types and numeric ranges. Assistant messages are limited to 1,000 characters; whitespace-only messages, unsafe session identifiers, unsupported dashboard sections, malformed selectors, and oversized scenario context are rejected.
- Scenario-result context is treated as untrusted explanatory input. Its envelope and nested JSON size, depth, values, and numeric magnitude are bounded; it does not invoke models or select files.
- Dataset queries keep fixed metric, column, grouping, and filter allowlists. Filter values use scalar types and supported categorical values or ranges. No query path evaluates Python, SQL, or user expressions.
- `CORS_ALLOWED_ORIGINS` accepts a comma-separated list of explicit HTTP(S) origins. The default permits local Streamlit origins on port 8501. Wildcards and credentials are not enabled. A deployed browser client must configure its exact origin.
- Both APIs add `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: strict-origin-when-cross-origin`, and a restrictive `Permissions-Policy`. No CSP is applied, so the built-in Swagger UI remains usable.
- Request-validation and unexpected tool errors return fixed messages rather than submitted values, stack traces, or local paths. Error details are retained only in server-side logs where needed for diagnosis.
- Assistant sessions are process-local, expire after one hour, and are capped at 1,024 entries. The service retains structured context, not the chat transcript.
- `.env`, `.env.local`, model artifacts, datasets, and Streamlit secrets are excluded from version control by `.gitignore`; `.env.example` contains no credentials.

## Configuration

For local use, the default CORS origins cover Streamlit at `http://localhost:8501` and `http://127.0.0.1:8501`. For a different browser origin, set for example:

```text
CORS_ALLOWED_ORIGINS=https://dashboard.example.org
```

Multiple explicit origins can be comma-separated. Do not set this value to `*`. Keep real environment files and credentials out of source control.

## Threats considered

The controls address malformed or oversized chat/API input, unbounded scenario context, unsupported dataset dimensions and filters, cross-origin browser calls, accidental reflection of validation input, and accidental public exposure of stack traces or filesystem paths. Model artifact paths remain fixed by application configuration; request fields do not select model files. Existing location resolution maps user-supplied places only through the synthetic catalog. There is no SQL layer, shell invocation, `eval`, or dynamic dataframe expression interface.

## Not provided

This release does not provide authentication, authorization, user accounts, enterprise identity, production-grade or distributed rate limiting, persistent audit logging, a WAF, TLS termination, deployment secret management, or a penetration-tested production deployment. The in-memory session cap is not a substitute for rate limiting. Deployments exposed beyond local development need an appropriate reverse proxy, TLS, identity/access controls, monitoring, abuse controls, and operational review. The project remains synthetic and does not provide live municipal WiFi data or production forecasting.
