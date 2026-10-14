# Changelog

## Unreleased — V4.0.0 preparation

### Fixed

- Standard classification responses now report the package version recorded in the trained artifact metadata, matching scenario predictions and prediction trace logs when persistent model volumes outlive the application image.
- Missing, malformed, incomplete, or inconsistent classifier metadata now prevents a prediction and returns a sanitized `503`; model metadata is checked against the persisted feature contract and the separate classifier record.
- API `/ready` now verifies that classifier and regressor artifacts can be deserialized and that their metadata versions agree. `/health` remains a process liveness check, while `/model-info` reports application and classifier artifact versions separately.

## V3.9.0 — One-Command Docker Setup & Reliability (Git tag)

### Added

- Python 3.12 runtime dependency lock with hashes, one-time runtime resource initialization using the existing synthetic data/preparation/training scripts, and persistent Compose-managed data, model, and report volumes.
- Compatibility requirements files now delegate to the `pyproject.toml` runtime/dev extras; CI and local development install the same project extra. The Starlette test client uses its supported `httpx2` dependency.
- Validated server-side API clients, a deployment smoke-test script, initialization and HTTP-client tests, and a Docker Compose CI validation job.

### Improved

- Compose now waits for runtime initialization and API health; API readiness covers the dataset, model artifacts and metadata, and dashboard comparison reports.
- Dashboard live scenario/location actions call the main API, and assistant chat calls the assistant API using loopback defaults locally and Compose DNS inside containers. The advanced feature-vector view retains the shared in-process inference path.
- English and Spanish quick starts describe the one-command Docker flow, persisted generated resources, health checks, recovery, and the synthetic-data limitations.
- Compose explicitly targets the Linux x86-64 architecture represented by the hash lock, including under ARM Docker Desktop emulation. The CI workflow now includes a persistence check across a Compose shutdown and restart.

### Version and verification status

- The remote `v3.9.0` tag points to commit `16128ff`; the Python package and Compose image remain `0.3.8`. The public GitHub Releases page has no published releases, so this tag is not a GitHub Release. The repository README previously called V3.8 current, which accounts for the stale version shown on the repository landing page.
- Docker CLI and Compose are installed in the current audit environment, but the Docker Desktop Linux engine pipe is unavailable. Configuration parsing is available; this session cannot verify image build, startup, container health, network smoke checks, or persistence. A prior successful Docker run was described in the project request, but its logs are not present in this checkout and could not be independently repeated.

## [0.3.8] - 2026-10-09

### Added

- Reusable Python 3.12 application image and Docker Compose services for the main API, assistant API, and Streamlit dashboard.
- Read-only mounts for local synthetic data and model artifacts; services run non-root with dropped capabilities and loopback-only host ports.
- Safe `/ready` endpoints and service health checks that verify required local data/model files without loading or retraining them.
- Readiness tests, deployment documentation, and Compose environment port examples.

### Improved

- Documented local/container setup, operational troubleshooting, and the dashboard's actual in-process shared-service topology in English and Spanish.
- Bumped package version to 0.3.8.

### Validation

- All 153 Python tests pass; Ruff, Black, Python compilation, Compose YAML parsing, and `git diff --check` pass.
- Docker CLI is unavailable in the implementation environment, so Compose CLI validation, image build, health checks, and container smoke tests remain unverified.
- Model behavior, artifacts, and synthetic data are unchanged; deployment checks do not establish real-world forecast validity.

## [0.3.7] - 2026-10-08

### Added

- Shared JSON application logging, request correlation through `X-Request-ID`, and monotonic request lifecycle timing for both APIs.
- Traceable standard/scenario prediction, assistant, historical, and dataset-query operation events with privacy-safe metadata.
- Observability tests for correlation, log parsing, privacy, error handling, and handler idempotence.

### Improved

- Safe diagnostic correlation across requests and service operations while retaining sanitized client errors.
- Bumped package version to 0.3.7.

## [0.3.6] - 2026-10-07

### Improved

- Assistant dataset queries reuse the assistant service's validated, normalized query engine instead of rebuilding and copying its full dataframe per request.
- Scenario explanations reuse analog baselines computed during scenario construction and avoid reconstructing/scanning the complete historical dataframe for each prediction.
- Feature sensitivity scoring reuses its object-converted feature row while isolating each model input row.
- Bumped package version to 0.3.6.

### Performance audit

- Added structural regression checks for query-frame reuse and single analog selection per prediction; no machine-specific timing thresholds or query-result cache were added.

## [0.3.5] - 2026-10-06

### Added

- Bounded request bodies, assistant messages, session identifiers, and scenario context.
- Restrictive configurable CORS defaults, API security headers, and sanitized validation errors.
- Security regression coverage for schemas, query filters, HTTP behavior, and CORS.

### Improved

- Dataset filters now validate supported scalar types, values, and ranges as well as allowlisted columns.
- Assistant context sections and session identifiers are constrained, while in-memory sessions remain bounded by count and TTL.
- Bumped package version to 0.3.5.

## [0.3.4] - 2026-10-05

### Added

- Deterministic historical analysis of average next-hour connections and LOW/HIGH demand rates.
- Historical zone and time-period comparisons, peak-hour and peak-weekday analysis, and month-over-month span summaries.
- English and Spanish historical question suggestions and focused assistant tests.

### Improved

- Historical questions reuse the allowlisted dataset query engine and remembered zone context without replacing scenario state.
- Demand Explorer assistant prompts are arranged in compact rows.
- Bumped package version to 0.3.4.

## [0.3.3] - 2026-10-03

### Added

- Dashboard-aware deterministic assistant explanations for all nine existing sections, in English and Spanish.
- Optional dashboard section and current scenario result in the existing chat context.
- Conceptual explanations for classification and regression metrics, preserving synthetic-data and non-causality limits.
- Dashboard-context assistant tests and section-aware prompt suggestions.

### Improved

- Streamlit assistant now passes the last visited dashboard section and displayed scenario result.
- Bumped package version to 0.3.3.

## [0.3.2] - 2026-10-02

### Added

- Context-aware deterministic English and Spanish demand questions with synthetic-zone aliases and simple time extraction.
- Follow-up demand queries reuse the conversation's saved location and call the existing scenario prediction service.
- Clarifying responses for missing zones, unknown synthetic zones, and invalid times.
- Demand scenario example prompts in the existing assistant page.

## [0.3.1] - 2026-10-01

### Added

- Deterministic English and Spanish greeting, identity, and capabilities intents in the assistant.
- Basic punctuation and surrounding whitespace normalization for conversational matches.
- Unit coverage for the requested conversation examples and matching behavior.

### Changed

- Bumped package version to 0.3.1.

## [0.3.0] - 2026-09-30

### Added

- Location-and-time scenario builder with deterministic synthetic historical analogs, replay safeguards, and nearest-AP geospatial resolution.
- Separate next-hour connection regressor alongside the backward-compatible LOW/HIGH classifier.
- Validation-calibrated split-conformal prediction intervals and capacity-use estimates.
- Deterministic tool-grounded assistant service on port 8001 and allowlisted aggregate dataset queries.
- Nine-section Streamlit navigation, scenario prediction, interactive synthetic AP map, and advanced feature form.
- Regression and scenario service tests, model metadata, and V3 documentation.

### Changed

- Model training/evaluation now reports classifier and regressor metrics on the existing temporal split.
- Removed generated HTML evaluation charts; evaluation writes compact PNG figures and JSON/CSV metrics.
- Set package version to 0.3.0.

## [0.2.0] - 2026-09-29

### Changed

- Redesigned the Streamlit experience with explicit overview, exploration, analysis, prediction, performance, and about navigation.
- Removed runtime import-path mutations and rely on the editable `src/` package installation.
- Updated model metadata and generated evaluation reports from the current temporal holdout.
- Improved API, README, Spanish README, and technical documentation contracts and synthetic-data disclosures.

### Fixed

- Corrected the expected-missingness test to edit an existing datetime-indexed row.
- Made temporal split tests use consistent timestamps and added mixed-format parsing in split logic.
- Resolved Ruff import ordering and Black formatting findings.
- Bumped application/package version to 0.2.0.

## [0.1.0] - 2026-09-29

### Added

- Synthetic dataset generation
- Data validation
- Feature engineering
- Classification baseline models
- FastAPI prediction API
- Streamlit dashboard
- Automated tests
- Technical documentation
