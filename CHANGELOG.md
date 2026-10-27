# Changelog

## Unreleased — V6.2.0

Planned work: bounded one-factor synthetic demand sensitivity and budget sweeps for saved V6.1 allocation runs; per-zone allocation stability, explicit hard-minimum budget shortfall, and incomplete-search reporting; additive persistence and report schema 1.4; Scenario Workbench and API integration; bilingual documentation and verification. This is planned work, not a published release or verified outcome. All analyses remain limited to synthetic assumptions and do not establish real-world WiFi demand or reliability in Tunja. See the [V6.2 guide](docs/robustness.md).

## V6.1.0 — Intelligent scenario optimization — 2026-10-26

Adds deterministic bounded integer allocation optimization to the existing synthetic scenario workbench. It minimizes total or explicitly priority-weighted predicted connection deficits subject to per-zone bounds and a shared budget, saves through the existing SQLite experiment store, and exports through existing report routes. Product release V6.1.0 maps to Python package and Docker image `0.6.1`; annotated Git tag `v6.1.0` and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.1.0) target verified commit `29b0814`. Local pytest (278), Ruff, Black, and Compose configuration passed; both jobs, including Docker and locked dependency audit, passed in [workflow 40418511852](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40418511852). All estimates remain conditional on synthetic model outputs and do not validate real public WiFi demand in Tunja. See the [V6.1.0 validation record](docs/releases/v6.1.0.md).

## V6.0.0 — Geospatial intelligence and what-if simulation — 2026-10-25

Adds reusable filtered analytics of historical synthetic observations, an offline Plotly geospatial view, and a typed deterministic what-if comparison service integrated with the existing scenario predictor, capacity planner, experiment store, API, and JSON/Markdown report schema 1.2. A bilingual dashboard workbench saves baseline and alternative assumptions. Predicted demand remains fixed while capacity assumptions vary. Product release V6.0.0 uses package and Docker image `0.6.0`. The release candidate passed 265 local tests, Ruff, Black, Compose configuration, and both remote CI jobs (including Docker startup, API smoke and persistent-volume restart) in [workflow 40240295823](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40240295823). The local Docker daemon was unavailable. All data and model results remain synthetic and do not validate real public WiFi demand or infrastructure in Tunja. See the [V6.0.0 validation record](docs/releases/v6.0.0.md).

## V5.3.0 — Experiment observability and auditability

V5.3.0 adds derived experiment-integrity status and stable reason codes; safe history pagination that reports and omits unreadable persisted rows without hiding valid records; structured comparison diagnostics; and JSON/Markdown reports that state integrity, assessability, and missing historical provenance explicitly. The dashboard surfaces integrity warnings and only offers comparison for verified records. Package and Docker version `0.5.3` correspond to product release V5.3.0. See [the V5.3.0 validation record](docs/releases/v5.3.0.md). All experiment values remain synthetic and do not establish real-world demand validity.

## V5.2.0 — Engineering upgrade

### Summary

Improves experiment comparison interpretation and history navigation while preserving stored experiment formats, prediction behavior, and the synthetic-data scope.

### Added and improved

- Experiment comparison responses now explicitly distinguish matching technical provenance from limited scientific interpretation; comparisons with no shared synthetic zone IDs return the `not_assessable` marker and a reason while preserving existing fields and status codes.
- Experiment History now navigates bounded 25-record pages through the existing API and shows a clear empty-page state.

### Fixed and tested

- Added regression coverage for compatible/no-overlap comparison interpretation and dashboard page navigation. Local checks passed (240 pytest tests, Ruff, Black, compileall, and Compose configuration); GitHub Actions passed both code-quality and container-smoke jobs for tagged release commit `64c4d93` ([run 40039016215](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40039016215)). See [the V5.2.0 validation record](docs/releases/v5.2.0.md).

### Compatibility and limitations

- Product target V5.2.0 uses Python package and Docker image `0.5.2`; no API route or saved-experiment schema changed. Comparison fields are additive. All data and estimates remain synthetic and do not establish real Tunja WiFi forecast validity.

## V5.1.0 — Stabilization Patch — 2026-10-22

### Fixed

- Experiment-store SQLite connections now close reliably after reads and writes, preventing open database handles from lingering (including on Windows).
- Experiment comparison now rejects malformed saved provenance, model traces, and zone metrics with a safe conflict response; the deterministic assistant returns a clarification for malformed comparisons instead of raising an internal exception.
- API and assistant resolve the same default experiment database under the repository root, independent of the process working directory. Relative `EXPERIMENT_DB_PATH` values also resolve from the repository root.
- Runtime initialization refreshes stale versioned evaluation reports only after validating the existing model bundle, without retraining or replacing model artifacts.
- The Docker image now creates and grants the unprivileged application user ownership of the mounted experiment-volume directory, allowing first-start SQLite initialization.
- Runtime model validation now checks classifier and regressor fingerprints against their corresponding metadata records, so newly generated integrity-tracked bundles pass readiness validation.

### Improved

- Updated the Python package and Docker image versions to `0.5.1` for product release V5.1.0. Model artifacts and their training metadata were not replaced.
- Updated the hashed runtime lock to Streamlit `1.66.1` so it accepts patched Pillow `13.0.0`; removed the no-longer-required `toml` transitive pin.

### Tested

- Regression coverage added for connection closure, working-directory-independent shared storage, malformed API/assistant comparisons, refreshing reports for existing model volumes without retraining, and integrity-tracked classifier/regressor bundles. Final check results are recorded in [the V5.1.0 validation record](docs/releases/v5.1.0.md).
- Remote CI exposed a Streamlit/Pillow lock conflict, missing synthetic test data, a missing writable experiment-volume mount point, and incorrect classifier/regressor fingerprint matching. Each was corrected; the final workflow passed both jobs, including container startup, HTTP smoke tests, and volume persistence (see the validation record for run evidence).

### Known limitations and compatibility

- Local quality and dependency checks and remote Docker/runtime and CI gates passed. Git tag `v5.1.0` and the GitHub Release were published as distinct operations.
- No API schemas, saved experiment schema, model artifacts, or prediction algorithms were changed. All data, observations, predictions, and capacity estimates remain synthetic and are not verified against real Tunja WiFi demand.

## V5.0.0 implementation candidate (historical; not released)

- Added reproducible synthetic capacity-planning scenarios, deterministic resource-budget allocation, SQLite experiment history, paginated retrieval, and JSON/Markdown exports through the existing API.
- Added compatibility-checked scenario comparison, dataset/model fingerprints, and a date-filtered synthetic demand summary by geographic zone.
- Added an Experiment History dashboard workflow, deterministic bilingual assistant explanations for saved results, and a dedicated persistent Compose volume. All demand, capacity, and allocation values remain synthetic connection-count simulations.
- Updated the project/package/image candidate version to `0.5.0`. Docker runtime, security audit, and remote CI results must be verified before publication; no V5.0.0 tag or GitHub Release is claimed.

## Unreleased — V4.4.0

### Candidate changes

- Adds a versioned diagnostics report with synthetic held-out classification/regression metrics, class support and confusion matrix, regression bias/error quantiles, and three expanding chronological windows whose pipelines are fit only on earlier rows.
- Validates report schema, package version, dataset fingerprint, and classifier/regressor fingerprints; missing or stale reports are unavailable to the API, dashboard, and deterministic assistant.
- Adds `GET /model-diagnostics`, report-backed Model Performance views, and bilingual assistant answers sourced from the validated report. Runtime initialization now evaluates models and creates the diagnostics artifact.
- Project release V4.4.0 uses Python package and Docker image candidate version `0.4.4`. All evaluation remains synthetic. No V4.4.0 tag or GitHub Release is claimed; see [the V4.4.0 validation record](docs/releases/v4.4.0.md).

## Unreleased — V4.3.0

### Added

- Newly trained classifier and regressor metadata now records the SHA-256 and byte length of the exact serialized artifact.
- Prediction, readiness, and model evaluation reject partial or mismatched artifact fingerprints before using newly fingerprinted bundles; legacy bundles without fingerprint fields remain supported.
- Added regression tests for fingerprint validation, fail-closed readiness, and sanitized prediction failure on artifact mismatch.

### Version and validation status

- Candidate project version V4.3.0 uses Python package and Docker image version `0.4.3`. No V4.3.0 tag or GitHub Release has been created.
- Local and remote validation is recorded in [the V4.3.0 validation record](docs/releases/v4.3.0.md). Docker runtime and GitHub Actions remain pending; this entry does not claim a stable release.

## Unreleased — V4.2.0

### Added

- Prediction traceability metadata now identifies the UTC prediction time, correlation/request ID, application and model versions, and SHA-256/size of the exact classifier and regressor artifact bytes used.
- Training outputs now include a shared training-run ID so readiness and scenario inference reject classifier/regressor artifacts from different runs.
- Dataset and metric validation now rejects malformed numeric values, non-finite observations/predictions, unsupported timestamps, inconsistent per-access-point timestamps, malformed classifier probabilities, and incompatible metric shapes.
- CI now supports manual dispatch, Python compilation, and a scoped audit of the hash-locked runtime dependency set.

### Improved

- Scenario UI presents LOW/HIGH classification separately from the numeric next-hour connection estimate and calibrated interval, shows model provenance, and displays sanitized failure messages.
- Python package and Docker image candidate versions are `0.4.2` for project release V4.2.0. No V4.2.0 tag or GitHub Release has been created.

### Verification status

- 204 tests, Ruff, Black, compilation, Compose configuration parsing, and the scoped `pip-audit` scan passed locally. A relative Markdown-link check passed after the validation record was added.
- Live HTTP smoke could not run because services were not started. Docker runtime is blocked by the missing Docker Desktop Linux engine; GitHub Actions lists an active workflow but zero runs. This is not a formal V4.2.0 release. See [the V4.2.0 implementation and validation record](docs/releases/v4.2.0.md) for exact commands and remaining gates.

## V4.1.0 — implementation tag; no GitHub Release

### Fixed

- Classifier artifact caching now keys cached estimators by file path, modification time, and size so replacing a persisted artifact during a running process reloads the model before subsequent predictions.
- Readiness and `/model-info` now reject classifiers that do not expose the supported `LOW`/`HIGH` label contract; regression model status checks validated, mutually consistent metadata.
- Runtime initialization validates an already complete persistent model bundle and fails safely without overwriting it when artifacts are invalid.
- Unexpected prediction failures now retain bounded stack-frame diagnostics in structured logs while returning sanitized client errors.

### Tests

- Added regression coverage for classifier cache refresh, unsupported classifier labels, invalid regression metadata/artifacts, safe refusal of corrupted complete volumes, inconsistent combined metadata, and sanitized prediction failures.

### Verification status

- Full test suite: 191 passed. Ruff, Black (65 files unchanged), compileall, `docker compose config --quiet`, and `git diff --check` passed on 2026-10-17. `pip-audit` reported no known vulnerabilities for `requirements.lock` with `--disable-pip --no-deps`.
- Docker image build, service startup, runtime health/smoke, and volume persistence were unverified: Docker could not connect to `npipe:////./pipe/dockerDesktopLinuxEngine`. The remote annotated tag `v4.1.0` now points to `edc10f0`; no GitHub Release exists. The tag was created after the earlier validation record was written, so references below to the then-missing tag are superseded by the current repository state.
- The detailed evidence and release blockers are recorded in [docs/releases/v4.1.0.md](docs/releases/v4.1.0.md).

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
