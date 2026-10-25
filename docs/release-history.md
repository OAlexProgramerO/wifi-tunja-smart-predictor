# Release history and V4.0 readiness

# V6.0.0 — Geospatial intelligence and what-if simulation — released 2026-10-25

Adds historical synthetic geospatial analysis, a deterministic multi-zone what-if workflow, persisted comparison records, and scenario-aware reports. Product version `6.0.0` maps to Python package and Docker image `0.6.0`. Annotated tag [`v6.0.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.0.0) points to release commit `39c10c450d205a3cd9b70f9119d99db291c51cd1`; the [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.0.0) is published. Local tests and quality checks passed, and both jobs passed in [workflow 40240295823](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40240295823). The Docker daemon was unavailable locally; remote CI performed container validation. Synthetic observations and outputs do not validate real public WiFi demand or service in Tunja. See [the V6.0.0 validation record](releases/v6.0.0.md).

## V5.3.0 — Experiment observability and auditability — released 2026-10-24

Adds derived experiment integrity status, reason-coded comparison failures, safe history warnings, report validation, and bilingual dashboard status. Product V5.3.0 uses package and Docker image `0.5.3`. Annotated tag [`v5.3.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v5.3.0) points to release commit `20fb2c6e140ce774843abbb9449c4d0889d39227`; the [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v5.3.0) is published. Local checks and both remote CI jobs passed in [workflow 40138776914](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40138776914). All experiment values remain synthetic. See [the V5.3.0 validation record](releases/v5.3.0.md).

## V5.2.0 — Engineering upgrade — released 2026-10-23

Adds explicit comparison interpretation for compatible experiments with or without shared synthetic zone IDs, plus bounded page navigation in Experiment History. Product V5.2.0 uses Python package and Docker image `0.5.2`. Annotated tag [`v5.2.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v5.2.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v5.2.0) point to release commit `64c4d9369dbe8efbf69e808d2cc82616f60b8dfd`. Local checks and remote CI passed; Docker runtime, HTTP smoke, and persistence passed in [workflow 40039016215](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40039016215). All data and metrics remain synthetic. See the [V5.2.0 validation record](releases/v5.2.0.md).

## V5.1.0 — Stabilization Patch — released 2026-10-22

Fixes experiment database connection cleanup, rejects malformed persisted comparison provenance/metrics safely in the API and assistant, makes API/assistant default experiment storage independent of the process working directory, and refreshes stale evaluation reports against an integrity-validated existing model bundle without retraining. It also fixes Docker experiment-volume permissions and classifier/regressor fingerprint matching. Product V5.1.0 uses Python package and Docker image version `0.5.1`. The annotated tag [`v5.1.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v5.1.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v5.1.0) were published separately. See the [V5.1.0 validation record](releases/v5.1.0.md). All observations and predictions remain synthetic.

## V5.0.0 — implementation candidate; not released

Adds deterministic synthetic capacity planning, multi-zone scenario execution through the existing model service, SQLite experiment persistence, API retrieval and reporting, and dashboard experiment history. The package and image candidate version is `0.5.0`. Docker runtime, security audit, and remote CI require fresh verification. No V5.0.0 tag or GitHub Release is claimed. All analysis remains synthetic. See the [V5.0.0 validation record](releases/v5.0.0.md), [capacity-planning guide](capacity-planning.md), and [experiment guide](experiments.md).

## V4.4.0 — implementation candidate; not released

Adds a provenance-checked diagnostics report with chronological expanding-window fits, an additive API endpoint, Model Performance views, and deterministic assistant metric answers. Python package and Docker image candidate versions are `0.4.4`. Docker/runtime and remote CI gates must be verified before a stable tag or GitHub Release. Metrics remain limited to synthetic data. See [V4.4.0 validation record](releases/v4.4.0.md) and [diagnostics documentation](model-diagnostics.md).

## V3.8 to V3.9: changes confirmed by Git history

`v3.8.0` points to `e935b33`; `v3.9.0` points to `16128ff` and is present on both the local repository and `origin`. The V3.9 range contains eight commits and changes 31 files (3,296 insertions, 166 deletions). The classifier, regressor, synthetic dataset methodology, scenario logic, and dashboard product sections predate V3.9; they were not introduced by the Docker work.

| V3.9 change | Evidence in the repository | Verification and remaining limit |
| --- | --- | --- |
| Compose application image and services | `Dockerfile`, `compose.yaml`; commits `af035aa` and `e23ae20` | `docker compose config --quiet` passed in this V4.0 audit. Building/running needs an active Docker engine. |
| Synthetic runtime initialization and persistent data/model/report volumes | `scripts/initialize_runtime.py`, `compose.yaml`; initializer checks required file presence, runs the existing generator/preparation/training scripts once, and refuses partial resources | Unit tests cover complete, partial, and failed initialization. This is file-presence idempotency, not validation of artifact provenance or content hashes. |
| Health and readiness coordination | At V3.9, `api/routes/prediction.py`, `assistant_api/main.py`, `src/wifi_tunja_smart_predictor/deployment.py` checked runtime file presence. | V4.0 extends readiness to deserialize both models and validate complete, matching feature/version metadata; tests cover corrupt model files and version mismatches. |
| Compose-aware dashboard API clients and HTTP smoke tool | `src/wifi_tunja_smart_predictor/api_client.py`, `scripts/smoke_test.py`, dashboard route changes | Unit tests cover clients and smoke criteria; the live container-network smoke test needs Docker. |
| Dependency source/lock and CI | `pyproject.toml`, requirements wrappers, hash-locked `requirements.lock`, `.github/workflows/ci.yml`; commits `a565bbf`, `01f70a3` | CI defines a build/start/smoke/restart persistence job. The workflow file alone does not prove a successful remote run. |
| Deployment, troubleshooting and validation documentation | READMEs, `docs/deployment.md`, architecture/API docs and the V3.9 acceptance plan | Documentation accurately marks Docker runtime checks pending for this audit session. |

The commit sequence (`0d3ca1f` through `16128ff`) shows that V3.9 progressed through planning, Compose implementation, documentation, dependency standardization, smoke/CI checks, and a final limitations update. The V3.9 tag does not itself prove every checklist item passed.

## Why GitHub showed V3.8

The source of that label was found in both default-branch README files: they explicitly said “Current release: 0.3.8 (V3.8)” / “Versión actual: 0.3.8 (V3.8)” despite the later V3.9 tag. Those lines are corrected. A separate check of GitHub's public Releases page on 2026-10-13 showed “There aren’t any releases here.” Therefore the observed state is:

| Version surface | Observed value | Meaning |
| --- | --- | --- |
| Local and remote Git tag | `v3.9.0` → commit `16128ff` | Tag exists and is pushed to `origin`. |
| GitHub Release | None listed | No release page/package was published through GitHub Releases. |
| Python package | `0.3.8` in `pyproject.toml` and package metadata | Installed application version, not a Git tag. |
| Docker image | `wifi-tunja-smart-predictor:0.3.8` in Compose | Local image tag; it is not evidence of a registry push. |
| README landing-page label (before this update) | V3.8 / 0.3.8 | Stale prose, corrected to distinguish project tag from package/image and GitHub Release. |
| Automated publishing/CI evidence | `.github/workflows/ci.yml` defines test and container-smoke jobs, but not a release/image-publish workflow. GitHub Actions displayed “There are no workflow runs yet” when checked. | The checked-in workflow is configured, but no public run result was available to confirm it passed. There is no automatic GitHub Release or registry publication configured. |

The GitHub webpage evidence is the public [Releases page](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases). `gh` is not installed in this environment, so the page was checked directly; remote Git references were separately checked with `git ls-remote --tags origin`. No existing tag was moved or removed.

## V4.0 audit and chosen scope

The review found a user-facing traceability defect: `POST /predict` returned the installed package version as `model_version`, while its event log used the persisted classifier's training metadata and scenario predictions used that metadata. Persistent model volumes deliberately survive image rebuilds, so those versions can differ. The response now reports only the version recorded in valid classifier metadata; missing, malformed, incomplete, or inconsistent metadata makes prediction unavailable with a sanitized `503` rather than substituting the application version.

The readiness endpoints also previously treated the presence of files as proof that the application could use its models. V4.0 now deserializes both artifacts, checks their prediction methods, and validates the feature contract and consistency between combined, classifier, and regressor metadata. `/health` remains a liveness check; `/ready` checks application/model readiness; `/model-info` reports classifier loadability and artifact version separately from the running package version.

The ignored local artifacts provide direct evidence of the mismatch: model metadata records project version `0.3.0`, classifier `random_forest`, regressor `random_forest_regressor`, dataset `synthetic_seed_42_v1`, and training time `2026-09-30T15:53:25-05:00`; the installed package reports `0.3.8`. A local FastAPI `TestClient` prediction against that persisted classifier returned HTTP 200 and `model_version: 0.3.0`, while `/health` and `/model-info` returned application version `0.3.8`.

The checked-in local metric artifacts describe a 7,574-row synthetic temporal test set from 2025-10-01 through 2025-12-31: classification F1 for `HIGH` 0.9049 and ROC-AUC 0.9804; regression MAE 27.906, RMSE 52.860, and R² 0.8358. The nominal 90% interval's empirical synthetic holdout coverage is 0.8957. These are stored results from the artifact generation dated 2026-09-30, not metrics regenerated or independently reevaluated for V4.0 and not evidence of real-world WiFi performance.

| Functionality | Confirmed issue | Chosen improvement | Acceptance check |
| --- | --- | --- | --- |
| Standard prediction response | `model_version` could describe a newer app image than the persisted model; absent/bad metadata could be hidden by a version fallback | Return only the recorded artifact `project_version`; reject unavailable or inconsistent metadata | Tests cover a different model version plus missing, malformed, and incomplete metadata. |
| Readiness and model status | File presence could mark an unreadable or metadata-inconsistent model bundle ready | Deserialize both artifacts and validate metadata before readiness succeeds; preserve `/health` as process liveness | Tests cover corrupt artifacts, mismatched versions, and readiness failure while health remains responsive. |
| Release/version documentation | README called V3.8 current even though `v3.9.0` exists remotely; GitHub Releases is empty | Correct both READMEs, changelog, deployment note; add evidence matrix and release plan | Verify local/remote refs, page state, package/image declarations, and docs agree. |
| Docker onboarding | Deployment reference assumes prior familiarity with volumes, health/readiness, Desktop/WSL2, and resource inspection | Add a junior developer guide using this repository's actual service names and safe commands | Documentation names working Compose commands and warns that `down -v` deletes persistent demo data. |

V4.0 remains an unreleased working label. The package stays `0.3.8` and image stays `0.3.8` until the required acceptance run is complete. If accepted, keep Python's pre-1.0 SemVer package version as `0.4.0`, use project Git tag `v4.0.0`, and build image `wifi-tunja-smart-predictor:0.4.0`; publish a GitHub Release only after explicit authorization. Never overwrite `v3.9.0`.

## Pending release acceptance

During this audit, `python -m pytest -q` passed all 179 tests; Ruff, Black, compileall, `docker compose config --quiet`, and `git diff --check` also passed. The existing `scripts/smoke_test.py` was run but failed because the main API, assistant API, and dashboard were not running locally. The Docker CLI and Compose are installed, but the Docker Desktop Linux engine pipe is unavailable, so image build, isolated initialization, service health, container-network smoke, and persistence after recreation remain unverified. Dependency auditing with `uvx --from pip-audit pip-audit -r requirements.lock` could not complete because the Linux-targeted lock attempts to build `uvloop`, which does not support this Windows host. No vulnerability-free result is claimed. GitHub Actions has no observed run for this checkout.

Before assigning a V4.0 package version or release tag, start Docker Desktop and run the documented build/start/smoke/recreate checks from [deployment](deployment.md), without deleting any existing volume. Run a dependency scan from a compatible Linux environment and inspect an actual CI run. No release, image push, tag, or publication was performed as part of this work.

## V4.1 implementation status

V4.1.0 work adds bounded reliability fixes without changing the classifier/regressor algorithms, training split, synthetic data, model metrics, or dashboard sections:

- The classifier cache now includes a file signature in its cache key, so a replaced model artifact is loaded again instead of continuing to serve an estimator cached only by path.
- Model readiness and `/model-info` require the classifier's supported `LOW`/`HIGH` classes. Regression status also validates its feature metadata and consistency with the combined model record.
- The one-time runtime initializer validates an already complete model bundle. If existing artifacts fail validation, it exits unsuccessfully and leaves every persistent resource untouched.
- Unexpected prediction errors retain bounded stack-frame fields in structured diagnostics without exposing exception details in the HTTP response.

Regression tests cover artifact replacement, invalid classifier labels, missing/malformed/incomplete regression metadata, corrupt regression/classifier artifacts, inconsistent nested regression contracts, safe initialization refusal, model-info consistency, and sanitized unexpected errors. The ML pipeline was reviewed: `temporal_split` sorts and partitions by timestamp, model pipelines fit on training rows, and test rows are used for reporting rather than model selection. This review does not establish real-world performance; all data and metrics remain synthetic.

### V4.1 verification record

See the dedicated [V4.1.0 implementation and validation record](releases/v4.1.0.md) for its exact commands and results. The annotated Git tag `v4.1.0` now points to commit `edc10f03b898217d7c0108115d0d111e53962d86`; the public GitHub Releases API still lists no published releases. The V4.1 package/image baseline was `0.3.8`; Docker runtime checks and an actual CI run were not independently confirmed for that cycle. The tag does not change those historical verification results.

## V4.2.0 implementation candidate

V4.2.0 adds exact-byte model fingerprints and prediction trace metadata, a shared training-run identity for classifier/regressor bundles, stricter dataset and metric validation, clearer classification/regression presentation, and expanded CI checks. The project label is `V4.2.0`; the Python package and Compose image candidate use `0.4.2`. Local quality checks pass, while live Docker validation and remote Actions results remain pending. No V4.2.0 tag or GitHub Release exists. See the [V4.2.0 validation record](releases/v4.2.0.md) for exact evidence and release gates. All model metrics and data remain synthetic.

## V4.3.0 implementation candidate

V4.3.0 records SHA-256 and byte length in newly trained model metadata and verifies those values against exact deserialized bytes before inference, readiness, and evaluation. The Python package and Compose image candidate use `0.4.3`. Docker runtime and remote Actions remain unverified; no V4.3.0 tag or GitHub Release exists. See [the V4.3.0 validation record](releases/v4.3.0.md). All datasets and evaluation metrics remain synthetic.

## Historical tag inventory audit (2026-10-21)

The public GitHub Releases API returned zero releases; a Git tag is therefore not evidence of a published GitHub Release. Remote branch `main` matched local `main` at `c6f04704228571e1fb785bf95b36180964938262` before V4.3 changes. The repository's explicit release history and package-version commits support repairing two missing annotated tags; both are pointed to their historical release commits, never to current `HEAD`.

| Version | Intended/corresponding commit | Local tag | Remote tag | GitHub Release | Evidence / action |
| --- | --- | --- | --- | --- | --- |
| V3.0.0 | `180471113ff48ac73c2adfe8f7533111264b23b3` | `v3.0.0` | `v3.0.0` | None observed | Existing tag preserved. |
| V3.1.0 | `504af22dfca574b3bee2f905dcbd202605087288` | `v3.1.0` | `v3.1.0` | None observed | Changelog `[0.3.1]` and this commit's package bump/documented release identify the commit; created and pushed the missing annotated tag here. |
| V3.2.0 | `1a6a522703518c9655645a9d487d866d60e79553` | `v3.2.0` | `v3.2.0` | None observed | Existing tag preserved. |
| V3.3.0 | `138fea6d7ea428c084e255aa676dbebe28dc3ba1` | `v3.3.0` | `v3.3.0` | None observed | Existing tag preserved. |
| V3.4.0 | `a2d9c019bf449eafc42d691db9403c23d7ca0865` | `v3.4.0` | `v3.4.0` | None observed | Existing tag preserved. |
| V3.5.0 | `0d337aaa7b5d47448f2bc9dc2e8c53bc6b60c830` | `v3.5.0` | `v3.5.0` | None observed | Existing tag preserved. |
| V3.6.0 | `42b5d50813ac114d88c95ed41872c5ecd7333689` | `v3.6.0` | `v3.6.0` | None observed | Release commit, changelog `[0.3.6]`, and package bump identify the commit; created and pushed the missing annotated tag here. |
| V3.7.0 | `b76797ca32ad3be2fb7814994f8ab78a6356b256` | `v3.7.0` | `v3.7.0` | None observed | Existing tag preserved. |
| V3.8.0 | `e935b339540c88910d827d8c16e818effe2ec276` | `v3.8.0` | `v3.8.0` | None observed | Existing tag preserved. |
| V3.9.0 | `16128ff72025ff62e00359a606a1c3b32081cabd` | `v3.9.0` | `v3.9.0` | None observed | Existing tag preserved. |
| V4.0.0 | `78e0a3c9c3043f7a8db2b3b1c9cd1bd8d67770da` (implementation commit only) | Missing | Missing | None observed | Unresolved: resulting changelog still says V4.0 preparation/unreleased and package/image remain `0.3.8`; no release tag created. |
| V4.1.0 | `edc10f03b898217d7c0108115d0d111e53962d86` | `v4.1.0` | `v4.1.0` | None observed | Existing tag preserved; runtime/CI verification limitations remain documented. |
| V4.2.0 | `c6f04704228571e1fb785bf95b36180964938262` (candidate) | Missing | Missing | None observed | Documented candidate with Docker and Actions gates pending; no tag created. |
| V4.3.0 | `53fb0749996938f692cbe0adbdbd0d09211cc0f8` (candidate on `main`) | Missing | Missing | None observed | Docker runtime and remote Actions gates are pending; no stable tag created. |
| V4.4.0 | `99a849796b01c67e4ccfa4f9403ad53745d154f7` (candidate) | Missing | Missing | None observed | Docker, security, and remote Actions gates are pending; no stable tag created. |
| V5.0.0 | `0107e883f6fff6b5681d92b413f950694c002c48` | Missing | Missing | None observed | Implementation candidate; local checks pass, but Docker runtime and Actions were not confirmed at that release audit. |
| V5.1.0 | `789478211d3b99295cf54030d7690187413b4644` | `v5.1.0` | `v5.1.0` | `39861122427` passed | Published after local tests, lint/format/security checks, and remote Docker startup, API smoke, and persistence checks passed; see the V5.1.0 validation record. |

The 2026-10-21 audit fetched `origin` before implementation: remote `main` was `e1af8606133dc7cbca80eba681b97ef90c16f53a`; local and remote agreed. Existing remote tags `v3.0.0`–`v3.9.0` and `v4.1.0` were preserved and their peeled commit targets reviewed. Historical gaps `v4.0.0`, `v4.2.0`, `v4.3.0`, and `v4.4.0` remain unresolved candidates, not missing releases to synthesize: prior records explicitly document unreleased status or incomplete gates. No existing or older tags were moved. The public GitHub Releases API listed no releases and the Actions API had no workflow runs at audit time.

The 2026-10-22 audit began from clean `main` at `0107e883f6fff6b5681d92b413f950694c002c48` with no `v5.1.0` tag. After fixes and successful verification, release commit `789478211d3b99295cf54030d7690187413b4644` was pushed, tagged, and published as a stable GitHub Release. The final Actions run `39861122427` passed both jobs for that commit.
