# Release history and V4.0 readiness

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

## Historical tag inventory audit (2026-10-19)

The public GitHub Releases API returned zero releases; a Git tag is therefore not evidence of a published GitHub Release. Remote branch `main` matched local `main` at `c6f04704228571e1fb785bf95b36180964938262` before V4.3 changes. The repository's explicit release history and package-version commits support repairing two missing annotated tags; both are pointed to their historical release commits, never to current `HEAD`.

| Version | Tag before audit | Intended commit | Evidence and audit action | GitHub Release |
| --- | --- | --- | --- | --- |
| V3.0.0 | `v3.0.0` local and remote | `180471113ff48ac73c2adfe8f7533111264b23b3` | Existing tag preserved. | None observed. |
| V3.1.0 | Missing before audit; now present locally/remotely as annotated `v3.1.0` | `504af22dfca574b3bee2f905dcbd202605087288` | Changelog records `[0.3.1]`, this commit bumps package to `0.3.1` and documents V3.1; created and pushed annotated tag at this exact commit. | None observed. |
| V3.2.0–V3.5.0 | Existing local and remote | Existing annotated tag targets | Tags preserved. | None observed. |
| V3.6.0 | Missing before audit; now present locally/remotely as annotated `v3.6.0` | `42b5d50813ac114d88c95ed41872c5ecd7333689` | Commit title/body explicitly releases V3.6, changelog records `[0.3.6]`, and package changes to `0.3.6`; created and pushed annotated tag at this exact commit. | None observed. |
| V3.7.0–V3.9.0 | Existing local and remote | Existing annotated tag targets | Tags preserved. | None observed. |
| V4.0.0 | Missing local and remote | Unresolved | `78e0a3c` says “release V4.0 improvements”, but its resulting changelog still labels V4.0 preparation/unreleased and package/image remain `0.3.8`; insufficient evidence for a release tag. No tag created. | None observed. |
| V4.1.0 | Existing local and remote | `edc10f03b898217d7c0108115d0d111e53962d86` | Existing tag preserved; runtime/CI evidence remains as recorded in its validation file. | None observed. |
| V4.2.0 | Missing local and remote | Candidate commit `c6f04704228571e1fb785bf95b36180964938262` | Explicitly documented as an implementation candidate with Docker and Actions gates pending; no tag created. | None observed. |
| V4.3.0 | Not created | Candidate release commit not yet determined | Docker runtime and remote Actions gates are pending; no stable tag created. | None observed. |
