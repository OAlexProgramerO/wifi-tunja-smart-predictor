# Release history and V4.0 readiness

## V3.8 to V3.9: changes confirmed by Git history

`v3.8.0` points to `e935b33`; `v3.9.0` points to `16128ff` and is present on both the local repository and `origin`. The V3.9 range contains eight commits and changes 31 files (3,296 insertions, 166 deletions). The classifier, regressor, synthetic dataset methodology, scenario logic, and dashboard product sections predate V3.9; they were not introduced by the Docker work.

| V3.9 change | Evidence in the repository | Verification and remaining limit |
| --- | --- | --- |
| Compose application image and services | `Dockerfile`, `compose.yaml`; commits `af035aa` and `e23ae20` | `docker compose config --quiet` is locally checkable. Building/running needs an active Docker engine. |
| Synthetic runtime initialization and persistent data/model/report volumes | `scripts/initialize_runtime.py`, `compose.yaml`; initializer checks required file presence, runs the existing generator/preparation/training scripts once, and refuses partial resources | Unit tests cover complete, partial, and failed initialization. This is file-presence idempotency, not validation of artifact provenance or content hashes. |
| Health and readiness coordination | `api/routes/prediction.py`, `assistant_api/main.py`, `src/wifi_tunja_smart_predictor/deployment.py` | Readiness checks required file presence; inference and smoke requests load/use artifacts. Readiness itself does not prove artifacts can score a request. |
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

The review found one confirmed, user-facing traceability defect with a bounded, backward-compatible fix: `POST /predict` returned the installed package version as `model_version`, while its event log used the persisted classifier's training metadata and scenario predictions already used that metadata. Persistent model volumes deliberately survive image rebuilds, so those versions can differ. The response now reports the classifier artifact's recorded version (falling back to the package version when metadata lacks it), and a regression test demonstrates the differing-version case.

The ignored local artifacts provide direct evidence of the mismatch: model metadata records project version `0.3.0`, classifier `random_forest`, regressor `random_forest_regressor`, dataset `synthetic_seed_42_v1`, and training time `2026-09-30T15:53:25-05:00`; the installed package reports `0.3.8`. A local FastAPI `TestClient` prediction against that persisted classifier returned HTTP 200 and `model_version: 0.3.0`, while `/health` and `/model-info` returned application version `0.3.8`.

The checked-in local metric artifacts describe a 7,574-row synthetic temporal test set from 2025-10-01 through 2025-12-31: classification F1 for `HIGH` 0.9049 and ROC-AUC 0.9804; regression MAE 27.906, RMSE 52.860, and R² 0.8358. The nominal 90% interval's empirical synthetic holdout coverage is 0.8957. These are stored results from the artifact generation dated 2026-09-30, not metrics regenerated or independently reevaluated for V4.0 and not evidence of real-world WiFi performance.

| Functionality | Confirmed issue | Chosen improvement | Acceptance check |
| --- | --- | --- | --- |
| Standard prediction response | `model_version` could describe a newer app image than the persisted model that made the prediction | Return the artifact's training `project_version`, like scenario responses and trace events | Test with artifact metadata version `0.3.7` while the installed package is `0.3.8`; response must say `0.3.7`. |
| Release/version documentation | README called V3.8 current even though `v3.9.0` exists remotely; GitHub Releases is empty | Correct both READMEs, changelog, deployment note; add evidence matrix and release plan | Verify local/remote refs, page state, package/image declarations, and docs agree. |
| Docker onboarding | Deployment reference assumes prior familiarity with volumes, health/readiness, Desktop/WSL2, and resource inspection | Add a junior developer guide using this repository's actual service names and safe commands | Documentation names working Compose commands and warns that `down -v` deletes persistent demo data. |

V4.0 remains an unreleased working label. The package stays `0.3.8` and image stays `0.3.8` until the required acceptance run is complete. If accepted, keep Python's pre-1.0 SemVer package version as `0.4.0`, use project Git tag `v4.0.0`, and build image `wifi-tunja-smart-predictor:0.4.0`; publish a GitHub Release only after explicit authorization. Never overwrite `v3.9.0`.

## Pending release acceptance

On 2026-10-13, Docker CLI and Compose were installed, but the Docker Desktop Linux engine pipe was unavailable. Thus this session could not verify Docker build, clean initialization in isolated volumes, service health, in-container API networking, smoke test, or persistence after container recreation. A previous successful run was described in the task brief, but its logs are not part of the repository and the run could not be repeated. This is a verification limitation, not evidence that the earlier run failed.

Before assigning a V4.0 package version or release tag, start Docker Desktop and run the documented build/start/smoke/recreate checks from [deployment](deployment.md), without deleting any existing volume. Re-run pytest, Ruff, Black, and compileall on the release candidate. No release, image push, tag, or publication was performed as part of this work.
