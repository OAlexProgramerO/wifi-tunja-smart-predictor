# One-command Docker deployment (V3.9)

This setup runs the existing main FastAPI API, deterministic assistant API, and Streamlit dashboard from one application image. A fresh clone needs Docker Desktop or Docker Engine with Compose v2 running; it does not need host Python, local package installation, generated data, model files, or an `.env` file. Compose targets `linux/amd64`, matching the Python 3.12 Bookworm image and hash lock; Docker Desktop can emulate this target on ARM hosts.

> The dataset and all reported predictions are synthetic demonstrations. Container startup and synthetic holdout metrics do not validate real-world WiFi demand in Tunja.

## First run

```powershell
git clone https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor.git
cd wifi-tunja-smart-predictor
docker compose up --build
```

The first startup runs the `runtime-init` service. It uses the existing deterministic generator, preparation, and training scripts to create the raw and processed datasets, both model artifacts and metadata, and model-comparison reports. The initializer runs only when its named volumes are empty. The APIs and dashboard start after initialization completes; the dashboard waits for both APIs to pass their readiness checks.

Open these local URLs after Compose reports the services healthy:

- Dashboard: <http://localhost:8501>
- Main API documentation: <http://localhost:8000/docs>
- Assistant API documentation: <http://localhost:8001/docs>

The Compose ports bind to loopback. API calls made by the dashboard server use `http://main-api:8000` and `http://assistant-api:8001` on the Compose network. Browser access remains through the dashboard's localhost URL. Live scenario/location actions use the main API; assistant chat uses the assistant API. Dataset visualization and model-performance pages use the same shared read-only runtime files.

## Inspect, smoke-test, and stop

```powershell
docker compose ps
docker compose logs -f
docker compose exec -T dashboard python scripts/smoke_test.py
docker compose down
```

The smoke test checks both API health and readiness, classifier/regressor loading, a valid standard prediction, a scenario prediction, an assistant greeting, and dashboard HTTP health. Run it inside the dashboard container so requests to API service DNS names come from the Compose network, matching the dashboard's server-side HTTP clients. It exits nonzero for any failed check and avoids printing request bodies or full service diagnostics.

`docker compose down` stops/removes the containers but retains the generated `runtime-data`, `runtime-models`, and `runtime-reports` named volumes, so later starts do not repeat training. To reset only this project's generated synthetic demo resources, run:

```powershell
docker compose down -v
```

This removes the Compose project's named data/model/report volumes; it does not delete repository source files. The next `docker compose up --build` initializes those resources again.

## Runtime resources and initialization safety

Raw/processed CSV files, classifier and regressor artifacts, their metadata, and classifier/regressor comparison reports are generated into named volumes by one initialization service. They are not tracked in Git, copied from the developer machine, baked into the image, or mounted from host `data/`, `models/`, or `reports/` directories. App services use those volumes read-only. The root filesystem is read-only, the application runs as UID 10001, all Linux capabilities are dropped, and `no-new-privileges` is enabled.

The initializer checks the full required file set. If it finds a partial volume, it stops without overwriting existing files. When all files are present, it validates model deserialization, prediction methods, metadata feature contracts, and classifier/regressor version consistency before reporting the volume initialized. If existing model artifacts are invalid, initialization fails without modifying them. If generator/preparation/training fails, Compose dependency conditions prevent the APIs/dashboard from starting. Valid complete volumes are reused without retraining on normal restarts. There is no persistent logging volume.

## Configuration

Compose defaults are safe for a clean clone. A root `.env` is optional; `.env.example` has example values and no credentials. Compose interpolation checks the shell environment and optional `.env` before applying the defaults below.

| Setting | Compose default | Use |
| --- | --- | --- |
| `RANDOM_SEED` | `42` | Existing app seed setting. The existing generator uses its deterministic seed. |
| `LOG_LEVEL` | `INFO` | Existing log-level setting. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Existing explicit browser CORS origins. |
| `MAIN_API_HOST_PORT` | `8000` | Host-side loopback port; container port remains 8000. |
| `ASSISTANT_API_HOST_PORT` | `8001` | Host-side loopback port; container port remains 8001. |
| `DASHBOARD_HOST_PORT` | `8501` | Host-side loopback port; container port remains 8501. |
| `MAIN_API_URL` | `http://main-api:8000` in Compose; `http://127.0.0.1:8000` outside Compose | Dashboard server's main API destination. |
| `ASSISTANT_API_URL` | `http://assistant-api:8001` in Compose; `http://127.0.0.1:8001` outside Compose | Dashboard server's assistant API destination. |

Change a host port by creating/editing `.env`, for example `DASHBOARD_HOST_PORT=8510`. Internal service addresses continue using Compose service names and fixed container ports. The two API URL overrides must be absolute HTTP(S) origins without credentials, paths, query strings, or fragments. Invalid explicit values fail with a safe configuration message rather than silently switching destinations. Do not set `MAIN_API_URL` to `localhost` in the dashboard container.

When running the dashboard directly on the host, keep the local defaults and start both APIs as described below. The optional Compose URL values are for container networking; they are not the URLs the browser opens. Local API CORS continues using its existing configured-origin policy.

## Local Python development

The existing local workflow remains available and requires Python 3.11+. `pyproject.toml` is the authoritative dependency declaration. `requirements.txt` and `requirements-dev.txt` are compatibility wrappers (`-e .` and `-e .[dev]`) and do not carry a second dependency list. Development/test tools (including `httpx2` for Starlette's `TestClient`) are in the `dev` optional dependency group; they are excluded from the container runtime lock.

```powershell
python -m pip install -e ".[dev]"
python scripts/generate_synthetic_dataset.py
python scripts/prepare_data.py
python scripts/train_model.py
uvicorn api.main:app --reload --port 8000
uvicorn assistant_api.main:app --reload --port 8001
streamlit run app/dashboard.py
```

For local development, the dashboard's server-side API clients default to loopback ports 8000 and 8001. Keep the APIs running while using dashboard scenario prediction or chat.

## Health, troubleshooting, and rebuilds

`GET /health` reports process health and application version; it does not claim models are usable. API `GET /ready` keeps the existing `{"status":"ready"}`/sanitized `503` response shape and checks the runtime files used by the supported prediction, assistant, and dashboard views: both datasets, both models, model metadata, and classifier/regressor comparison files. It also deserializes both model artifacts and validates required metadata fields, feature contracts, and cross-model version consistency. `/model-info` distinguishes application and classifier artifact versions. These checks establish loadability and metadata consistency, not prediction quality or real-world accuracy. Streamlit's process health check uses its bundled `/_stcore/health` endpoint.

| Symptom | Checks and action |
| --- | --- |
| Docker daemon unavailable | Start Docker Desktop/Engine, then run `docker info` and `docker compose version`. The project cannot install the host Docker engine. |
| Port already occupied | Set the corresponding `*_HOST_PORT` in optional `.env` to an unused local port, then recreate with `docker compose up --build --force-recreate`. |
| Image build/dependency resolution failure | Inspect build output; run `docker compose build --no-cache`. The runtime dependency lock uses exact versions and hashes; update it from `pyproject.toml` using the command below only when dependencies intentionally change. |
| Initializer exits unsuccessfully | Run `docker compose logs runtime-init`. It shows which existing generation/preparation/training stage failed. Do not delete source files; if the volume is partial and disposable, reset with `docker compose down -v` and retry. |
| API stays unhealthy | Check `docker compose ps`, `docker compose logs main-api assistant-api runtime-init`, and the API `/ready` route. Verify initializer completed and volumes were not externally altered. |
| Dashboard cannot reach an API | Run the smoke command from the dashboard container. Ensure Compose URLs use `main-api` and `assistant-api`, not `localhost`; inspect dashboard and API logs. |
| App source changed but old data/model volume remains | Normal startup intentionally reuses existing resources and never retrains over them. Rebuild the image with `docker compose build --no-cache`; reset the generated volumes only when you explicitly want fresh synthetic resources. |

`requirements.lock` is the generated, runtime-only resolution of the dependencies in `pyproject.toml`; it targets Python 3.12 on Linux x86-64, includes exact transitive versions and hashes, and omits the development extra and the project itself. Docker installs that file with pip's `--require-hashes`; the image imports the source tree using `PYTHONPATH`.

Regenerate the runtime lock only as part of a deliberate dependency update (requires `uv`):

```powershell
uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes -o requirements.lock
```

Update the dependency ranges in `pyproject.toml` first, regenerate the lock, review all resolved versions and markers, then verify a clean container build, imports, tests, and smoke checks before accepting the change. Do not edit the generated lock by hand. The runtime lock is for Docker; local development installs the `dev` extra from the same project metadata and currently resolves for the active operating system.

Validate the resolved deployment and rebuild with:

```powershell
docker compose config
docker compose build --no-cache
docker compose up --build --wait --wait-timeout 900
docker compose exec -T dashboard python scripts/smoke_test.py
docker compose logs --no-color
docker compose down
```

The repository's unit suite remains independent of Docker:

```powershell
pytest -q
ruff check .
black --check .
```

## Validation status and limitations

The Compose implementation and smoke tooling are intended for local synthetic demonstration. During the 2026-10-17 V4.1.0 candidate verification, Docker CLI 29.8.2 and Compose 5.6.0 were installed, but `docker info` could not connect to `npipe:////./pipe/dockerDesktopLinuxEngine`. `docker compose config --quiet` passed independently; image build, service health, networking, persistence after restart, and live smoke tests remain unverified because they require a running engine. A previous successful run was reported in the project brief, but its logs are not in this checkout and were not independently reproduced. See the [V4.1.0 validation record](releases/v4.1.0.md). The setup does not add authentication, TLS termination, hosted deployment, automatic updates, durable audit logs, or external monitoring. It does not establish model accuracy on actual Tunja WiFi data.

## Version and release references

The remote tag `v4.1.0` points to `edc10f03b898217d7c0108115d0d111e53962d86`; it is an annotated Git tag, not a published GitHub Release. The V4.4.0 implementation candidate sets the Python package and Compose image to `0.4.4` and generates a diagnostics report during runtime initialization. Docker runtime and remote CI must be verified before stable publication. See [release history](release-history.md) and the [V4.4.0 validation record](releases/v4.4.0.md). No V4.4.0 tag or release is asserted here.

The V5.3.0 candidate uses package/image `0.5.3`; verification is pending and no tag or GitHub Release exists. V5.2.0 used `0.5.2`, V5.1.0 used `0.5.1`, and V5.0.0 introduced the `runtime-experiments` Compose volume. Runtime initialization creates the versioned SQLite experiment database without deleting existing records. The main API mounts it writable; assistant and dashboard containers mount it read-only. Both API and assistant resolve the same default database under the repository root, independent of process working directory; an explicit relative `EXPERIMENT_DB_PATH` is also resolved from the repository root. If model artifact fingerprints validate but a versioned diagnostics report is stale, initialization reevaluates the existing models and refreshes reports without retraining or replacing those models. `docker compose down` preserves experiments; `docker compose down -v` removes all Compose-managed data, including experiments. V5.2.0 runtime initialization, HTTP smoke, and persistence passed remote workflow [40039016215](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/actions/runs/40039016215); the local Docker engine was unavailable. See [experiment storage](experiments.md), the [V5.0.0 validation record](releases/v5.0.0.md), and the [V5.1.0 validation record](releases/v5.1.0.md).
