# Reproducible deployment (V3.8)

This guide runs the existing main FastAPI API, deterministic assistant API, and Streamlit dashboard locally or with Docker Compose. Runtime data and trained model artifacts remain local inputs; the image contains application code and dependencies only. All observations and predictions remain synthetic and do not validate real-world Tunja WiFi demand.

## Prerequisites and runtime assets

- Local development: Python 3.11 or newer, as declared in `pyproject.toml`.
- Containers: Docker Engine with Docker Compose v2 (`docker compose`). The image uses the pinned `python:3.12.14-slim-bookworm` base.
- A generated raw dataset, prepared dataset, and both trained model artifacts. Starting from a fresh clone, create them using the existing workflow:

```powershell
python -m pip install -r requirements-dev.txt
python scripts/generate_synthetic_dataset.py
python scripts/prepare_data.py
python scripts/train_model.py
```

These commands use the established synthetic generator and training pipeline. They are setup steps, not part of container startup. Do not retrain or regenerate when you intend to retain existing artifacts.

## Local development

Existing commands remain supported:

```powershell
python -m pip install -r requirements-dev.txt
streamlit run app/dashboard.py
uvicorn api.main:app --reload --port 8000
uvicorn assistant_api.main:app --reload --port 8001
```

The dashboard invokes shared Python services directly and reads the same local dataset/model files; it does not make HTTP calls to either API. API endpoints remain available to other clients.

## Docker Compose

From the repository root, optionally copy `.env.example` to `.env` and edit safe local settings. Then:

```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f main-api assistant-api dashboard
docker compose down
```

`docker compose up --build` combines build and startup. Compose publishes only loopback host ports by default:

- Dashboard: <http://127.0.0.1:8501>
- Main API: <http://127.0.0.1:8000>; OpenAPI docs at `/docs`.
- Assistant API: <http://127.0.0.1:8001>; OpenAPI docs at `/docs`.

The three services share one image. They run as UID 10001, with a read-only root filesystem, dropped Linux capabilities, and `no-new-privileges`. The `data/` and `models/` directories are mounted read-only from the host. No data volume is created or modified by Compose. The dashboard uses in-process services and shared files, so no API URL settings or API startup dependency is required.

## Configuration

Compose reads an optional root `.env`; `.env` is ignored by Git and must not contain values placed in the image. Safe defaults are also provided in `compose.yaml`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `RANDOM_SEED` | `42` | Existing deterministic data/configuration seed. |
| `LOG_LEVEL` | `INFO` | Existing application logging setting. |
| `CORS_ALLOWED_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Explicit browser origins accepted by the APIs. Set to the exact dashboard origin if its host port or hostname is changed. |
| `MAIN_API_HOST_PORT` | `8000` | Loopback host port mapped to main API container port 8000. |
| `ASSISTANT_API_HOST_PORT` | `8001` | Loopback host port mapped to assistant container port 8001. |
| `DASHBOARD_HOST_PORT` | `8501` | Loopback host port mapped to dashboard container port 8501. |

The host port variables change only the published host side; the container ports are fixed. Compose environment values override defaults. Local development retains the current application defaults and does not require the host port variables.

## Health and troubleshooting

`GET /health` retains its existing liveness contract. The new `GET /ready` on each API returns `200` only when the raw dataset and both model files are present; otherwise it returns a sanitized `503`. It checks file presence only: it does not load artifacts, validate their contents, or prove a prediction can run. Streamlit's health check uses its built-in `/_stcore/health` endpoint. Compose does not wait for APIs before starting the dashboard because the dashboard does not depend on them over HTTP.

Inspect service status and logs with:

```powershell
docker compose ps
docker compose logs --tail=100 main-api
docker compose logs --tail=100 assistant-api
docker compose logs --tail=100 dashboard
```

For an unhealthy API, verify the required dataset and both model files exist on the host in their expected `data/` and `models/` locations, then inspect its logs. Files are ignored by Git and must be prepared locally. For a port binding error, set the relevant `*_HOST_PORT` to an unused port; if the browser-visible dashboard origin changes, update `CORS_ALLOWED_ORIGINS` to match. Validate resolved settings with `docker compose config`. API logs remain structured JSON on standard output and preserve V3.7 request IDs; request bodies and chat transcripts are not logged.

## Validation

Run the Python checks from the root:

```powershell
pytest -q
ruff check .
black --check .
```

When Docker is available, validate configuration/build/startup and health using the Compose commands above, then call `GET /ready`, a representative API prediction, and assistant `/chat`; confirm dashboard availability at its URL and stop cleanly with `docker compose down`. A successful readiness response alone verifies only required-file presence. Container startup or local synthetic predictions do not demonstrate forecast accuracy on real Tunja data.

## Limitations

This setup is for reproducible local development and demonstration. It does not add authentication, TLS termination, durable logs, external monitoring, distributed tracing, multi-host networking, automatic data/model provisioning, or a production deployment guarantee. Data is synthetic and is not verified operational telemetry from Tunja.
