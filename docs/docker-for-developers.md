# Docker guide for developers

This guide uses the services and volumes in this repository. The demo uses synthetic data only.

## The main Docker pieces

- The `Dockerfile` describes how to assemble the application image: the Python 3.12 base, locked dependencies, project source, and non-root runtime user.
- An **image** is the built template. Compose labels the local image `wifi-tunja-smart-predictor:0.3.8`.
- A **container** is a running instance of that image. This project runs `runtime-init`, `main-api`, `assistant-api`, and `dashboard` containers.
- `compose.yaml` configures those containers, their startup order, health checks, network, host ports, environment, and persistent named volumes.

## Start, inspect, stop, and restart

Install Docker Desktop on Windows and wait for its Linux engine to report ready. Then run from the repository root:

```powershell
docker compose config --quiet
docker compose up -d --build --wait --wait-timeout 900
docker compose ps
docker compose exec -T dashboard python scripts/smoke_test.py
```

Open the dashboard at `http://localhost:8501`; the main API is `http://localhost:8000` and its interactive docs are `/docs`; the assistant API is `http://localhost:8001` with `/docs`. Inside the Compose network, the dashboard calls `main-api:8000` and `assistant-api:8001`, not `localhost`.

```powershell
docker compose logs -f
docker compose stop
docker compose start
docker compose down
docker compose up -d --build --wait --wait-timeout 900
```

`stop` halts containers without removing them. `start` resumes those containers. `down` removes project containers and network but keeps named volumes. `up --build` rebuilds the image if needed and recreates containers; the initializer reuses complete data/model/report volumes. It does not mean retraining on every restart.

## Health checks and readiness

Compose checks the main API at `/ready`, the assistant at `/ready`, and Streamlit at `/_stcore/health`. A health check reports whether a process/resource check responds; it is different from `GET /health`, which reports process identity/version. API readiness checks required runtime files, deserializes both models, and validates model metadata consistency. The smoke command also exercises model-info, prediction, scenario prediction, assistant chat, and dashboard HTTP. Passing health alone does not prove the model is scientifically valid or accurate on real WiFi data.

For a failing container, inspect:

```powershell
docker compose ps --all
docker compose logs --tail 200 runtime-init main-api assistant-api dashboard
docker inspect --format '{{json .State.Health}}' wifi-tunja-smart-predictor-main-api-1
```

Container names may include a project prefix or index; use the names printed by `docker compose ps` if they differ. `runtime-init` logs identify which generation, preparation, or training step failed. The initializer refuses to overwrite partial volumes and validates complete model bundles before reuse; invalid artifacts cause a safe startup failure without replacing persisted resources.

## Volumes and persistent resources

`runtime-data`, `runtime-models`, and `runtime-reports` are named Compose volumes. They hold the synthetic CSVs, classifier/regressor artifacts and metadata, and metric comparison files. Removing containers with `docker compose down` does not remove them. Inspect them with:

```powershell
docker volume ls
docker volume inspect wifi-tunja-smart-predictor_runtime-data
```

Do not run `docker compose down -v` as routine cleanup: `-v` removes this Compose project's named volumes, including persisted datasets and trained models. Rebuild an image or recreate a container without `-v` when checking source changes. A fresh initialization test should use a separate Compose project name and its own clearly identified volumes; it must not reset this project's existing volumes.

## Docker Desktop, WSL2, and resource use

Docker Desktop provides the Windows UI/engine integration. On the common WSL2 backend, Linux containers run inside a managed lightweight Linux VM; WSL2 supplies that virtualization layer. `VmmemWSL` is a Windows process representing resources used by that VM, not an application container. Its memory/CPU use can include the Docker engine and all running Linux containers.

Use Docker Desktop's **Containers** view or `docker stats` to inspect container-level CPU and memory:

```powershell
docker stats
docker compose top
```

Use Task Manager or Resource Monitor to inspect host processes such as `VmmemWSL`. Stop unneeded containers with `docker compose stop` or `docker compose down`; neither command removes the project's persistent data volumes. Docker Desktop settings let you adjust CPU/memory limits where the selected backend supports them. This project trains synthetic models during the first initialization, which can temporarily use more CPU and memory than later restarts.

## Safe troubleshooting checklist

1. Check `docker info` and `docker compose version`; start Docker Desktop if the Linux engine is unavailable.
2. Check `docker compose config --quiet` and `docker compose ps --all`.
3. Read `docker compose logs runtime-init` before attempting recovery.
4. Check whether host ports 8000, 8001, or 8501 are already in use; change the corresponding `*_HOST_PORT` value in `.env` if necessary.
5. Keep persisted volumes intact unless you deliberately want to discard generated demo data. `docker compose down -v` is destructive to those generated resources.

For environment variables, full service details, and API endpoints, see [deployment](deployment.md). For the V3.8/V3.9/V4.0 version map, see [release history](release-history.md).
