# Architecture

The V6.0.0 candidate extends the installable package, main FastAPI service, deterministic assistant API, and existing Streamlit dashboard. Classifier/regressor, diagnostics and prediction boundaries remain unchanged. The geography analytics service produces filtered historical synthetic observation summaries; scenario simulation calls the existing prediction service once per common zone/time and evaluates alternatives through the existing deterministic capacity planner. The existing experiment store persists the comparison additively under the established record schema, and the shared report serializers expose JSON/Markdown schema 1.2. The main API has write access to a dedicated Compose experiment volume, while assistant and dashboard mounts are read-only. API and assistant share a repository-root-relative default database location; SQLite connections close after every operation, and malformed saved comparisons fail safely. On an application-version upgrade, initialization can refresh diagnostics from validated existing model artifacts without retraining. Assistant answers remain deterministic and do not use an LLM. V4.x model metadata continues recording artifact SHA-256 and byte length and verifying those fingerprints against deserialized bytes; legacy metadata compatibility remains supported. The V6 product version `6.0.0` maps to package and image version `0.6.0`; see the [scenario workbench guide](scenario-workbench.md) and [validation record](releases/v6.0.0.md).

Scenario experiment flow: dashboard/API request → existing `ScenarioPredictionService` for each synthetic zone → deterministic capacity allocation and aggregate metrics → SQLite transaction → compatible comparison and report serialization. The experiment database is initialized by `runtime-init`; no model training or model artifact mutation occurs inside scenario requests.

**The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.** Coordinates and access-point identifiers do not correspond to real public WiFi infrastructure in Tunja.

Runtime and development dependency declarations live in `pyproject.toml`. `requirements.lock` is generated from the runtime dependencies for Python 3.12/Linux x86-64 and installed with hash verification in the image. Compose pins `linux/amd64`; local development supports Python 3.11+ and installs the `dev` extra without using the container-only lock.

## ASCII diagram

```
                    +------------------------------+
                    | scripts/generate_synthetic_  |
                    | dataset.py  (seed=42)        |
                    +--------------+---------------+
                                   | CSV
                                   v
                    data/raw/wifi_tunja_public.csv
                                   |
          +------------------------+------------------------+
          |                        |                        |
          v                        v                        v
   scripts/prepare_data.py   notebooks (EDA)         tests (fixtures)
          | load / validate / dedupe / engineer
          v
   data/processed/*.csv
          |
          v
   scripts/train_model.py
          | temporal split -> sklearn Pipeline fit on TRAIN only
          v
   models/wifi_demand_classifier.joblib + wifi_demand_regressor.joblib
          |                          \
          v                           v
   scripts/evaluate_model.py    src/.../models/predict.py
          | reports/                   |
          v                    +-------+--------+
   reports/metrics, figures    |                |
                               v                v
                         api/ (FastAPI)   app/dashboard.py
                               |                |
                               + assistant_api/ + tests/
```

## Layers

### Data layer (`src/wifi_tunja_smart_predictor/data/`)

- `loader.py` locates the CSV from the repository root and parses timestamps.
- `validator.py` separates **expected** quality issues (generator-injected missingness and a few duplicates) from **critical** contract violations (missing schema, invalid targets, unparsable timestamps).

### Feature engineering layer (`features/engineering.py`)

- Cyclical calendar encodings and prediction-time ratios.
- Implemented as a stateless sklearn transformer so the same code is persisted with the model.
- Never reads `demand_level` or `connections_next_hour`.

### Preprocessing layer (`preprocessing/pipeline.py`)

- `ColumnTransformer`: median imputation + optional scaling for numerics; most-frequent imputation + `OneHotEncoder(handle_unknown="ignore")` for categoricals.
- Fitted only after the temporal split, on training rows.

### Model layer (`models/train.py`, `models/regression.py`, `models/evaluate.py`)

- Five baseline classifiers with bounded hyperparameters.
- Comparison table written to `reports/metrics/model_comparison.csv`.
- Selection uses **validation F1** for class `HIGH`. Test metrics are reported for all models and are not used for selection.
- Separate next-hour connection regressors are selected by validation MAE and persisted with independent metadata.
- A split-conformal interval uses validation absolute residuals; test coverage is reported separately.

### Inference layer (`models/predict.py`)

- `load_model`, `predict_demand`, `predict_probability`.
- Shared by FastAPI and Streamlit.

### API layer (`api/`)

- The original `/health`, `/model-info`, and `/predict` contracts remain available; V3 adds `/locations` and `/scenario/predict`.
- OpenAPI at `/docs`.

### Dashboard layer (`app/dashboard.py`)

- Overview, scenario prediction, demand explorer, geographic analysis, network analysis, model performance, assistant, advanced prediction, and about.
- Location/time scenario construction runs through shared package services. Frames and model resources are cached.

### Assistant layer (`assistant/`, `assistant_api/`)

- Deterministic intent routing invokes fixed prediction, location, model metadata, and aggregate-query tools.
- Dataset queries use strict allowlists; assistant input is never executed as code or SQL.
- Session context is bounded, process-local, and non-durable.

### Testing layer (`tests/`)

- Fast unit tests with in-memory frames. CI does not train on 60,000 rows.

## End-to-end flow

1. Generate the immutable source CSV with `scripts/generate_synthetic_dataset.py`.
2. `scripts/prepare_data.py` validates it, removes exact duplicates in a derived copy, and writes processed data.
3. `scripts/train_model.py` splits chronologically, fits candidate pipelines on train, chooses by validation F1, and writes the selected artifact and metadata.
4. `scripts/evaluate_model.py` evaluates the saved artifact on the held-out test period and writes metrics and figures.
5. Scenario APIs, dashboard, and assistant use the same scenario builder and model service. Tests use small in-memory data and injected models.

## Modeling location of identifiers

`wifi_id`, `zone_id`, and `zone_name` are **not** model inputs. `zone_name` is a one-to-one proxy for `zone_id` and would act as a location dummy. **`zone_type`** is used instead: it describes synthetic land use and can generalise across access points.

## Runtime resource reuse (V3.6)

The API and dashboard retain their existing bounded service/model/data caches. The assistant owns one normalized `DatasetQueryEngine` and the dataset query route reuses it; it does not rebuild a full frame for each request. Scenario construction computes deterministic analog feature baselines alongside the scenario features, and the explanation step consumes those values rather than reselecting analog rows from a copied historical frame. No prediction model, features, endpoint schema, or dashboard behavior changed. See [performance notes](performance.md).

## Local container topology and runtime resources (V3.9 implementation)

```text
docker compose up --build
          |
          v
 runtime-init (only for empty named volumes)
   generate → prepare → train existing pipeline
          |
     successful exit
       /       \
      v         v
 main-api    assistant-api
   |  \          |
   |   \         | /chat
   |    v        v
   +-- dashboard (server-side Compose DNS clients)

runtime-data, runtime-models, runtime-reports
       initialized once, mounted read-only by app services
```

The initializer refuses to overwrite a partially initialized volume and validates complete existing model bundles before reuse; invalid artifacts fail initialization without being replaced. A failed initialization blocks API and dashboard startup; its stage output is available with `docker compose logs runtime-init`. Normal `docker compose down` preserves generated volumes and avoids retraining on the next start. `docker compose down -v` removes the Compose project's generated volumes for an explicit synthetic-data reset. See [deployment](deployment.md) for commands and failure recovery.

Both APIs retain `/health` as a process liveness check. `/ready` checks required datasets, model and metadata files, comparison reports, deserialization of both models, and consistency of model versions and feature metadata. `/model-info` separates the application package version from the classifier artifact version. Compose starts APIs after initialization succeeds and the dashboard only after both APIs pass readiness. A separate standard-library smoke script exercises the real HTTP APIs from the dashboard container's network context. This setup remains a local synthetic demonstration and does not imply real-world forecast validity.

## Observability (V3.7)

Both FastAPI services install the shared request observability middleware outside the existing body-limit, security-header, and CORS middleware. It validates or generates a request ID, places it in a request-scoped context, returns it in `X-Request-ID`, and emits one monotonic-timed lifecycle event. API routes and the assistant service emit safe business-operation events using that context. The shared formatter writes JSON records; no request body or conversation is persisted. See [observability](observability.md).
