# WiFi Tunja Smart Predictor — V6.4.0

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Read this in [Español](README.es.md)

**Release status:** V6.4.0 is published as annotated Git tag [`v6.4.0`](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/tree/v6.4.0) and [GitHub Release](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor/releases/tag/v6.4.0). Release commit: `8841996993aec18405f19e143796175cafa51540`; Python package and Docker image: `0.6.4`. See the [V6.4.0 validation record](docs/releases/v6.4.0.md), [model diagnostics guide](docs/model-diagnostics.md), and [release history](docs/release-history.md).

WiFi Tunja Smart Predictor is a synthetic-data demonstration of hourly demand classification and connection-count estimation, delivered through FastAPI, a Streamlit dashboard, and a deterministic assistant. V6.4.0 makes the existing evaluation summary clearer in Model Performance without changing the models or evaluation methodology.

V4.3.0 artifact fingerprint checks, V4.2.0 prediction traceability, and earlier validation improvements remain in place. See the [V4.4.0 diagnostics design](docs/model-diagnostics.md) for the report schema, metric semantics, time windows, and provenance checks.

The project uses synthetic data only. Its metrics do not establish predictive performance for real public WiFi demand or municipal infrastructure in Tunja. See the [methodology](docs/methodology.md), [limitations](docs/limitations.md), [API guide](docs/api.md), and [assistant guide](docs/assistant.md).

**Security boundary:** API protections are a practical baseline. Authentication, authorization, production rate limiting, WAF protection, and penetration-tested deployment are not provided.

> **Synthetic data only.** Access points, coordinates, demand, weather, events, network metrics, and historical values are simulated. They do not represent actual public WiFi usage or municipal infrastructure in Tunja.

## Docker quick start

Prerequisite: install and start Docker Desktop (or Docker Engine with Compose). No host Python, virtual environment, manual data generation, or model training is needed for the container demo.

```powershell
git clone https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor.git
cd wifi-tunja-smart-predictor
docker compose up --build --wait --wait-timeout 900
```

On its first run, the `runtime-init` service creates the synthetic dataset, trains both existing models, and evaluates them to create the diagnostics report in Compose-managed volumes. Wait for the APIs and dashboard to become healthy. Open the dashboard at <http://localhost:8501>, the main API docs at <http://localhost:8000/docs>, and assistant API docs at <http://localhost:8001/docs>. Inspect with `docker compose ps`, follow logs with `docker compose logs -f`, run the HTTP smoke checks from the dashboard container with `docker compose exec -T dashboard python scripts/smoke_test.py`, and stop with `docker compose down`. Data/model volumes persist after `down`; `docker compose down -v` removes only this Compose project's generated volumes.

The V5.3.0 verification record includes passing remote Docker build/startup, HTTP smoke, and persistence checks; this local machine did not have a running Docker engine. See the [V5.3.0 validation record](docs/releases/v5.3.0.md) for observed results and audit scope.

## 1. Project overview

The project explores how an hourly WiFi demand classification system can be structured as maintainable software. A user can generate data, validate and prepare it, train and evaluate candidate classifiers, then use the selected pipeline through an API or dashboard.

## 2. Key features

- Deterministic synthetic data generator and structured data validation.
- Historical and network snapshot features with explicit prediction-time semantics.
- Chronological train, validation, and test periods.
- Five scikit-learn baselines and a persisted preprocessing/model pipeline.
- FastAPI endpoints with typed request and response contracts.
- Baseline API protections: bounded request bodies, controlled CORS, security headers, sanitized validation errors, and constrained assistant context.
- A deterministic scenario builder, synthetic access-point resolver, and historical analog context; no live telemetry or paid map key is required.
- Separate classification and regression artifacts trained and evaluated on the same chronological split.
- A deterministic tool-grounded assistant with a separate API on port 8001; no LLM credential is required.
- JSON request lifecycle records and safe operation events correlate through `X-Request-ID` without logging chat messages or request bodies.
- English and Spanish greetings, assistant identity, and capability questions with case- and punctuation-insensitive matching.
- Context-aware natural demand questions in English and Spanish, including zone/time aliases and follow-up location reuse through the existing scenario service.
- Dashboard-aware assistant explanations for the dashboard sections, current scenario results, indicators, and model metrics in English and Spanish.
- Historical demand analysis in English and Spanish, including zone comparisons, HIGH-demand rates, peak hours/weekdays, time-of-day comparisons, and supported monthly change summaries.
- Ten dashboard sections: Overview, Live Scenario, Demand Explorer, Geographic Analysis, Network Analysis, Model Performance, Experiment History, AI Assistant, Advanced Prediction, and About.
- Reproducible capacity-planning experiments with SQLite persistence, deterministic budget allocation, paginated API history, and JSON/Markdown exports.
- Geospatial Intelligence filters and summarizes generated historical observations by inclusive date range, zone, hour, weekday, and demand class, with an offline synthetic access-point map.
- Scenario Workbench compares fixed model demand predictions against baseline and alternative capacity/budget assumptions, persists reproducible records, and exports scenario-aware reports.
- Scenario Workbench also offers V6.1 deterministic integer allocation optimization for total or explicitly weighted synthetic deficits; see the [optimization method and constraints](docs/optimization.md).
- V6.2.0 adds bounded one-factor synthetic demand sensitivity and budget sweeps to saved optimization runs; see the [sensitivity and robustness guide](docs/robustness.md). These selected tests do not estimate probabilities or real-world reliability.
- V6.3.0 adds a bounded temporal evaluation API and Model Performance workflow with versioned JSON/Markdown reporting; see the [guide](docs/temporal-evaluation.md). Results use synthetic data only.
- Experiment History dashboard page connected to saved scenario runs; simulated demand and capacity are connection counts, not throughput.
- pytest, Ruff, Black, and GitHub Actions configuration.

## Local development quick start

For local development, install with `python -m pip install -e ".[dev]"`, train both models using `python scripts/train_model.py`, and run evaluation with `python scripts/evaluate_model.py`. Start the dashboard with `streamlit run app/dashboard.py`, the main API with `uvicorn api.main:app --reload --port 8000`, and the assistant API with `uvicorn assistant_api.main:app --reload --port 8001`. `pyproject.toml` is the dependency source of truth; the two requirements files are compatibility wrappers. Local development supports Python 3.11+, while CI and the container use Python 3.12. Docker's hash-locked runtime dependencies target Linux x86-64; Compose selects `linux/amd64` so Docker Desktop on ARM hosts can use emulation.

The scenario endpoint is `POST /scenario/predict`; evaluation includes `POST /evaluation/temporal`; experiment routes include `POST /experiments/scenarios`, V6 `POST /experiments/what-if`, V6.1 `POST /experiments/optimize`, V6.2.0 `POST /experiments/robustness`, paginated `GET /experiments`, `GET /experiments/{id}`, and JSON/Markdown `GET /experiments/{id}/report`. These routes use port 8000. OpenAPI docs are available at each service's `/docs` path. See [temporal evaluation](docs/temporal-evaluation.md), [capacity planning](docs/capacity-planning.md), [experiments and reporting](docs/experiments.md), [the scenario workbench](docs/scenario-workbench.md), [optimization](docs/optimization.md), and [sensitivity and robustness](docs/robustness.md).

To create an experiment, open **Live Scenario**, choose a synthetic zone and time, run the prediction, set simulated capacity and an optional connection-count allocation budget, then choose **Run and save experiment**. **Experiment History** lists saved runs and downloads a versioned JSON report. For comparison, take two experiment IDs from `GET /experiments` and send them in `POST /experiments/compare`; comparisons require the same time, dataset fingerprint, and model artifact fingerprints. The API never labels these simulations as operational network recommendations.

See [Scenario Prediction](docs/scenario_prediction.md), [Assistant](docs/assistant.md), [Performance](docs/performance.md), [Geospatial Resolution](docs/geospatial.md), [Security Baseline](docs/security.md), and [Limitations, Privacy, and Security](docs/limitations.md).

## 3. Problem statement

The classifier predicts `demand_level` (`LOW` or `HIGH`) and the separate regressor estimates the existing `connections_next_hour` target. The scenario builder maps a synthetic location and time to historical context. This is a demonstration prototype, not a production or live forecast service.

## 4. Architecture

```text
Synthetic CSV → validation / deduplication → temporal split
                                      ↓
                         sklearn Pipeline (fit on train)
                           ↙                    ↘
                    FastAPI API          Streamlit dashboard
```

Reusable code lives in `src/wifi_tunja_smart_predictor/`; scripts orchestrate work, and the API and dashboard share the same model inference functions. See [architecture](docs/architecture.md).

## 5. Dataset

The generator writes `data/raw/wifi_tunja_public.csv` (about 60,000 rows, 55 source columns, 30 synthetic access points, and 12 synthetic zones, covering 2024–2025). It includes expected missing values and injected exact duplicates. Run the generator to recreate the file; preparation removes duplicates in the processed copy and preserves the raw CSV.

## 6. Synthetic data disclaimer

**The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.** Access points and coordinates are synthetic; demand observations, weather, events, network metrics, and historical values are simulated. Nothing in this repository should be interpreted as real Tunja WiFi usage.

## 7. Machine learning workflow

`prepare_data.py` validates the source, reports expected quality issues, removes exact duplicates, engineers documented ratios and cyclical features, and writes processed and sample datasets. Training uses a scikit-learn `Pipeline` containing a stateless feature transformer, a `ColumnTransformer`, and a classifier. Imputers, encoders, and scalers learn from the training period only.

## 8. Data leakage prevention

The row timestamp is the prediction time. Historical connection features refer to earlier hours; network snapshots summarize the preceding period. `demand_level` and `connections_next_hour` are never model inputs and are not used by inference preprocessing. Evaluation is chronological so later data cannot influence earlier training.

## 9. Models

Five classification candidates and two regression candidates are compared. The classifier is selected by validation F1 for the `HIGH` class; the regressor is selected by validation MAE. Both models use the same chronological split. The regression interval is calibrated from validation residuals with split conformal prediction. Test metrics do not select either model.

## 10. Evaluation

Metrics include accuracy, precision, recall, F1 for `HIGH`, ROC-AUC where available, and a confusion matrix. **The evaluation is based on synthetic data.** Current generated values are stored in [selected model test metrics](reports/metrics/selected_model_test_metrics.json), with split details in [model metadata](models/model_metadata.json); regenerate these artifacts after retraining. They are not evidence of real-world accuracy.

Model Performance presents the versioned [diagnostics report](docs/model-diagnostics.md), clearly surfacing the model version, evaluation date, headline classification and regression metrics, and synthetic-data disclaimer. Each chronological window fits on earlier synthetic rows only; none of these results validates real public WiFi demand.

## 11. Dashboard

Run `streamlit run app/dashboard.py`. The ten pages include location/time scenario prediction, experiment history and deterministic assistant chat. Geographic points are simulated. Predictions are estimates from synthetic data, not live telemetry; local sensitivity summaries are not causal explanations.

## 12. API

Run `uvicorn api.main:app --reload --port 8000`, then visit [Swagger UI](http://127.0.0.1:8000/docs). The main service preserves `GET /health`, `GET /model-info`, and `POST /predict`, and adds `GET /locations` and `POST /scenario/predict`. Start the deterministic assistant separately on port 8001. See [API documentation](docs/api.md).

## 13. Project structure

```text
api/                       FastAPI routes and schemas
app/                       Streamlit dashboard
data/{raw,processed,sample} generated datasets
docs/                      Architecture, methodology, API, and data docs
models/                    Persisted pipeline and metadata
notebooks/                 Exploration and analysis notebooks
reports/                   Generated metrics and figures
scripts/                   Generate, prepare, train, and evaluate
src/wifi_tunja_smart_predictor/ reusable package
tests/                     In-memory unit and API tests
```

## 14. Installation

Requires Python 3.11 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## 15. Dataset generation

```powershell
python scripts/generate_synthetic_dataset.py
```

This creates or replaces the generated raw CSV. Preparation writes separate files and does not alter the raw dataset.

## 16. Data preparation

```powershell
python scripts/prepare_data.py
```

## 17. Training

```powershell
python scripts/train_model.py
```

## 18. Evaluation

```powershell
python scripts/evaluate_model.py
```

## 19. API execution

```powershell
uvicorn api.main:app --reload --port 8000
```

Health and model metadata are available at `/health` and `/model-info`; interactive request documentation is at `/docs`.

## 20. Dashboard execution

```powershell
streamlit run app/dashboard.py
```

## 21. Testing and code quality

```powershell
pytest -q
ruff check .
black --check .
```

## 22. Project limitations

- All observations and evaluation results are synthetic.
- There is no live data ingestion, municipal telemetry connection, or model monitoring.
- Predicted probabilities are model scores, not calibrated operational confidence.
- Regression, capacity use, and intervals all describe synthetic targets; interval coverage may change under distribution shift.

## 23. Future improvements

Potential next steps include authorized real datasets, live providers, production deployment, monitoring, and persistent session storage.

## 24. Author

Created by [OAlexProgramerO](https://github.com/OAlexProgramerO). Repository: [wifi-tunja-smart-predictor](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor).
