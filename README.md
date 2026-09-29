# WiFi Tunja Smart Predictor

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Read this in [Español](README.es.md)

An end-to-end portfolio application for classifying simulated WiFi demand as **LOW** or **HIGH**. It demonstrates a reproducible data pipeline, leakage-aware temporal evaluation, a reusable scikit-learn model pipeline, a FastAPI service, and an interactive Streamlit dashboard.

> **Synthetic data only.** Access points, coordinates, demand, weather, events, network metrics, and historical values are simulated. They do not represent actual public WiFi usage or municipal infrastructure in Tunja.

## 1. Project overview

The project explores how an hourly WiFi demand classification system can be structured as maintainable software. A user can generate data, validate and prepare it, train and evaluate candidate classifiers, then use the selected pipeline through an API or dashboard.

## 2. Key features

- Deterministic synthetic data generator and structured data validation.
- Historical and network snapshot features with explicit prediction-time semantics.
- Chronological train, validation, and test periods.
- Five scikit-learn baselines and a persisted preprocessing/model pipeline.
- FastAPI endpoints with typed request and response contracts.
- Streamlit pages for overview, demand exploration, geographic and network summaries, model evaluation, prediction, and project details.
- pytest, Ruff, Black, and GitHub Actions configuration.

## 3. Problem statement

The prototype classifies the expected demand level for a prediction hour. `demand_level` is the primary target (`LOW` or `HIGH`). It is a demonstration of engineering and evaluation practice, not a production forecast service.

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

Logistic Regression, Decision Tree, Random Forest, K-Nearest Neighbors, and Gaussian Naive Bayes are compared. The persisted model is selected using validation F1 for the `HIGH` class. Test metrics do not select the model. The comparison file is `reports/metrics/model_comparison.csv`.

## 10. Evaluation

Metrics include accuracy, precision, recall, F1 for `HIGH`, ROC-AUC where available, and a confusion matrix. **The evaluation is based on synthetic data.** Current generated values are stored in [selected model test metrics](reports/metrics/selected_model_test_metrics.json), with split details in [model metadata](models/model_metadata.json); regenerate these artifacts after retraining. They are not evidence of real-world accuracy.

## 11. Dashboard

Run `streamlit run app/dashboard.py`. Pages include Overview, Demand Explorer, Geographic Analysis, Network Analysis, Model Performance, Predict Demand, and About. Geographic points are simulated. Prediction output is a synthetic-data classification, not live telemetry or a causal explanation.

## 12. API

Run `uvicorn api.main:app --reload`, then visit [Swagger UI](http://127.0.0.1:8000/docs). The service provides `GET /health`, `GET /model-info`, and `POST /predict`. See [API documentation](docs/api.md) for the supported input contract.

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
pip install -r requirements-dev.txt
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
uvicorn api.main:app --reload
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
- There is no live data ingestion, municipal telemetry connection, deployment, or model monitoring.
- Predicted probabilities are classifier outputs and are not asserted to be calibrated operational confidence.
- `connections_next_hour` is a future-count target only; regression is a possible future extension and is not implemented here.

## 23. Future improvements

Potential next steps include authorized real datasets, regression for future connection counts, deployment, monitoring, persistent storage, ingestion, and model explanation tools. They are not part of this classification prototype.

## 24. Author

Created by [OAlexProgramerO](https://github.com/OAlexProgramerO). Repository: [wifi-tunja-smart-predictor](https://github.com/OAlexProgramerO/wifi-tunja-smart-predictor).
