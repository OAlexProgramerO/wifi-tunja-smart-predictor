# Architecture

VERSION **0.3.6** of WiFi Tunja Smart Predictor is an installable package under `src/`, command-line training/evaluation scripts, a backward-compatible main FastAPI service, an independent deterministic assistant API, and a nine-section Streamlit dashboard.

**The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.** Coordinates and access-point identifiers do not correspond to real public WiFi infrastructure in Tunja.

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
