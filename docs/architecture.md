# Architecture

VERSION **0.2.0** of WiFi Tunja Smart Predictor is a single-repository prototype: an installable package under `src/`, command-line scripts, a FastAPI service, and a navigable Streamlit dashboard.

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
   models/wifi_demand_classifier.joblib
          |                          \
          v                           v
   scripts/evaluate_model.py    src/.../models/predict.py
          | reports/                   |
          v                    +-------+--------+
   reports/metrics, figures    |                |
                               v                v
                         api/ (FastAPI)   app/dashboard.py
                               |                |
                               +---- tests/ ----+
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

### Model layer (`models/train.py`, `models/evaluate.py`)

- Five baseline classifiers with bounded hyperparameters.
- Comparison table written to `reports/metrics/model_comparison.csv`.
- Selection uses **validation F1** for class `HIGH`. Test metrics are reported for all models and are not used for selection.

### Inference layer (`models/predict.py`)

- `load_model`, `predict_demand`, `predict_probability`.
- Shared by FastAPI and Streamlit.

### API layer (`api/`)

- HTTP adapters only. `/health`, `/model-info`, `/predict`.
- OpenAPI at `/docs`.

### Dashboard layer (`app/dashboard.py`)

- Overview, demand explorer, geographic analysis, network analysis, model performance, prediction, and about pages.
- Dataset loading is cached by Streamlit; model loading is cached by the shared inference layer.
- Calls the inference layer; does not reimplement preprocessing.

### Testing layer (`tests/`)

- Fast unit tests with in-memory frames. CI does not train on 60,000 rows.

## End-to-end flow

1. Generate the immutable source CSV with `scripts/generate_synthetic_dataset.py`.
2. `scripts/prepare_data.py` validates it, removes exact duplicates in a derived copy, and writes processed data.
3. `scripts/train_model.py` splits chronologically, fits candidate pipelines on train, chooses by validation F1, and writes the selected artifact and metadata.
4. `scripts/evaluate_model.py` evaluates the saved artifact on the held-out test period and writes metrics and figures.
5. The API and dashboard call the same package inference functions. Tests use small in-memory data and an isolated artifact.

## Modeling location of identifiers

`wifi_id`, `zone_id`, and `zone_name` are **not** model inputs. `zone_name` is a one-to-one proxy for `zone_id` and would act as a location dummy. **`zone_type`** is used instead: it describes synthetic land use and can generalise across access points.
