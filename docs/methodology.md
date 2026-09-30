# Methodology

**The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.** It does not represent real municipal measurements or real public WiFi infrastructure in Tunja.

## Dataset generation

`scripts/generate_synthetic_dataset.py` simulates 30 access points in 12 named synthetic zones over 2024-01-01 through 2025-12-31, then samples approximately 60,000 prediction-time rows (seed **42**). A warm-up window is simulated so lag features are defined on the first exported day. Exact generation details live in that script; this document describes how VERSION 0.3 uses the file.

## Target definition

- **Primary classification target:** `demand_level` ∈ {`LOW`, `HIGH`}.
- The generator labels HIGH when simulated `connections_next_hour / access_point_capacity` exceeds a calibrated load threshold.
- **Secondary series:** `connections_next_hour` (count in the hour starting at `timestamp`).
- Neither target is a model input. No derived feature is built from them.

## Timestamp contract

`timestamp` is the **prediction time** (start of the hour being classified). Historical connection features use hours **strictly before** `timestamp`. Network snapshot metrics (devices, latency, utilisation, …) describe the hour that **just ended**, never the hour being predicted.

## Feature groups

| Group | Role in VERSION 0.3 |
| --- | --- |
| Identifiers (`wifi_id`, `zone_id`, `zone_name`, lat/lon/alt) | Not used as model inputs (location identity / dummy risk). |
| `zone_type` | Categorical feature (land-use generalisation). |
| Temporal | Cyclical hour / weekday / month; calendar flags; `time_period`. |
| Weather, context, events | Prediction-time covariates (with generator missingness). |
| Network | Last completed hour snapshot. |
| Historical | Lagged connection summaries available before `timestamp`. |
| Targets | Excluded from X. |

## Missing values and duplicates

The generator injects roughly 1–3% missingness in selected non-target columns and a small number of exact duplicate rows. The validator treats those as **expected quality issues**. The training pipeline imputes missing values **after** the split, using training-set statistics only. `scripts/prepare_data.py` drops exact duplicates and writes `data/processed/` without modifying `data/raw/`.

## Temporal validation

Random row-wise `train_test_split` would mix future hours into training. VERSION 0.3 uses a calendar holdout:

| Split | Timestamp rule |
| --- | --- |
| Train | `timestamp` ≤ 2025-06-30 23:59:59 |
| Validation | 2025-07-01 through 2025-09-30 23:59:59 |
| Test | `timestamp` > 2025-09-30 23:59:59 |

Imputers, scalers, and encoders are fitted on **train** only. The persisted model is the pipeline fitted on train. Model **selection** uses validation F1 for `HIGH`. Reported comparison metrics are computed on **test**. This reflects deployment chronology: future observations must not contribute to model fitting or preprocessing of earlier predictions.

## Leakage prevention

- Forbidden inputs: `demand_level`, `connections_next_hour`, and any feature derived from them.
- No preprocessing `fit` on train+val+test combined.
- Temporal split prevents future rows from influencing learned parameters.
- Feature engineering is stateless (no global dataset statistics).

## Preprocessing and models

Sklearn `Pipeline`: `FeatureEngineer` → `ColumnTransformer` → classifier.

Classification candidates: Logistic Regression, Decision Tree, Random Forest, KNN, Gaussian Naive Bayes. Regression candidates: Random Forest Regressor and HistGradientBoostingRegressor. Classifier selection uses validation F1 for HIGH; regressor selection uses validation MAE. Test metrics are reserved for final reporting.

The regression target is the existing `connections_next_hour`; it is excluded from all feature inputs. The nominal 90% split-conformal interval uses the finite-sample quantile of absolute validation residuals around the selected regressor. Test coverage and mean interval width are reported. This procedure does not guarantee coverage after distribution shift, and all results remain specific to synthetic data.

Class imbalance is reported from the generated synthetic labels (generator target: roughly 55–70% LOW). VERSION 0.3 does **not** apply oversampling; precision/recall/F1 are reported for `HIGH`.

## Evaluation

Accuracy, precision, recall, F1 (`HIGH` as positive class), ROC-AUC when `predict_proba` exists, and a confusion matrix. All numbers must come from execution. They describe **the synthetic evaluation dataset only**.

