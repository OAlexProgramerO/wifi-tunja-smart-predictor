# V6.3.0 — Temporal evaluation and synthetic drift diagnostics

`POST /evaluation/temporal` runs an additional, bounded evaluation on the project's synthetic dataset. It clones the persisted classifier and regressor pipelines, fits each clone on timestamps strictly earlier than its test block, and reports both model metrics and selected feature-distribution comparisons. This is separate from the existing chronological holdout and saved expanding-window diagnostics. It does not change model selection or saved model artifacts.

## Windows and metrics

The service sorts unique timestamps in stable order, reserves the first half as initial training history, then partitions the remaining period into 2–5 non-overlapping test blocks. Training expands to include prior rows only. The request can set minimum training and test row counts (defaults 100 and 20); too-small windows and training windows with one target class are reported as skipped. At least one usable timestamp per requested window is required. Missing or invalid timestamps and unsupported features return a validation error.

Classification includes accuracy, positive-class precision/recall/F1, ROC-AUC when both target classes and probabilities are available, and class support/confusion counts. Regression includes MAE, RMSE, R² when defined, bias, and absolute-error quantiles. A one-class test block remains evaluable; ROC-AUC is then undefined. Undefined/non-finite values serialize as JSON `null`.

For numerical features, the diagnostic is the absolute mean difference divided by that difference plus the pooled standard deviation, bounded from 0 to 1. For categorical features, it is total variation distance between category proportions, also bounded from 0 to 1. Both report missing proportions and sample counts; fewer than two observed values on either side yields `insufficient_samples`. These are descriptive comparisons, not alert thresholds or tests of causality.

## API

Example request:

```http
POST /evaluation/temporal
Content-Type: application/json
```

```json
{"window_count": 3, "min_train_rows": 100, "min_test_rows": 20, "features": ["hour", "zone_type", "temperature_c"]}
```

The versioned JSON response contains project and preprocessing feature versions, settings, ordered train/test boundaries and counts, available classification/regression metrics, feature diagnostics, dataset SHA-256, model artifact SHA-256 fingerprints, a generation timestamp, and limitations. Use `POST /evaluation/temporal?report_format=markdown` for a concise Markdown summary. Settings are bounded to 2–5 windows, at most 50,000 minimum training rows, 10,000 minimum test rows, and 12 unique model input features. Larger datasets or model fits can take longer; no background job or persistent report store is created.

## Dashboard and interpretation

In **Model Performance**, choose 2–5 windows and up to 12 features, then run the evaluation. The dashboard shows window metrics and feature comparisons and provides JSON and Markdown downloads. Results are recomputed from the current synthetic dataset and loaded model artifacts; the JSON includes fingerprints for reproducibility. Timestamp and dataset changes therefore change the report identity.

All rows, targets, locations, and measurements in this project are synthetic. A measured feature-distribution change does not prove model degradation, causality, live drift, reduced service quality, or valid forecasts for real public WiFi demand in Tunja. The diagnostics do not validate municipal infrastructure or production monitoring.
