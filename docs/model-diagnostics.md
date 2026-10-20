# Model diagnostics

V4.4 adds `reports/metrics/model_diagnostics.json`, a generated report shared by the API, the Model Performance dashboard section, and deterministic assistant answers. The report is runtime data and is never committed as a model-quality claim.

## Generation and provenance

Run `python scripts/evaluate_model.py` after training. Compose runtime initialization runs the same evaluation stage after generating data and training. The writer serializes JSON to a temporary file in the destination directory, flushes it, then atomically replaces the prior report.

The report includes a schema version, UTC generation time, project/model versions, dataset row count and SHA-256, duplicate-row and missing-value summaries, classifier and regressor SHA-256 values and selected names, temporal holdout boundaries, metrics, chronological windows, and synthetic-data limitations. Consumers reject a report whose schema, package version, dataset fingerprint, or either model fingerprint no longer matches current resources. The HTTP endpoint responds with `503` for missing, malformed, or stale reports; the dashboard and assistant state that the report is unavailable.

## Metrics and time windows

The final test section evaluates the already selected classifier and regressor on the existing chronological held-out test period. Classification includes fixed `LOW`/`HIGH` confusion labels, support and per-class precision/recall/F1, accuracy, and ROC-AUC only when both classes and probabilities are available. Undefined ROC-AUC is `null` with a reason; absent class denominators are `null`, not fabricated zeros. Regression includes MAE, RMSE, R² when at least two rows exist, mean prediction bias, and absolute-error quantiles at 50%, 90%, and 95%.

Three expanding chronological windows use row-ordered cut points at 50%, 60%, and 70% for training, followed by a 10% evaluation block. Every fold clones and fits both complete estimator pipelines using only the earlier training rows; preprocessing is therefore fitted within each fold. Start/end times and row counts are recorded. A fold that cannot be evaluated is marked `skipped` with a reason. These fold windows are supplementary to, and do not replace, the final test period.

The report does not claim probability calibration or real-world forecast validity. Dataset values, coordinates, network measurements, and labels are synthetic; no result here validates public WiFi demand in Tunja.

## Consumers

- `GET /model-diagnostics` serves a typed response only after current-provenance checks.
- Model Performance shows report provenance, class metrics/confusion matrix, rolling windows, regression errors, and warnings.
- The deterministic assistant answers current metric-value questions from the validated report in English or Spanish; it does not embed metric values in templates.

See [API contracts](api.md), [assistant behavior](assistant.md), and [limitations](limitations.md).
