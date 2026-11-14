# Model run comparison

## Scope and source records

`GET /model-runs/comparison` reads the persisted classifier and regressor candidate evaluation tables written by the existing training workflow, plus their selected-model metadata. It compares candidates only within the same task (classification with classification, regression with regression) because their metrics and targets differ. It does not retrain models or write to model/report files.

The repository retains the latest candidate tables rather than an append-only history of training runs. Each row is therefore a candidate evaluation from that saved training cycle, not a sequence of independent deployments. The response reports the model/package version, training timestamp, dataset version and fingerprint when available, current selected artifact fingerprint, and the shared train/validation/test windows. Unselected candidates have no persisted artifact fingerprint. Missing lineage evidence reduces comparison status to `PARTIAL`; mismatched dataset fingerprints or evaluation windows are `INCOMPATIBLE` and receive no metric deltas.

## Status meanings

- **VALID** — records are from the same task and have matching dataset fingerprints and evaluation windows; available metric deltas are safe to interpret within this synthetic comparison.
- **PARTIAL** — some records or lineage fields are missing, malformed, or lack artifact fingerprints, so only the available details are shown.
- **INCOMPATIBLE** — dataset identity or evaluation windows differ; metric differences are omitted.
- **UNAVAILABLE** — the saved tables or metadata cannot be read.

Metric deltas are candidate minus the selected candidate. Higher is generally favorable for accuracy, precision, recall, F1, ROC-AUC, and R²; lower is generally favorable for MAE, RMSE, and training time. These are descriptive synthetic evaluation results, not evidence of real-world WiFi demand performance. Official MinTIC records are not read by this feature.

The files contain only the most recent candidate comparison; historical run comparisons require a separately retained run archive, which this release does not introduce.
