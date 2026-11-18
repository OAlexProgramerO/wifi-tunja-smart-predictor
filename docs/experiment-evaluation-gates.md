# Automated ML experiment evaluation gates

V8.3 evaluation gates apply configurable metric thresholds to saved, completed synthetic ML experiments. Gates are read-only: they do not fit models, alter predictions, or promote candidates.

## API

`POST /ml-experiments/gates` accepts 2–10 experiment IDs and criteria for one task:

```json
{
  "experiment_ids": ["saved-run-a", "saved-run-b"],
  "task": "classification",
  "criteria": [{"metric": "f1", "operator": "gte", "threshold": 0.7}]
}
```

Allowed classification metrics are accuracy, precision, recall, F1, and ROC-AUC. Allowed regression metrics are MAE, RMSE, and R². `gte` means a recorded value must be at least the threshold; `lte` means it must be at most the threshold. The operator must match the metric's usual direction (higher for classification metrics and R²; lower for MAE/RMSE). Boundary equality passes.

## Results

Each experiment gets `PASSED` when every criterion has valid evidence and meets its threshold, `FAILED` when evidence is valid and at least one criterion does not meet its threshold, or `INCONCLUSIVE` when required metrics/provenance are missing or invalid. The report includes criterion-level values, thresholds, explanations, dataset fingerprint, split/window provenance, experiment timestamps, and schema version.

Classification and regression cannot share one gate request. A report also requires the selected runs to be compatible under the V8.2 comparison contract: same synthetic dataset fingerprint, task, split, runtime, feature/target configuration, and chronological windows. Incompatible groups are `INCONCLUSIVE`, and no thresholds are judged. Do not interpret a passed gate as a universal model ranking or evidence of real-world performance.

Only synthetic ML experiment records are accepted. Official MinTIC data is not used. These gates do not test statistical significance or real-world forecast validity and never modify production artifacts or automatically promote a model.
