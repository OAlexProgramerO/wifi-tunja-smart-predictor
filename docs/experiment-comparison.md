# Advanced ML experiment comparison

## API

Send two to ten unique saved experiment IDs to `POST /ml-experiments/compare`:

```json
{
  "experiment_ids": ["saved-run-id-a", "saved-run-id-b"]
}
```

The endpoint reads the existing `ml_experiments` records and returns a versioned JSON report. It does not modify experiment history or model artifacts. The dashboard's Experiment Orchestration page exposes the same comparison and a JSON download.

## Compatibility contract

Comparisons are strictly pairwise. A pair is comparable only when both records are completed and contain consistent provenance for synthetic data; task, source dataset fingerprint and algorithm, project/runtime versions, random seed, target and ordered model features, and the chronological training/evaluation windows must match. Both records must use the same `validation` or `test` partition. Missing or inconsistent fields make the pair `INCOMPATIBLE`; reasons are listed and metric deltas are omitted.

Candidate model names and hyperparameters are intentionally allowed to differ: those are the settings being examined. Candidate artifact fingerprints are reported as provenance but are not required to match because distinct fitted candidates naturally have distinct artifacts. In V8.1 candidate artifacts were not persisted, so fingerprints are unavailable.

## Metric interpretation

Classification and regression are never compared to each other. For classification, higher accuracy, precision, recall, F1, and ROC-AUC are treated as improvements. For regression, lower MAE/RMSE and higher R² are treated as improvements. A difference with absolute magnitude at most `1e-9` is a tie. Only shared finite metrics are compared. Missing or undefined metrics are listed in `unavailable_metrics`; a pair with some missing metrics is `PARTIAL` while valid shared metric comparisons remain visible.

For 3–10 records the report includes every unordered pair. Overall status is `COMPARABLE` when every pair is fully comparable, `PARTIAL` when at least one pair has useful or partial comparisons but not all pairs are fully comparable, and `INCOMPATIBLE` when every pair fails the compatibility checks. The report does not aggregate unlike metrics or select a universal best model. Review each pair and metric in context.

## Data and limitations

The engine defensively checks that target/future fields are absent from the recorded feature list and that the evaluation partition and row count agree with the saved chronological split. It uses only saved experiment metadata and metrics; it never loads MinTIC records, retrains models, recalculates scores, or changes predictions. Comparisons describe synthetic evaluation runs only and do not demonstrate real-world performance in Tunja. Exact run replay remains subject to the recorded `TRACEABLE_NOT_SELF_CONTAINED` limitations in the experiment report.
