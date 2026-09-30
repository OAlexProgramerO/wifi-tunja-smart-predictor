# Model evaluation

**All metrics below describe only a synthetic temporal holdout. They do not estimate real-world public WiFi performance.** Re-run `python scripts/train_model.py` and `python scripts/evaluate_model.py` to regenerate reports.

## Temporal split

| Split | Rows | Start | End |
| --- | ---: | --- | --- |
| Train | 44,819 | 2024-01-01 00:00 | 2025-06-30 23:00 |
| Validation | 7,547 | 2025-07-01 00:00 | 2025-09-30 23:00 |
| Test | 7,574 | 2025-10-01 00:00 | 2025-12-31 23:00 |

Both models use the same ordered split. Classifier selection uses validation F1 for `HIGH`; regressor selection uses validation MAE. Test rows are reserved for final reporting.

## Classification

Five bounded scikit-learn candidates are compared. The selected random forest test metrics from the current generated run are:

| Accuracy | Precision HIGH | Recall HIGH | F1 HIGH | ROC-AUC |
| ---: | ---: | ---: | ---: | ---: |
| 0.9246 | 0.9186 | 0.8917 | 0.9049 | 0.9804 |

Metrics are in `reports/metrics/selected_model_test_metrics.json`; the full candidate table is `reports/metrics/model_comparison.csv`.

## Regression and intervals

The target is the existing `connections_next_hour` column. The selected random forest regressor's current test metrics are MAE **27.91**, RMSE **52.86**, and R² **0.836** connections. Its nominal 90% split-conformal interval uses the finite-sample absolute-residual quantile from validation predictions, radius **72.04** connections. Test coverage was **89.6%**, with mean width **116.31** connections.

Regression metadata is stored in `models/regression_metadata.json`; evaluation summaries are in `reports/metrics/selected_regression_test_metrics.json`. Coverage is empirical on this synthetic holdout and is not guaranteed under distribution shift.

## Interpretation limits

- All outcomes, features, APs, and coordinates are simulated.
- Class probabilities are model scores, not operationally calibrated confidence.
- Connection counts serve only as a capacity-use proxy; detailed RF airtime and throughput are not modeled.
- Local one-feature replacement measures sensitivity, not causation.
