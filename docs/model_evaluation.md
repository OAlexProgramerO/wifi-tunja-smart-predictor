# Model evaluation

**Performance measured on the synthetic evaluation dataset.** These results are not estimates of real-world predictive performance for public WiFi in Tunja.

**The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes.**

## Periods

Configured in `src/wifi_tunja_smart_predictor/config.py`:

| Split | Rule |
| --- | --- |
| Train | `timestamp` ≤ `2025-06-30 23:59:59` |
| Validation | `2025-07-01` … `2025-09-30 23:59:59` |
| Test | `timestamp` > `2025-09-30 23:59:59` |

Exact min/max timestamps and row counts after generation are stored in `models/model_metadata.json` when you run `python scripts/train_model.py`.

A temporal split is appropriate because each row is a forecast at a point in time. Using future hours to fit imputers, encodings, or trees would leak information that would not exist in an operational setting.

## Temporal split realized in the current generated run

| Split | Rows | Start | End |
| --- | ---: | --- | --- |
| Train | 44,819 | 2024-01-01 00:00 | 2025-06-30 23:00 |
| Validation | 7,547 | 2025-07-01 00:00 | 2025-09-30 23:00 |
| Test | 7,574 | 2025-10-01 00:00 | 2025-12-31 23:00 |

## Model configurations (VERSION 0.2)

| model | notes |
| --- | --- |
| logistic_regression | `max_iter=400`, `random_state=42`, numeric scaling |
| decision_tree | `max_depth=10`, `min_samples_leaf=20` |
| random_forest | `n_estimators=80`, `max_depth=12`, `min_samples_leaf=10` |
| knn | `n_neighbors=7`, numeric scaling |
| naive_bayes | GaussianNB, numeric scaling |

No large hyperparameter search is run in this version.

## Metrics

Comparison table (all models, **test** set): `reports/metrics/model_comparison.csv`

Columns: `model`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `training_time_seconds` (plus `validation_f1` used only for selection).

Precision, recall, and F1 use **HIGH** as the positive class.

The persisted artefact `models/wifi_demand_classifier.joblib` is the pipeline with the highest **validation** F1. Test metrics for that model are also in `reports/metrics/selected_model_test_metrics.json`.

The following values come from the generated `reports/metrics/model_comparison.csv` for this run:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9218 | 0.9171 | 0.8858 | 0.9012 | 0.9785 |
| Decision Tree | 0.9105 | 0.8977 | 0.8776 | 0.8875 | 0.9648 |
| Random Forest (selected) | 0.9246 | 0.9186 | 0.8917 | 0.9049 | 0.9804 |
| K-Nearest Neighbors | 0.9096 | 0.9224 | 0.8465 | 0.8828 | 0.9637 |
| Gaussian Naive Bayes | 0.8725 | 0.8739 | 0.7982 | 0.8344 | 0.9426 |

These test results are generated from the current synthetic dataset and are not real-world estimates. Re-run `python scripts/train_model.py` followed by `python scripts/evaluate_model.py` to regenerate the reports.

## Limitations

- Labels and features are simulated.
- Geographic fields are excluded from the classifier to avoid encoding unique synthetic sites.
- KNN cost grows with training size; hyperparameters are kept small.
- Probabilities are not claimed to be calibrated for real operations.
