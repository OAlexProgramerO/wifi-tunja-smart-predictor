# Synthetic feature drift monitoring — V8.0.0

## Data and leakage boundary

Drift compares the model's allowlisted `MODEL_INPUT_COLUMNS` from the chronological training partition against a separately selected `validation` or `test` partition of the same synthetic dataset. The target columns `demand_level` and `connections_next_hour` are never read as features; the monitor does not accept an arbitrary feature list. It also excludes identifiers and every column outside the explicit model-input allowlist. Lag features are safe under the generator's documented rule that they use only observations strictly before each row's prediction timestamp.

The official MinTIC database and its snapshots are not used. The monitor does not load or execute either ML model, retrain, change artifacts, or alter predictions.

## Statistics and status thresholds

- Numeric columns report reference/evaluation mean, median, standard deviation, missing proportions, invalid numeric counts, and Population Stability Index (PSI). PSI uses up to ten quantile bins derived from the reference partition and a small fixed smoothing value for empty bins.
- Categorical columns report category counts, total-variation distance over the union of observed values, missing proportions, and categories absent from the reference but present in evaluation.
- The default minimum is 20 non-missing values in both partitions. Smaller samples are labeled `INSUFFICIENT_SAMPLES`; they do not produce a drift statistic.
- A numeric PSI of 0.10–0.20, normalized mean shift of 1–2 reference standard deviations, categorical total-variation distance of 0.10–0.25, missing-proportion difference of 0.05–0.20, or any unseen category yields `WARNING`. PSI ≥ 0.20, mean shift ≥ 2 reference standard deviations, total-variation distance ≥ 0.25, missing-proportion difference ≥ 0.20, or unseen-category share ≥ 0.10 yields `DRIFT_DETECTED` for that feature.
- Overall status is `DRIFT_DETECTED` if any feature meets its drift threshold; otherwise `WARNING` if any feature warns or lacks enough samples; otherwise `STABLE`.

If an approved model input column is absent from either frame, it is listed as `MISSING_FEATURE` and the overall report is at least `WARNING`.

These are fixed descriptive thresholds, not hypothesis tests. PSI and total variation can be sensitive to bin/category definitions and sample size; no multiple-comparison correction or model-performance test is performed. A drift signal does not prove model degradation, causality, real-world distribution change, or changing public WiFi demand in Tunja.

## API and persisted reports

`GET /drift/monitor?evaluation_split=validation|test` computes a fresh report from the available synthetic dataset and does not write files. The source, reference, and evaluation fingerprints hash only allowlisted model input features; target values never enter the fingerprints. `scripts/generate_drift_report.py --evaluation-split validation|test` explicitly stores a versioned JSON report under `reports/metrics/drift/`, including feature-only fingerprints, generation time, feature names, split, thresholds, and sample configuration. The JSON file is generated output and is not a training artifact. Run the script separately for each evaluation split.
