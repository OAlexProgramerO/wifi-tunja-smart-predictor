# Scenario prediction

The V3 scenario API accepts a local date/time and one synthetic location selector (friendly zone name, zone ID, AP ID, or paired coordinates). The builder resolves the location to a synthetic AP and creates the 40 model inputs from deterministic historical analog rows. It excludes both outcomes from the feature contract.

Requests within the dataset time range are labeled `HISTORICAL_REPLAY`; observations later than the request time are excluded from analog selection. Times outside the dataset are labeled `SYNTHETIC_SCENARIO`; analogs are sampled deterministically from the available synthetic history. No live provider is called.

`POST /scenario/predict` on the main API returns classification probabilities, predicted `connections_next_hour`, capacity-proxy estimates, scenario provenance, and a split-conformal interval. The interval radius is the finite-sample absolute-residual quantile calculated on validation predictions at the configured 90% nominal confidence. It is not a guarantee under distribution shift.

The feature summary compares the prediction with one-feature-at-a-time analog baselines. It reports local model sensitivity, not causal effects. For model-input details use `Advanced Prediction`; typical users only need a location and time.

## Retrain and evaluate

Run `python scripts/train_model.py` then `python scripts/evaluate_model.py`. Classifier and regressor artifacts and their metadata are separate. Both use the same chronological train/validation/test partitions. Candidate selection uses validation metrics; the test partition is reserved for final reporting.
