# Trained model artefacts

This directory stores pipelines produced by `python scripts/train_model.py`.

Typical files:

- `wifi_demand_classifier.joblib` — sklearn `Pipeline` (feature engineering + preprocessing + classifier)
- `wifi_demand_regressor.joblib` — independent sklearn `Pipeline` for `connections_next_hour`
- `model_metadata.json` — combined version, temporal periods, and artifact references
- `classification_metadata.json` / `regression_metadata.json` — target-specific model and evaluation metadata

Artefacts are generated locally and are not required to run unit tests.

**Disclaimer:** The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes. Metrics stored here describe performance on the synthetic evaluation dataset only.
