# Trained model artefacts

This directory stores pipelines produced by `python scripts/train_model.py`.

Typical files:

- `wifi_demand_classifier.joblib` — sklearn `Pipeline` (feature engineering + preprocessing + classifier)
- `model_metadata.json` — training periods, selected model name, and synthetic test metrics

Artefacts are generated locally and are not required to run unit tests.

**Disclaimer:** The dataset is synthetic and was generated for software development, machine learning experimentation, demonstration, and portfolio purposes. Metrics stored here describe performance on the synthetic evaluation dataset only.
