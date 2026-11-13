# ML artifact inventory (V7.8.0)

The main API exposes `GET /model-artifacts/inventory`. The existing About page displays a compact summary with a per-artifact detail expander. The route reads the classifier and regressor files, their training metadata, the selected-model evaluation report, and the current synthetic dataset fingerprint. It does not train, deserialize, overwrite, repair, or otherwise modify model files.

## Status meanings

- **READY:** Required artifact and metadata files exist, metadata is readable and consistent, recorded integrity fingerprints match the available bytes, and recorded dataset compatibility checks agree.
- **MISSING:** A required classifier/regressor artifact file is absent.
- **INCOMPLETE:** Metadata or an evaluation report is missing, malformed, internally inconsistent, or too old to verify compatibility.
- **RETRAINING_REQUIRED:** A recorded training dataset version or integrity fingerprint conflicts with the current synthetic assets. Review the dataset/model lineage and run the project's approved training/evaluation workflow explicitly; this endpoint never starts it.

Overall status summarizes the most serious artifact state. A stale evaluation report can be incomplete even when both artifact files remain present. Package version and recorded model-training version are separate values; a difference alone does not imply that a model must be retrained.

## Inventory fields

The JSON response identifies each artifact by its role and safe display name, then reports availability, state, recorded model version, selected estimator, training timestamp, synthetic dataset and preprocessing versions, training/validation/test periods, selected validation/test metrics, current SHA-256 fingerprint, recorded fingerprint when present, and compatibility checks. It omits absolute paths and unrelated environment configuration. The evaluation section reports its own status, generation time, dataset fingerprint, and model fingerprints without returning the full evaluation report.

SHA-256 checks compare the current local bytes to recorded fingerprints when those are available. Dataset compatibility is limited to the fingerprints and dataset-version fields actually recorded by this project. An absent training-data fingerprint is reported as unverified; the service does not invent one or infer it from a later evaluation.

## Data boundary and limitations

All model training/evaluation metadata describes synthetic data only. This inventory does not read, compare, or combine official MinTIC connectivity records. A matching fingerprint proves byte identity with the recorded file, not scientific validity or predictive performance for real WiFi demand in Tunja. READY is an integrity/metadata state, not a quality endorsement. No automatic repair or retraining is performed.
