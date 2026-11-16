# Reproducible ML experiment orchestration

## Scope

The orchestration service runs one allowlisted classifier or regressor candidate against the synthetic dataset using the project's chronological training cutoff. It evaluates on either the existing validation or test period. Targets are used only as labels, never as features. The service does not read MinTIC snapshots, persist fitted artifacts, retrain production models, or change prediction behavior.

Each record has a UUID, UTC creation/update times, task, candidate, bounded parameters, execution state, dataset fingerprint, temporal partition boundaries, evaluation metrics, and an artifact fingerprint only if an artifact was actually persisted. V8.1 does not persist candidate artifacts, so that fingerprint is null and the report says so. Failure details contain a sanitized exception type and generic message, never exception text, paths, request bodies, or credentials.

## Configuration and lifecycle

An experiment configuration contains `task` (`classification` or `regression`), an allowlisted `model_candidate`, `evaluation_split` (`validation` or `test`), and a small primitive-only `hyperparameters` object. Candidate names and parameter names are checked against existing estimator specifications; values are bounded before fitting. The shared `temporal_split` function and project cutoffs are reused. A dataset fingerprint and split bounds identify the inputs and temporal scope. An incompatible idempotency-key replay is rejected; a matching replay returns the original record.

Example:

```json
{
  "task": "regression",
  "model_candidate": "random_forest_regressor",
  "evaluation_split": "validation",
  "hyperparameters": {"n_estimators": 120, "max_depth": 12}
}
```

```text
validated configuration -> PENDING -> RUNNING -> COMPLETED
                                              \-> FAILED
```

The FastAPI request executes synchronously and persists each state transition to a dedicated `ml_experiments` table in the existing local SQLite experiment database. This table is additive; existing scenario records and schema are unchanged. A process interruption may leave a record in `RUNNING`; there is no background worker or automatic retry in this release. Results, configuration, and safe errors are JSON encoded. Reports are generated from saved records without rerunning training.

## API and reports

- `POST /ml-experiments` validates and runs one configuration. Supply `X-Idempotency-Key` to make retries safe.
- `GET /ml-experiments?limit=20&offset=0` lists newest records with bounded pagination.
- `GET /ml-experiments/{experiment_id}` retrieves one record.
- `GET /ml-experiments/{experiment_id}/report` exports its JSON report.

The Streamlit Experiment Orchestration page submits a run, inspects history/details, and downloads its report. This is local synchronous execution, not distributed scheduling. Reports use `TRACEABLE_NOT_SELF_CONTAINED` and do not claim exact replay was verified: reproduction requires the same code, recorded library versions, and synthetic dataset bytes. Test-split runs are for explicit inspection, not iterative model selection. Results are not evidence of real-world forecast validity; official MinTIC data is not an input.
