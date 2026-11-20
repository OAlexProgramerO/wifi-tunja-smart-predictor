# Reusable ML evaluation policy registry

V8.4 adds immutable, versioned acceptance profiles over the V8.3 evaluation-gates engine. A profile has a stable `policy_id`, incrementing integer version, name, description, tags, creation timestamp, and task-specific criteria. A profile can define classification criteria, regression criteria, or both.

## API

- `POST /ml-experiments/policies` creates a new profile at version 1.
- Posting the same `policy_id` appends the next version; existing versions are never edited or deleted.
- `GET /ml-experiments/policies?limit=20&offset=0` lists the latest version of each profile.
- `GET /ml-experiments/policies/{policy_id}` retrieves the latest version; add `?version=n` to retrieve an immutable historical version.
- `GET /ml-experiments/policies/{policy_id}/versions?limit=100&offset=0` lists immutable versions newest first.
- `POST /ml-experiments/policies/{policy_id}/versions/{version}/evaluate` accepts an `experiment_id` and applies the selected task criteria.

The dashboard displays the latest profile versions, allows creation of a new profile or the next version, and lets the user choose a listed historical version for evaluation.

Evaluation reports identify the exact policy ID/version and include the complete saved profile, experiment ID/timestamps, synthetic dataset fingerprint, chronological split/window, metrics, criterion results, and compatibility evidence. Unsupported tasks, incompatible or incomplete experiment provenance, and missing required metrics produce explicit `INCONCLUSIVE` results. Only completed, compatible synthetic records can pass or fail criteria.

Policies are persisted in an additive table in the existing ML experiment SQLite database. Creating a policy version never updates earlier versions. Policy evaluation reuses the V8.3 metric threshold and compatibility engine. It does not retrain, promote, or modify models or experiment records. Official MinTIC records are excluded.

## Evaluation run history (V8.5)

Every call to the versioned evaluation endpoint creates an immutable run record, including `INCONCLUSIVE` outcomes. The record stores a unique run ID and UTC timestamp, experiment ID, exact policy ID/version, status, complete criterion-level evidence, and the saved report with synthetic dataset and chronological evaluation provenance. A later policy version does not alter previous reports.

Use `GET /ml-experiments/policy-runs` with `limit`, `offset`, and optional `policy_id`, `policy_version`, `experiment_id`, `status`, `created_from_utc`, or `created_to_utc` filters. Dates are inclusive UTC-aware ISO timestamps. Retrieve a run with `GET /ml-experiments/policy-runs/{run_id}` or download its saved JSON from `/report`.

Compare two run IDs with `POST /ml-experiments/policy-runs/compare`. Runs must use the same experiment, policy ID, task, synthetic fingerprint, evaluation split/window, evaluator version, and metric/operator set; the policy versions and thresholds may differ. Compatible comparisons show threshold and outcome changes without ranking models. Missing metric evidence returns `PARTIAL`. Incompatible context returns `INCOMPATIBLE`, clear reason codes, and no criterion comparisons. The dashboard exposes these filters, run details, JSON download, and two-run comparison. Evaluation and comparison use synthetic ML data only and do not modify policies, experiments, or models.
