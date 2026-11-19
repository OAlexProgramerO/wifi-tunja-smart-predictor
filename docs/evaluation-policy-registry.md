# Reusable ML evaluation policy registry

V8.4 adds immutable, versioned acceptance profiles over the V8.3 evaluation-gates engine. A profile has a stable `policy_id`, incrementing integer version, name, description, tags, creation timestamp, and task-specific criteria. A profile can define classification criteria, regression criteria, or both.

## API

- `POST /ml-experiments/policies` creates a new profile at version 1.
- Posting the same `policy_id` appends the next version; existing versions are never edited or deleted.
- `GET /ml-experiments/policies?limit=20&offset=0` lists the latest version of each profile.
- `GET /ml-experiments/policies/{policy_id}` retrieves the latest version; add `?version=n` to retrieve an immutable historical version.
- `POST /ml-experiments/policies/{policy_id}/versions/{version}/evaluate` accepts an `experiment_id` and applies the selected task criteria.

Evaluation reports identify the exact policy ID/version and include the complete saved profile, experiment ID/timestamps, synthetic dataset fingerprint, chronological split/window, metrics, criterion results, and compatibility evidence. Unsupported tasks, incompatible or incomplete experiment provenance, and missing required metrics produce explicit `INCONCLUSIVE` results. Only completed, compatible synthetic records can pass or fail criteria.

Policies are persisted in an additive table in the existing ML experiment SQLite database. Creating a policy version never updates earlier versions. Policy evaluation reuses the V8.3 metric threshold and compatibility engine. It does not retrain, promote, or modify models or experiment records. Official MinTIC records are excluded.
