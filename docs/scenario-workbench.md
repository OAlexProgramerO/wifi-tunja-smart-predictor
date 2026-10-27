# Geospatial intelligence and scenario workbench

The Geography page summarizes filtered historical **synthetic observations** by zone, hour, weekday, and synthetic access point. Date bounds are inclusive. The map uses built-in Plotly geographic rendering and simulated coordinates; invalid coordinates are excluded from points while their rows remain available to tabular analysis. `connections_next_hour` here is the observed synthetic target in generated history, not a fresh model prediction.

The Scenario Workbench sends a typed comparison to `POST /experiments/what-if`. A request defines one timestamp, a baseline, and one to five alternatives. Each definition must use the same unique zone IDs (up to 12), nonnegative bounded capacity assumptions, and a nonnegative bounded additional-connection budget. The existing prediction service runs once for each selected zone at the shared timestamp. Its predicted demand and classification are held fixed across alternatives. Capacity and allocation calculations are deterministic and do not feed back into the model.

For each zone, the existing greedy planner directs the optional budget to the largest positive `predicted demand - assumed capacity` deficit; ties are resolved by zone ID. Allocations do not exceed the scenario budget. Effective capacity is assumed capacity plus the simulated allocation. The reported deficit is `max(0, predicted demand - effective capacity)` and surplus is `max(0, effective capacity - predicted demand)`. Utilization proxy is `predicted demand / effective capacity * 100`; it is unavailable when effective capacity is zero. This is a connection-count proxy, not measured network utilization. Relative difference is `(alternative - baseline) / abs(baseline) * 100`; when baseline is zero it is unavailable and the report carries a reason code.

Successful comparisons are saved as existing `capacity_scenario` experiment records, with `scenario_kind=what_if_comparison` and the typed result under `result.what_if`. Existing history, retrieval, and JSON/Markdown report routes are reused. JSON report schema 1.2 includes the scenario analysis and the versioned metric registry. Existing scenario creation and comparison APIs remain available.

All points, observations, model outputs, capacities, and allocations are synthetic or hypothetical. No result validates real public WiFi performance or forecast accuracy in Tunja, and capacity changes do not claim to cause a demand change or service improvement.

## API example

```json
{
  "name": "Illustrative zone comparison",
  "datetime": "2025-01-01T10:00:00",
  "baseline": {
    "name": "Baseline",
    "zones": [{"zone_id": "zone_1", "capacity_connections": 20}],
    "resource_budget_connections": 5
  },
  "alternatives": [{
    "name": "Alternative A",
    "zones": [{"zone_id": "zone_1", "capacity_connections": 30}],
    "resource_budget_connections": 5
  }]
}
```

On success the API returns HTTP 201 with the saved experiment record and stable identifier. Retrieve it with `GET /experiments/{id}` and export via `GET /experiments/{id}/report?format=json|markdown`. Validation errors use FastAPI's existing sanitized error handling. The request is bounded by the existing API body limit.

V6.1 adds a second workflow within this workbench: deterministic integer allocation optimization. It shares selected zones, timestamp, prediction service, store, history, and report routes. It supports an unweighted or explicitly weighted synthetic deficit objective, a total budget, per-zone baseline/minimum/maximum values, and a search-step cap. Successful optimal search is globally optimal only for these additive deficit objectives and independent bounds; a step-limited solution is clearly marked unproven. The dashboard shows the baseline and selected per-zone allocations and offers persisted JSON/Markdown downloads. See the [optimization guide](optimization.md) for the formula and an illustrative request.

V6.2.0 development adds sensitivity analysis for saved optimization runs: one-zone-at-a-time synthetic demand multipliers and a separate bounded budget sweep using the unchanged optimizer. The dashboard displays evaluated cases, original and reoptimized objectives, a budget frontier, per-zone stability, partial status, and persisted JSON/Markdown exports. The selected cases do not estimate probabilities or real-world reliability. See [sensitivity and robustness](robustness.md). This work is not released.

## Limitations

- The model predicts one fixed next-hour demand for each selected synthetic zone and timestamp; the workbench does not model interventions, demand elasticity, throughput, latency, cost, or service quality.
- Predictions depend on the installed data and model artifacts. Fingerprint/version omissions mark provenance as not assessable rather than fabricating verification.
- Historical charts describe generated observations and can contain synthetic target values that would not be available at live prediction time.
- Map markers are simulated access points without live basemap tiles; geographic framing is schematic and not proof of actual infrastructure placement.
