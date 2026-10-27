# V6.2.0 — Sensitivity and robustness analysis

V6.2 extends the existing Scenario Workbench and V6.1 bounded integer optimizer. A saved allocation optimization can be analyzed with one-factor-at-a-time (OAT) synthetic demand stresses and a bounded budget sweep. The analysis reuses the saved demand estimates and optimizer; it does not rerun model inference, change the optimizer, or alter model artifacts.

## Method and interpretation

For each stored synthetic zone estimate \(d_i\), an OAT case changes one zone only to \(d'_i=m d_i\), where the user selects a multiplier from 0.25 through 2.0. The stored decimal estimate is multiplied without integer rounding. Other zone estimates remain fixed. The reoptimizer uses the original objective, priority weights, capacities, and per-zone bounds, with a bounded integer budget. Demand and budget effects are not combined exhaustively.

The analysis separately evaluates the original reference allocation under each varied assumption and the reoptimized allocation under that same assumption. Objective degradation is the reoptimized score minus the original allocation's score for that case; its percentage uses the absolute original score as denominator and is null when that score is zero. This distinguishes sensitivity of the reference allocation from reoptimization results.

The budget frontier reports only budgets actually evaluated; it does not interpolate. Minimum additional budget to satisfy the declared hard minimum allocations is exactly `max(0, sum(zone minimum allocations) - tested budget)`. It is a constraint calculation, not a recommendation.

Per-zone stability is the share of assessable evaluated cases with a feasible optimizer candidate in which that zone's allocation equals the saved reference allocation. Each zone reports its denominator, unchanged count, mean absolute change, and maximum absolute change. The “most sensitive” ordering is descriptive and sorts by maximum, then mean absolute change, then zone ID.

Feasibility rate is feasible evaluated cases divided by all evaluated cases. Parent integrity/provenance assessability is reported separately. The tested cases are deliberately selected scenarios, not a random sample: counts and rates are not probabilities, confidence intervals, calibrated worst cases, or estimates of real operational reliability.

## Limits and bounds

- The saved parent must be an allocation optimization record. Missing/invalid integrity prevents a verified assessment; numeric results are still marked with their provenance status.
- At most 12 zones, 6 multiplier values including the inserted 1.0 reference, 20 budget values including the inserted parent budget, 100 scenario evaluations, and 10,000 optimizer steps per evaluation are accepted.
- The multiplier range and budget range are inclusive. The multiplier 1.0 and parent budget are included as reference cases when omitted from the entered ranges. If the evaluation cap omits cases, or any inner optimizer reaches its step cap, the analysis is marked partial and reports the evaluated denominator and incomplete optimizer count.
- `search_limit_reached` is retained for each optimizer run; it does not establish optimality.
- `optimizer_incomplete_scenario_count` counts evaluated cases whose inner optimizer stopped at its step limit; a returned candidate may be feasible, but optimality is unproven.
- Infeasibility under hard minimums is explicit. Additional budget is computed from those minimums; maxima and demand do not change this calculation.
- Data, demand estimates, capacities, objectives, budgets, and allocations are synthetic or hypothetical. Nothing here has been validated against real public WiFi demand or infrastructure in Tunja.

## API example

```json
{
  "name": "bounded synthetic stress check",
  "parent_experiment_id": "<saved-optimization-id>",
  "demand_multiplier_min": 0.8,
  "demand_multiplier_max": 1.2,
  "demand_multiplier_step": 0.2,
  "budget_min_connections": 0,
  "budget_max_connections": 1000,
  "budget_step_connections": 100,
  "max_evaluations": 100,
  "max_allocation_steps_per_evaluation": 10000
}
```

Submit to `POST /experiments/robustness`. The result is saved in the existing SQLite experiment store and retrieved/exported through the existing history and report routes. Report schema 1.4 adds `robustness_analysis`; older records remain readable.
