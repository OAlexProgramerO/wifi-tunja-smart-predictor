# Synthetic capacity planning

`POST /experiments/scenarios` evaluates 1–12 synthetic `zone_id` values at one timestamp. Each zone receives a user-supplied `capacity_connections` assumption and the request can include a finite `resource_budget_connections`. The existing deterministic scenario service supplies the regressor estimate for next-hour connections; the capacity planner then derives headroom, overload, unmet demand, and utilization.

## Units and rules

- Demand, capacity, headroom, overload, unmet demand, and resource budget are **connection counts**. They do not represent Mbps, airtime, coverage, or engineered throughput.
- Utilization is `demand / effective_capacity * 100`. At zero effective capacity it is `null`, avoiding division by zero.
- Optional resource budget is assigned greedily to the zones with the largest `max(0, demand - configured capacity)`; ties sort by `zone_id`. Allocation never exceeds the remaining budget or the zone's estimated deficit.
- This deterministic baseline is explainable, but is not claimed to be optimal. It represents a hypothetical synthetic scenario, not a recommendation for municipal infrastructure.
- Demand estimates come from the currently loaded model artifacts and synthetic historical analogs. The stored experiment includes per-zone model traceability.

The request is typed and bounded; unknown fields are rejected. A persisted experiment can be retrieved and exported in JSON or Markdown. Model predictions remain separate from user-supplied capacity assumptions and rules-derived metrics.

All data, locations, estimates, and scenarios are synthetic. No result establishes forecast validity for real WiFi demand in Tunja.

## V6.1 bounded integer optimization

The existing scenario workbench also offers an optimizer for fixed next-hour synthetic demand. It minimizes either total deficit or an explicitly weighted total, under an integer shared budget and independent per-zone minimum/maximum allocations. It uses deterministic diminishing marginal benefit with lexical `zone_id` ties. Under these constraints, a completed search proves a global optimum for the selected deficit function; a configured step limit returns only a feasible best-so-far candidate and explicitly marks optimality unproven. Minimum allocations exceeding the budget yield an infeasible result with no recommendation. This is a mathematical result for hypothetical synthetic connection counts, not an operational or economic network plan. See [optimization methodology](optimization.md).
