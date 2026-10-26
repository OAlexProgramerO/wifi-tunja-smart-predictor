# V6.1.0 — Scenario allocation optimization

V6.1 extends the existing scenario workbench with deterministic integer allocation optimization. It uses the current model prediction service and experiment store; it does not train or modify models, add a solver dependency, or claim operational recommendations.

## Inputs and objectives

The API `POST /experiments/optimize` accepts a scenario timestamp, one to twelve synthetic zones, base capacity, optional baseline allocation, integer per-zone minimum and maximum allocations, an integer total budget, an objective, and a maximum allocation-step count. The weighted objective requires one positive priority weight per zone. The unweighted objective rejects weights so assumptions cannot be silently ignored.

For each zone `i`, the service receives fixed model-estimated synthetic demand `d_i`, base capacity `c_i`, and integer allocation `x_i`. The resulting deficit is `q_i = max(0, d_i - c_i - x_i)`. It supports:

- `MINIMIZE_TOTAL_DEFICIT`: minimize `sum(q_i)`.
- `MINIMIZE_WEIGHTED_DEFICIT`: minimize `sum(w_i * q_i)`, where each positive `w_i` is supplied explicitly.

Demand is distinct from allocated simulated capacity. Model demand remains fixed while allocating. Cost, latency, throughput, energy, coverage and network quality are not optimized because no corresponding valid models are available.

## Method and optimality

The pure service starts at each zone's minimum and repeatedly assigns one integer unit to the zone with the greatest marginal deficit reduction, using ascending `zone_id` as a deterministic tie-break. It stops when no unit can improve the chosen deficit objective, the budget is spent, all zone maxima are reached, or the configured step limit is reached.

For the supported additive objectives, marginal reductions are non-increasing, per-zone bounds are independent, and there is one shared budget. The heap method therefore proves a global optimum when it completes without reaching its step limit. When beneficial allocation remains at the step limit, the returned candidate is feasible but is explicitly marked `search_limit_reached`; optimality is not claimed. If the sum of zone minimums exceeds the budget, the request is infeasible and no selected candidate is returned. Equal marginal values are assigned using lexical zone ID.

The candidate counters describe the minimum-feasible starting allocation plus each accepted one-unit allocation state, not an enumeration of every possible vector. Infeasible candidate count is one for a hard-infeasible problem and zero for a feasible greedy search. These counters describe the bounded search path.

## Worked synthetic example

Suppose two illustrative zones have fixed predicted demands of 8 and 5 connections, base capacities of 3 and 4, zero minimum allocations, and a shared budget of 3 integer units. Their starting deficits are 5 and 1. Under the unweighted objective, each unit reduces a deficit by at most one; deterministic tie-breaking assigns all three units to the first zone if its ID sorts first. The resulting deficits are 2 and 1 (total 3), compared with baseline total deficit 6. This is an illustrative arithmetic example, not a real Tunja estimate or recommended deployment.

## Persistence, reports, and provenance

Runs are persisted as additive JSON fields in the existing SQLite `capacity_scenario` record format, with `scenario_kind=allocation_optimization`; there is no second database or schema migration. Retrieve records with `GET /experiments` and `GET /experiments/{id}`. Export JSON or Markdown with `GET /experiments/{id}/report?format=json|markdown` (report schema 1.3). The report includes objective, bounds and budget, baseline, selected plan or infeasible status, per-zone fixed predicted demand and deficits, reason codes, search limits, model/dataset provenance, and limitations.

The Streamlit Scenario Workbench offers an optimization workflow alongside existing what-if comparison. It supports English and Spanish labels, explicit constraints, save/retrieve, and JSON/Markdown download.

## Interpretation limits

The dataset, locations, predictions, and allocations are synthetic. Results are conditional on recorded model outputs and do not establish real public WiFi demand, service, infrastructure needs, or forecast validity in Tunja. Mathematical optimality applies only to this specified synthetic deficit objective and constraints. Incomplete model or dataset provenance is surfaced as not assessable rather than repaired or inferred.
