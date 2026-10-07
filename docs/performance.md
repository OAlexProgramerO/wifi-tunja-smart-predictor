# Runtime performance (V3.6)

## Scope and method

V3.6 measured current local workloads before optimization, inspected API and Streamlit resource lifetimes, and removed repeated full-dataframe work where the request path showed a clear cost. Measurements below use the repository synthetic dataset (59,940 processed rows, 67 columns; approximately 35.19 MiB as a pandas frame) and the local Windows development environment. Times are observations, not service-level guarantees; machine load and filesystem/model cache state affect them. No network-dependent benchmark or timing-threshold test was added.

## Findings and changes

- Dataset loading, model loading, and service initialization already have bounded reuse at their entry points: API services and location catalogs use one-entry LRU caches; model loading is cached; Streamlit caches dataframe and service resources; assistant sessions are bounded to 1,024 entries and a 60-minute TTL. These boundaries remain in place. V3.6 added no global or query-result cache.
- Before the change, the assistant dataset-query route built a new `DatasetQueryEngine` for every request. A standalone initialization copied and normalized the full dataframe and took a median of about 31.8 ms in four baseline samples. The route now reuses the already initialized `AssistantService.queries`, whose read-only frame is the service's normalized frame. Query execution itself varied around 25–43 ms, so it was not separately optimized or cached.
- Before the change, every scenario prediction's explanation rebuilt `HistoricalAnalogEngine`, copied the full dataset, selected matching rows, and recomputed medians/modes already used by scenario construction. The builder now records those same deterministic analog baselines in its private context; explanation reuses them.
- Feature sensitivity scoring now converts the one-row feature input to object dtype once and copies only that small row for each isolated model evaluation, avoiding repeated conversion.

## Measured evidence

A same-process paired measurement used the same loaded dataset/models, same downtown scenario at 18:00, warmed both old and optimized explanation paths, and alternated 10 runs per path. Median end-to-end scenario prediction was 247.22 ms with the previous explanation path and 210.09 ms with the optimized path (about 15% lower in this sample). The full prediction dictionaries were equivalent within a 1e-12 floating-point tolerance, treating missing numeric values as equivalent; observed differences were limited to tiny floating-point inference variations. This is a local sample, not a general performance guarantee.

The baseline full-frame query-engine initialization measured about 31.8 ms median. The optimized assistant route avoids that initialization and copy on each query by reusing its existing engine. Query execution time measurements were noisy and did not establish a meaningful execution-time improvement.

## Dataset, model, dashboard, and memory boundaries

No dataset/model loading mechanism was replaced because the application already reuses these resources through bounded API and Streamlit caches. Metadata loaded by a cached service remains stable for the service lifetime. The dashboard sections, displayed meanings, and per-session state are unchanged. The assistant query engine shares the service-owned normalized frame and is used read-only by its query path; standalone `DatasetQueryEngine(frame)` keeps its defensive copy by default. No query cache, new dependency, or unbounded structure was introduced. V3.5 request validation, allowlists, CORS, security headers, safe errors, and session bounds remain in place.

## Not optimized

The measured query execution path still spends time filtering and aggregating pandas data per query. There is no repeated-query cache because the baseline did not justify its invalidation and memory complexity. Scenario building still performs analog selection once per request, and model inference remains synchronous. Cross-worker resource sharing, production load behavior, and dashboard interaction latency were not benchmarked. These are candidates for future profiling only if real workloads show a need.
