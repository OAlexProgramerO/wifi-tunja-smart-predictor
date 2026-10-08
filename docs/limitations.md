# Limitations, privacy, and security

## Scientific limitations

Every AP, coordinate, observation, environmental value, network snapshot, and outcome is synthetic. Metrics describe only a held-out period from that synthetic generator and do not establish accuracy for real users or networks. No live telemetry, weather feed, event service, municipal inventory, or external geocoder is connected.

The classifier predicts the synthetic `demand_level` target. The regressor estimates synthetic `connections_next_hour`. Capacity utilization treats the predicted connection count as a proxy; actual airtime, device capability, throughput, channel contention, and service quality are not modeled. Local feature replacement is sensitivity analysis, not causal attribution. Conformal coverage can change with distribution shift.

## Privacy

The project does not request personal identifiers or precise real user location. Scenario coordinates are matched locally against the synthetic catalog. Assistant context is held in a bounded in-process memory store for up to one hour and is lost on process restart; deployers should avoid passing sensitive context and should use a production session store only after reviewing retention requirements.

## Security and operational notes

Request schemas reject extra prediction fields. Assistant aggregate queries use fixed allowlists and do not execute arbitrary code or SQL. Keep local `.env` and `.streamlit/secrets.toml` files out of version control. The development servers are intended for local demonstration; production deployment needs authentication, TLS, rate limiting, monitoring, and durable-state/privacy review.

See [Security baseline](security.md) for the V3.5 controls and remaining deployment work.

## Runtime performance limits

V3.6 removes repeated full-frame query-engine setup from assistant dataset-query requests and repeated full-history analog selection from scenario explanations. Actual request latency still depends on dataset size, hardware, model inference, and concurrent load. There is no query-result cache, process-wide cross-worker cache, production load test, or latency guarantee. See [performance notes](performance.md).

## Observability limits

V3.7 emits structured JSON diagnostics to the configured application logging destination and correlates requests in process. It does not provide durable audit retention, centralized log aggregation, distributed tracing, metrics dashboards, log rotation, or a production monitoring service. Request IDs are correlation labels, not credentials. Avoid forwarding logs to an external system without reviewing its privacy and retention behavior. See [observability](observability.md).
