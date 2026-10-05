# Deterministic assistant

The assistant is a separate FastAPI app, started with `uvicorn assistant_api.main:app --reload --port 8001`. It provides `/health`, `/suggestions`, `/chat`, and `/dataset/query`. It runs without an LLM key and reports the internal tool sources used for each answer.

`POST /chat` accepts a message, an optional session ID, and structured location/time context. The process-local bounded session store retains resolved location and scenario details for follow-up questions; it is not durable across restarts or shared between workers. Supported questions route to fixed location, historical aggregate, model metadata, scenario prediction, explanation, limitations, and dashboard-help tools.

Before technical routing, the assistant handles deterministic `GREETING`, `IDENTITY`, and `CAPABILITIES` intents in English and Spanish. Examples include `hi`, `hola`, `who are you?`, `quién eres?`, `what can you do?`, and `qué puedes hacer?`. Matching is case-insensitive and ignores basic punctuation and whitespace around the message. These replies require no dataset query or model invocation.

## Context-aware demand questions (0.3.2)

The deterministic `DEMAND_SCENARIO` intent recognizes simple demand/connection questions with existing synthetic zones and friendly aliases (for example, downtown/center/centro, north/norte, south/sur, university/universidad, commercial/comercial, and transport/transporte). Supported times include `18:00`, `6 PM`, `6PM`, `18`, `6 de la tarde`, and `7 de la noche`. The assistant maps these to the canonical 24-hour value and calls the existing `ScenarioPredictionService`; it does not recreate prediction logic.

When a follow-up omits its zone, the assistant reuses location stored in the existing session context. It asks for a zone if none is available, rejects unrecognized zones rather than inventing one, and requests a valid time for malformed time input. Responses summarize the returned demand class, next-hour connection estimate, HIGH probability, capacity utilization, and interval when those values are available. All estimates remain synthetic.

Dataset queries validate metric, columns, filters, and groupings against explicit allowlists. The service never evaluates user-provided code or SQL. Responses are grounded in the synthetic dataset or generated model metadata and include a synthetic-data disclosure. Unsupported filters receive a validation error.

Example:

```json
{
  "message": "What is the expected demand downtown today at 6 PM?",
  "session_id": "optional-client-session",
  "context": null
}
```

Optional future LLM and live providers are intentionally not activated in this release. A deterministic provider remains the default and only configured answer path.

## Dashboard-aware explanations (0.3.3)

The existing chat context accepts optional `dashboard_section` and `scenario_result` fields inside `context`. Section names are validated against the dashboard's nine existing sections. The bounded process-local session store retains these values alongside the V3.2 location and time so short follow-ups such as “Explain this” can use the active context. Without a recognized section, the assistant asks which section to explain.

The assistant explains Overview, Live Scenario, Demand Explorer, Geographic Analysis, Network Analysis, Model Performance, AI Assistant, Advanced Prediction, and About in English and Spanish. It also describes accuracy, precision, recall, F1, ROC-AUC, MAE, RMSE, and R² conceptually. No metric values are fabricated. Scenario explanations use the supplied or previously returned scenario result rather than running inference again. Local sensitivity factors are described as associations, not causes. All data, locations, outcomes, and performance metrics remain synthetic and do not represent real Tunja WiFi telemetry.

The Streamlit page passes the last visited dashboard section and the currently displayed scenario result through the existing chat context. The assistant remains deterministic and requires no LLM credential.

## Historical demand analysis (0.3.4)

Historical questions use the existing allowlisted `DatasetQueryEngine` over loaded synthetic observations. The assistant can report average `connections_next_hour` separately from the `demand_level` HIGH rate, compare two resolved synthetic zones, rank zone demand, compare morning/evening or weekday/weekend, and identify peak hours or weekdays. “How has demand changed?” compares the earliest and latest available year/month groups and reports only when at least two distinct periods exist. A single remembered zone is reused for a historical follow-up; comparisons do not overwrite the active scenario location/time or run prediction inference.

Unknown zones are rejected instead of mapped to an arbitrary place, and unsupported/empty time filters receive a deterministic clarification. Historical responses explicitly identify simulated records and are not claims about real Tunja WiFi use. Classification frequency and next-hour connection averages are labeled separately.
