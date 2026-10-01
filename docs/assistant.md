# Deterministic assistant

The assistant is a separate FastAPI app, started with `uvicorn assistant_api.main:app --reload --port 8001`. It provides `/health`, `/suggestions`, `/chat`, and `/dataset/query`. It runs without an LLM key and reports the internal tool sources used for each answer.

`POST /chat` accepts a message, an optional session ID, and structured location/time context. The process-local bounded session store retains resolved location and scenario details for follow-up questions; it is not durable across restarts or shared between workers. Supported questions route to fixed location, historical aggregate, model metadata, scenario prediction, explanation, limitations, and dashboard-help tools.

Before technical routing, the assistant handles deterministic `GREETING`, `IDENTITY`, and `CAPABILITIES` intents in English and Spanish. Examples include `hi`, `hola`, `who are you?`, `quién eres?`, `what can you do?`, and `qué puedes hacer?`. Matching is case-insensitive and ignores basic punctuation and whitespace around the message. These replies require no dataset query or model invocation.

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
