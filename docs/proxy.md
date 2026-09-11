# Proxy: `POST /v1/chat/completions`

The OpenAI-compatible inference entry point. Point any OpenAI-compatible client at orla and add a stage header.

## Wire shape

Request: standard OpenAI chat completion plus identity metadata. See [`concepts.md`](concepts.md#identity-tags) for the full tag list.

Response: standard OpenAI chat completion. The response's `model` field reports the **resolved backend name**, not necessarily what the client sent in its `model` field. Divergence is by design under the developer and platform engineer split documented in [`concepts.md`](concepts.md).

Streaming follows OpenAI's data-only SSE format, terminated by `data: [DONE]`.

## Request handling order

The handler runs these checks in order. Each is a 400 unless noted.

1. **Check Content-Type.** Every `POST`, `PUT`, and `PATCH` across the API must send `application/json`. Anything else returns 415. `GET`, `HEAD`, and `DELETE` are unchecked.
2. **Decode body.** Body too large, over 10 MB, returns 400.
3. **`messages` non-empty.**
4. **Stage extracted.** From `X-Orla-Stage` header, falling back to `metadata.orla.stage` in the body. Missing returns 400.
5. **Resolve backend.** `registry.GetOrCreate(stage)` auto-creates a default stage record on first sighting. If the request carries `X-Orla-Mapping` and that variant overrides this stage, use the variant's backend, and the stage mapper is not consulted. Otherwise, if a dynamic stage mapper is configured, ask it (see below). Otherwise use `stage.Backend`. If that is empty, fall back to `req.Model`. If nothing resolves, return 400.
6. **Apply inference policy** from the stage record. This sets `reasoning_effort` and, when the stage has a `prompt`, substitutes it for the leading instruction message.
7. **Convert messages and tools** to the internal model types.
8. **Dispatch** via `LayerExecute`, then `BackendManager.ScheduleChat`, into the per-backend queue and a worker that calls the openai-go provider.
9. **Encode response** as OpenAI chat completion or stream chunks.

## Auto-create on first sighting

If a developer uses a stage id that orla has never seen, the daemon inserts a default stage record with no backend and the request falls back to `req.Model` for that one call. The platform engineer can later `PUT /api/v1/stages/{id}` to map it.

This means a developer can deploy new agent code without coordinating with the platform engineer. Their requests still flow, and the mapper picks them up on the next pass.

## Mapping variants and shadow testing

A mapping variant is a named, sparse set of per-stage backend overrides layered on the live stage mapping. A request selects a variant with the `X-Orla-Mapping` header, falling back to `metadata.orla.mapping`. When the variant overrides the request's stage, its backend wins. A stage absent from the variant falls through to the stage's live backend, so a candidate that differs from live on one stage is a single-entry variant.

The optimizer uses variants for shadow testing. The live critical path sends no header and resolves the live mapping. A shadow request sends `X-Orla-Mapping` naming a candidate variant, so it runs the same workload under a different mapping without disturbing the live one. Both can stream concurrently because nothing mutates the live mapping. The variant name is recorded on each completion, so cost separates by mapping even when critical and shadow traffic interleave. Manage variants with `POST/GET/DELETE /api/v1/mappings`. See [`storage.md`](storage.md).

## Stage prompt override

A stage can carry a `prompt`. When it is non-empty the proxy substitutes it for the request's leading instruction message before forwarding. The instruction message is the system or developer message the request opens with, since SDKs differ on which role they use for instructions. If the first message is a system or developer message its content is replaced and its role kept. Otherwise a system message is prepended. The rest of the conversation is untouched, so a tool-calling loop keeps its accumulated scratchpad and only the instructions change.

The override is opt-in. A stage with an empty prompt forwards the client's messages verbatim, so an agent that does not want orla managing its prompt is unaffected. The contract is "the stage's prompt is the leading instruction message," which holds for a single-shot stage like a composer and for a multi-step loop whose every call repeats the same instruction message. An agent that hides its instructions in a user turn or a tool description should leave the stage prompt empty and apply the prompt itself. See [`prompts.md`](prompts.md).

## Identity tags become completion-record dimensions

Every dispatched request results in one row in `completion_records` with the following columns:

- `completion_id`: UUID assigned by orla
- `stage_id`: from the request
- `workflow_run`: from the request, nullable
- `backend`: the resolved name
- `mapping`: the variant that served the request, empty for the live critical path
- `tags_json`: the full `X-Orla-Tag-*` map
- `prompt_tokens`, `completion_tokens`, `latency_ms`, `cost_usd`, `status`, `created_at`

This is the mapper's primary observation channel. See [`storage.md`](storage.md).

## Dynamic stage mapper

A stage's backend is normally the static mapping set with `orlactl stage map`. A dynamic stage mapper makes that choice per request instead, by asking an external service to map the stage. Configure it with `orlactl stage mapper set --url http://mapper:8091/v1/map --timeout-ms 50`, read it back with `orlactl stage mapper show`, and revert to static routing with `orlactl stage mapper disable`. The setting is control-plane state, restored at boot, and a change takes effect on the next request without a restart.

For each request the proxy POSTs the decision context to the URL. Candidates are every registered LLM backend, priced the way the completion will be billed: the live polled price when one is held and the static columns otherwise. Queue depth, in-flight count, capacity, and circuit state come from the scheduler at that moment.

```json
{
  "stage": "hop",
  "tags": {"tenant": "acme"},
  "current": "caiso-np15",
  "candidates": [
    {"name": "caiso-np15", "quality": 0.9, "input_cost_per_mtoken": 0.00039,
     "output_cost_per_mtoken": 0.0039, "queue_depth": 2, "in_flight": 1,
     "capacity": 64, "circuit": "closed"}
  ]
}
```

The service answers with one backend name, or an empty string to decline, which routes the stage by its static mapping:

```json
{"backend": "miso-louisiana"}
```

A mapper is a routing hint and never gates availability. A timeout, an error, or a name the proxy did not offer falls back to the static mapping with a warning, and `orla_stage_mapper_decisions_total{outcome}` counts every decision by outcome so an operator can see a failing mapper service. An explicit `X-Orla-Mapping` variant wins over the mapper, since a variant is a per-request pin for shadow testing.

## Dynamic cost sources

A backend's per-million-token costs are usually the static `input_cost_per_mtoken` and `output_cost_per_mtoken` columns. A provider that caches a prompt prefix reports the repeat read as `prompt_tokens_details.cached_tokens`, a subset of `prompt_tokens`, and orla prices that share at `cache_read_cost_per_mtoken` when the backend declares one. A backend that declares no cache rate prices a cached token at the input rate. A backend whose price changes over time can instead carry a `cost_source` URL. The daemon polls every configured source on one interval and holds the latest price in memory. When the proxy computes `cost_usd` for a completion it uses the live price if one is held and the static columns otherwise. The static columns are never rewritten.

The polling cadence is control-plane state, not startup configuration. Read it with `GET /api/v1/costs/policy` and change it with `PUT /api/v1/costs/policy` or `orlactl costs policy set --refresh-interval 30s`. It defaults to 60s. The poller re-reads it before every round, so a change takes effect within one round without a restart.

The source must answer GET with both fields, priced in USD per million tokens:

```json
{"input_cost_per_mtoken": 0.09, "output_cost_per_mtoken": 0.29}
```

Both fields are required and must be finite and non-negative. A fetch that fails, times out, or returns an invalid body keeps the last known price and logs a warning, so a flapping cost service degrades to a stale price rather than to missing cost records. Clearing `cost_source` returns the backend to its static columns on the next polling round. `cost_source` is only valid for LLM backends, and the API returns 400 otherwise.

Nothing bounds how stale a kept price may become, so a cost source that stays down means completions keep being priced from the last value it served. Three metrics make that visible. `orla_cost_fetch_failures_total{backend}` counts failed reads, `orla_cost_price_age_seconds{backend}` reports how long ago the held price was fetched, and `orla_cost_input_per_mtoken_usd{backend}` with `orla_cost_output_per_mtoken_usd{backend}` report the price itself. Alert on a price age that climbs past a few refresh intervals.

## Per-stage request and response capture

A stage can carry `capture_io`. When it is on the proxy records the request and response content of every call tagged with the stage into the `completion_io` table, keyed by `completion_id` and grouped by `workflow_run`. It is off by default, so no content is stored until an operator opts a stage in with `orlactl stage capture STAGE on`.

Capture is a diagnostic aid, not part of the metadata write path. It answers "what did this one stage see and produce on this workflow run" when attributing an outcome to a stage. The content lives in a separate table with its own access control and retention, and the write is best-effort. A capture that fails to store logs and moves on without touching the metadata write or the response to the client.

The request side is the raw request body. The response side is the full response JSON for a non-streaming call and the concatenated assistant text for a streaming call, accumulated from the delta chunks only when capture is on. Read one workflow run's captured I/O with `GET /api/v1/workflows/{run}/completions`. See [`storage.md`](storage.md).

## Streaming semantics

For `stream: true`:

- Orla opens an upstream stream and proxies chunks. Each chunk is rewritten to include the resolved backend in the `model` field.
- The worker holds its concurrency slot until the upstream stream finishes draining, not just until the first chunk arrives. This is load-bearing for the `max_concurrency` invariant.
- On client disconnect, orla drains the upstream stream silently and the worker releases its slot.

## Error shape

All non-200 responses use the OpenAI error envelope:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error" | "permission_denied" | "rate_limit_exceeded" | "server_error" | "api_error"
  }
}
```

Status code drives the `type` field. Clients that already handle OpenAI errors handle these correctly.

Some requests fail before dispatch and never reach `completion_records`. These are counted in `orla_scheduler_rejections_total{backend, reason}`. The `reason` label is one of:

- `unknown_backend`: the resolved backend is not registered. Returns 502.
- `wrong_kind`: the backend exists but is the wrong kind for the route, an LLM asked to serve a tool or the reverse. Returns 500.
- `circuit_open`: the backend's circuit breaker is open and is failing fast. Returns 503 with `Retry-After`.
- `canceled`: the client canceled while the request was queued. Returns 408.
- `deadline_exceeded`: the request deadline passed while the request was queued. Returns 408.
- `internal_error`: any other acquire failure. Returns 500.

The `chat/completions` route falls back to the client-supplied `model` field when a stage has no backend mapping, so an unknown backend name can be arbitrary client input. To keep the metric from minting unbounded label series, the `unknown_backend` case records a fixed `backend="unregistered"` label. The real name still appears in the error body and logs.

These rejections do not increment `orla_requests_total`, which counts only requests that reach dispatch, so a total error rate must union both counters.

Tool dispatches whose reported cost exceeds a $1,000 sanity ceiling are logged and counted in `orla_tool_cost_anomaly_total{backend}`, and still recorded as-is. Orla has no independent way to verify a tool's self-reported cost, so the counter flags an implausible value for a human to investigate.
