# Long-running deep research on Orla, measured in joules and seconds

A lead-and-subagents research team that answers
[BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) queries against a
fixed corpus, with every model call priced by how much electricity the serving
hardware draws per token. The run reports joules and seconds per answered
question.

The harness exists to profile what a research job may spend. A lead that spawns
six subagents instead of two finds more and costs more, and the question worth
asking is what the extra correctness cost in energy and in wall clock. Orla sees
every call from every subagent, so Orla is where the accounting happens.

## Why BrowseComp-Plus

BrowseComp-Plus takes the 830 queries of OpenAI's BrowseComp and replaces live
web search with a fixed corpus of 100,195 documents. Each query carries
human-verified gold documents, a mean of 2.9 of them, alongside mined hard
negatives. A query retrieves the same documents on every run.

Reproducible retrieval is what makes both measurements mean anything. Live web
search returns different pages of different lengths on every run, so token
counts move for reasons unrelated to the knob under test. Live fetch latency
varies by seconds, which would swamp the timing entirely. Local BM25 answers in
one to two milliseconds deterministically, so a difference in joules or seconds
comes from the models and the orchestration.

The queries are hard by design. Each one describes a single entity through
several independent constraints, which is what gives a lead something to
delegate. The benchmark's own headline is a knob result: GPT-5 answers 55.9%
with BM25 retrieval and 70.1% with a Qwen3 embedding retriever, using fewer
search calls to do it.

## The team

Two stages, each tagged on every call so Orla routes and prices them apart.

    research-lead  ->  research-subagent (many, in parallel)

The **lead** reads the question, delegates constraints to researcher subagents
through a `task` tool, reads what they report, delegates again when a lead needs
following up, and answers once the entity is confirmed.

Each **subagent** researches its own constraint. It calls `search` for ranked
BM25 snippets and `open` to read a document page by page, then reports what it
found, which document ids support it, and what it ruled out.

Delegation comes from
[subagents-pydantic-ai](https://pypi.org/project/subagents-pydantic-ai/), the
focused package underneath
[pydantic-deep](https://github.com/vstorm-co/pydantic-deepagents). It depends on
`pydantic-ai-slim`, `pydantic`, and `typing-extensions` alone, so the example
gets delegation, per-subagent budgets, and sync or async execution without the
sandbox, browser, and console that the full harness carries.

Retrieval is BM25 over the corpus through [bm25s](https://github.com/xhluca/bm25s),
running in process. The index builds once in about a minute and is saved under
`index/`.

## The knobs

Each knob bounds what one job may spend, and every one of them is applied when
the lead spawns a subagent rather than when the team is built.

| Knob | Default | What it bounds |
|---|---|---|
| `MAX_SUBAGENTS` | 4 | Subagents one job may spawn. |
| `MAX_TOOL_CALLS` | 12 | Tool calls one subagent may make. |
| `MAX_OUTPUT_TOKENS` | 8192 | Output tokens one subagent may generate. |
| `SUBAGENT_MODE` | `async` | `sync` runs subagents one at a time, `async` runs them together. |
| `LEAD_REQUEST_LIMIT` | 20 | Model calls the lead may make. |
| `TOP_K` | 5 | Documents returned per search. |
| `PAGE_CHARS` | 4000 | Characters returned per `open` call. |

`SUBAGENT_MODE` is the time knob. Running subagents together shortens the job
and leaves the energy roughly where it was, so it moves one goal without moving
the other.

Reaching a budget ends that subagent and hands the lead a message saying so. The
lead then answers from what it already has. A job that dies on an exhausted
budget produces no measurement, which is the opposite of what a profiling run
needs.

Which model serves a stage is Orla's decision rather than a knob here.

```bash
orlactl stage map research-subagent <other-backend>
```

The team runs on the new backend with no code change and no restart. Registering
several serving profiles as several backends and mapping the two stages across
them is how a model sweep runs without editing anything.

## Energy and time accounting

A serving profile is one way of serving one model: the model, the GPU and how
many of them, the inference engine, the quantization, the batch size, and the
context length. Energy per token varies by five to ten times across those
choices for a single model, so a joules per token figure belongs to a profile
rather than to a model name.

`profiles.json` is the table. Six profiles, each carrying its serving parameters
and its energy draw at the GPU:

```json
"llama-8b-a100-vllm": {
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "gpu": "NVIDIA A100-SXM4-80GB",
  "gpu_count": 1,
  "engine": "vllm",
  "quantization": "bf16",
  "batch_size": 32,
  "context_tokens": 8192,
  "joules_per_prompt_token": 0.027,
  "joules_per_completion_token": 0.16,
  "measured": false
}
```

Every row carries `"measured": false`, because every number in the file is a
placeholder chosen for plausible magnitude. Replacing the file with measured
joules per token is the whole integration. Nothing else in the example reads the
energy table.

`energy_service.py` serves that table two ways. Orla polls `/price/{profile}`
for the cost-source contract, dollars per million tokens, computed from the
profile's joules per token scaled by PUE and priced at `usd_per_kwh`. The
harness reads `/energy/{profile}` for the joules figures themselves.

```bash
curl http://127.0.0.1:9300/price/llama-8b-a100-vllm
{"input_cost_per_mtoken":0.00108,"output_cost_per_mtoken":0.0064}
```

The arithmetic runs 0.027 J per prompt token, times 1.2 for datacenter
overhead, times a million tokens, priced at $0.12 per kWh, giving $0.00108 per
million prompt tokens.

Orla records that cost and a latency on every completion, so the run's
electricity and the time the backends were busy both come back out of
`GET /api/v1/stages/{stage}/metrics`. The harness times each job itself, which
is what a user waits. Summed model latency exceeds job time once subagents run
together, and the report prints the ratio.

The harness also computes joules from its own recorded token counts and warns
when that disagrees with Orla by more than one percent, which catches a call
served by a backend on a different profile.

## Run it

You need a running Orla daemon per the
[quickstart](https://orlaserver.github.io/#/v2/quickstart), plus an
OpenAI-compatible model endpoint and its API key exported.

```bash
just energy    # the energy service on port 9300, in its own terminal
just index     # download the corpus and build the BM25 index, about 3 minutes
N=20 just run  # research 20 queries and report
```

`just run` registers the profile as an Orla backend pointing at your endpoint,
maps the two stages to it, and sets the cost polling interval. Nothing needs
setting up by hand. The run waits until Orla holds a fresh price for the backend
before dispatching, because a call made before then records no cost and would
silently understate the total.

The first run downloads 1.76 GB of corpus into the Hugging Face cache and writes
a 444 MB index under `index/`.

Results land in `trace.jsonl`, which doubles as a resume log, so re-running
after an interruption picks up where it stopped. The report covers the queries
one run researched, and a resumed run leaves the earlier records in the trace
and out of its totals, so quality, joules, and seconds describe the same calls.
Each record also carries the ReAct loops the query ran. A loop is one agent
working its own turn cycle, either the lead or one delegated researcher. A
turn is one model call, so a loop of five turns called tools on four of them.
The report groups the loops by role:

```
  turns per query             9.0

  loop                 loops   turns   turns/loop   s/loop
  research-lead            2       6          3.0     41.0
  research-subagent        4      12          3.0     11.5
```

Turns per query counts every model call one query made, across both roles.
In the table, `loops` is how many loops that role ran, `turns` is the model
calls they made between them, and the last two columns average the turns and
the wall time over those loops. The two roles show 2 lead loops and 4
researcher loops here because the run researched 2 queries, each delegating
to 2 researchers.

The lead's loop spans the whole job, since it waits on every researcher it
spawns. The lead's wall time is therefore the job time, and the researchers'
times overlap inside it.

| Variable | Default | Meaning |
|---|---|---|
| `PROFILE` | `hosted-frontier` | Which serving profile prices the run. |
| `UPSTREAM` | `https://api.openai.com/v1` | The model endpoint Orla dispatches to. |
| `UPSTREAM_MODEL` | `openai:gpt-4o-mini` | The backend's `model_id`, provider-prefixed. |
| `UPSTREAM_KEY_ENV` | `OPENAI_API_KEY` | The env var Orla reads the key from at dispatch. |
| `N` | 20 | How many queries to research. |
| `CONCURRENCY` | 4 | Jobs researched at once, each fanning out to its own subagents. |
| `ORLA_API` | `http://localhost:8081` | Orla's API root. |
| `ORLA_BASE_URL` | `http://localhost:8081/v1` | Orla's OpenAI-compatible endpoint. |
| `ENERGY_URL` | `http://127.0.0.1:9300` | Where the energy service listens. |
| `TRACE` | `trace.jsonl` | Where results are appended. |
| `TRACE_PLAINTEXT` | `0` | Set to 1 to record question, gold, and prediction text. |

## Scope and fidelity

Every energy number in `profiles.json` is a placeholder. The run reports
"placeholder energy table" until a profile carries `"measured": true`.

A hosted API's serving stack is unpublished, so pointing a hosted backend at the
`hosted-frontier` profile attributes an assumed energy draw. The profiles for
open models served locally on known hardware are the ones a measurement can
justify.

Orla polls a cost source per backend rather than per request, so one profile
pins one context length bucket. Profiling context length means registering one
backend per bucket.

The example reaches Orla through pydantic-ai under the orchestration clause in
[AGENTS.md](../../AGENTS.md). Delegation, per-subagent budgets, and parallel
execution are the subject being measured here, and writing them by hand would
put the thing being measured into example code. Every call carries
`X-Orla-Stage` and `X-Orla-Workflow-Run`, and `RecordingModel` keeps each
completion id so feedback posts against it.

Two behaviors sit on top of the delegation library. Its `max_agents` bounds
`create_agent` registrations rather than `task` delegations, and it lets a
subagent's exhausted budget travel up and end the whole job.
`BudgetedSubAgents` in `agent.py` counts spawns per job and turns an exhausted
budget into a message the lead can read.

Retrieval runs in process and contributes no energy or latency to the totals.
Routing the search tool through an Orla tool backend with a `rates` map would
price the retrieval CPU alongside the inference.

Answer scoring is normalized exact match, reported alongside a looser
containment check. The official BrowseComp-Plus scorer is an LLM judge, so the
exact-match figure reads as a floor and the containment figure as a ceiling.
Recall against the gold documents is reported two ways, the share the team
retrieved and the share it opened.

The benchmark ships its queries encrypted under a canary string so they stay out
of training corpora. Decryption happens in memory, and `trace.jsonl` records
scores and token counts with no question or answer text unless
`TRACE_PLAINTEXT` is set. `trace.jsonl` is gitignored either way.

The `usd_per_kwh` in `profiles.json` is one flat number. The
[spatial-shifting](../energy-pricing/spatial-shifting/README.md) experiment
serves real grid prices that move by region and by the minute, so pointing this
harness at that price series would compose the two effects.
