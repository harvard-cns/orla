"""A lead-and-subagents research team that runs its whole loop through Orla.

Two stages, each tagged on every call so Orla routes and prices them apart:

    research-lead  ->  research-subagent (many, in parallel)

The lead delegates constraints to subagents through the `task` tool and
synthesizes what they report. Each subagent searches the corpus on its own.
Delegation and the per-subagent budgets come from
[subagents-pydantic-ai](https://pypi.org/project/subagents-pydantic-ai/).

The knobs bound what one job may spend, and the toolset applies them when the
lead spawns a subagent. Which model serves a stage is Orla's decision, so
`orlactl stage map` moves a stage onto another backend with no code change.

Environment: ORLA_BASE_URL, MAX_SUBAGENTS, MAX_TOOL_CALLS, MAX_OUTPUT_TOKENS,
LEAD_REQUEST_LIMIT, SUBAGENT_MODE, TOP_K, PAGE_CHARS.
"""

from __future__ import annotations

import os
import threading
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Literal, cast

from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import ToolsetTool
from pydantic_ai.usage import UsageLimits
from subagents_pydantic_ai import SubAgentConfig, SubAgentToolset
from subagents_pydantic_ai.types import UsageLimitsFactory

from corpus import Corpus, Hit

BASE_URL = os.environ.get("ORLA_BASE_URL", "http://localhost:8081/v1")

MAX_SUBAGENTS = int(os.environ.get("MAX_SUBAGENTS", "4"))
MAX_TOOL_CALLS = int(os.environ.get("MAX_TOOL_CALLS", "12"))
MAX_OUTPUT_TOKENS = int(os.environ.get("MAX_OUTPUT_TOKENS", "8192"))
LEAD_REQUEST_LIMIT = int(os.environ.get("LEAD_REQUEST_LIMIT", "20"))


def _subagent_mode() -> Literal["sync", "async", "auto"]:
    mode = os.environ.get("SUBAGENT_MODE", "async")
    if mode not in ("sync", "async", "auto"):
        raise SystemExit(f'SUBAGENT_MODE must be sync, async, or auto, got "{mode}"')
    return cast(Literal["sync", "async", "auto"], mode)


SUBAGENT_MODE = _subagent_mode()
TOP_K = int(os.environ.get("TOP_K", "5"))
PAGE_CHARS = int(os.environ.get("PAGE_CHARS", "4000"))

LEAD_STAGE = "research-lead"
SUBAGENT_STAGE = "research-subagent"
STAGES = [LEAD_STAGE, SUBAGENT_STAGE]

LEAD_INSTRUCTIONS = (
    "You lead a research team answering a hard question about one entity. The "
    "question describes that entity through several independent constraints. "
    "Delegate constraints to researcher subagents with the task tool, one "
    "constraint or one pair of related constraints per subagent. Read what they "
    "report, delegate again when a lead needs following up, and answer once the "
    "entity is confirmed against the constraints. Give the answer as the shortest "
    "phrase that names the entity, with a confidence from 0 to 100."
)

SUBAGENT_INSTRUCTIONS = (
    "You research one part of a larger question against a fixed document corpus. "
    "Call search to see ranked snippets and open to read a document page by page. "
    "Search with the rare, specific words a matching document would contain. "
    "Report what you found, the document ids that support it, and what you ruled out."
)


class Answer(BaseModel):
    answer: str
    confidence: int


LEAD_LOOP = "lead"

# The ReAct loop a model call belongs to. Each delegation runs its subagent in
# its own context, so the calls a subagent makes carry that delegation's name.
_active_loop: ContextVar[str] = ContextVar("active_loop", default=LEAD_LOOP)


@dataclass(frozen=True)
class Call:
    """One model call: the stage that made it, what Orla called it, and what it
    cost in tokens. Orla prices the same call from the backend's energy profile.
    A call is one turn of the loop it names."""

    stage: str
    completion_id: str
    prompt_tokens: int
    completion_tokens: int
    loop: str = LEAD_LOOP


@dataclass(frozen=True)
class Loop:
    """One agent's ReAct loop and how long it ran. Turns are the calls naming
    this loop, so a loop that took n turns called tools on n-1 of them."""

    name: str
    stage: str
    seconds: float


@dataclass
class Research:
    """Everything one question produced, scored against the benchmark's gold
    documents and priced against the energy table. Tool functions run off the
    event loop in a worker thread and several subagents write here at once, so
    the lock guards every field."""

    answer: str = ""
    confidence: int = 0
    subagents: int = 0
    seconds: float = 0.0
    calls: list[Call] = field(default_factory=list)
    loops: list[Loop] = field(default_factory=list)
    searches: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_call(self, call: Call) -> None:
        with self._lock:
            self.calls.append(call)

    def add_loop(self, loop: Loop) -> None:
        with self._lock:
            self.loops.append(loop)

    def add_search(self, query: str, hits: list[Hit]) -> None:
        with self._lock:
            self.searches.append(query)
            self.retrieved.extend(h.docid for h in hits)

    def add_open(self, docid: str) -> None:
        with self._lock:
            self.opened.append(docid)


@dataclass
class Deps:
    """What every agent in one job shares. The subagent toolset clones this per
    delegation, and the corpus and the record are shared on purpose so one job's
    searches are counted together."""

    corpus: Corpus
    record: Research

    def clone_for_subagent(self, max_depth: int = 0) -> Deps:
        return Deps(corpus=self.corpus, record=self.record)


class RecordingModel(OpenAIChatModel):
    """An Orla-tagged model that records the completion id and token usage of
    every call it makes. Subagent calls happen inside the delegation toolset and
    never reach the lead's message history, so recording at the model is what
    lets one job account for all of its calls."""

    def __init__(self, stage: str, run: str, record: Research, provider: OpenAIProvider) -> None:
        super().__init__(
            "orla",
            provider=provider,
            settings=OpenAIChatModelSettings(
                extra_headers={"X-Orla-Stage": stage, "X-Orla-Workflow-Run": run},
                max_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        self._stage = stage
        self._record = record

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await super().request(messages, model_settings, model_request_parameters)
        usage = response.usage
        self._record.add_call(
            Call(
                stage=self._stage,
                completion_id=response.provider_response_id or "",
                prompt_tokens=usage.input_tokens or 0,
                completion_tokens=usage.output_tokens or 0,
                loop=_active_loop.get(),
            )
        )
        return response


def _subagent_limits(ctx: RunContext[Deps], config: SubAgentConfig) -> UsageLimits:
    """The budget each subagent runs under, decided when the lead spawns it."""
    return UsageLimits(
        tool_calls_limit=MAX_TOOL_CALLS,
        output_tokens_limit=MAX_OUTPUT_TOKENS,
    )


class BudgetedSubAgents(SubAgentToolset):
    """Bounds how many subagents one job spawns and keeps a spent budget from
    ending the job. The library's `max_agents` bounds `create_agent`
    registrations rather than `task` delegations, and a subagent's
    `UsageLimitExceeded` travels up through the lead. Reaching either budget
    here ends that subagent and lets the lead answer from what it has."""

    def __init__(
        self,
        *,
        max_spawns: int,
        subagents: list[SubAgentConfig],
        default_model: Model,
        usage_limits: UsageLimitsFactory,
    ) -> None:
        super().__init__(
            subagents=subagents,
            default_model=default_model,
            usage_limits=usage_limits,
            include_general_purpose=False,
        )
        self._max_spawns = max_spawns
        self.spawned = 0

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, object],
        ctx: RunContext[Deps],
        tool: ToolsetTool[Deps],
    ) -> object:
        token: Token[str] | None = None
        loop, started = "", 0.0
        if name == "task":
            # Subagents share one event loop and this check reaches no await,
            # so the count cannot race.
            if self.spawned >= self._max_spawns:
                return f"error: this job may spawn at most {self._max_spawns} subagents"
            self.spawned += 1
            loop = f"researcher-{self.spawned}"
            token = _active_loop.set(loop)
            started = time.monotonic()
        try:
            return await super().call_tool(name, tool_args, ctx, tool)
        except UsageLimitExceeded as e:
            return f"the subagent stopped on its budget and reported nothing further: {e}"
        finally:
            if token is not None:
                _active_loop.reset(token)
                elapsed = time.monotonic() - started
                ctx.deps.record.add_loop(Loop(name=loop, stage=SUBAGENT_STAGE, seconds=elapsed))


def _build_subagent(model: RecordingModel) -> Agent[Deps, str]:
    agent: Agent[Deps, str] = Agent(
        model,
        deps_type=Deps,
        instructions=SUBAGENT_INSTRUCTIONS,
    )

    @agent.tool
    def search(ctx: RunContext[Deps], query: str) -> str:
        """Search the corpus. Returns ranked document ids with a snippet."""
        hits = ctx.deps.corpus.search(query, TOP_K)
        ctx.deps.record.add_search(query, hits)
        if not hits:
            return "no results"
        return "\n".join(f"[{h.docid}] {h.url}\n{h.snippet}" for h in hits)

    @agent.tool
    def open(ctx: RunContext[Deps], docid: str, page: int = 1) -> str:
        """Read one page of a document by id. Pages are 1-indexed."""
        try:
            text, pages = ctx.deps.corpus.open(docid, page, PAGE_CHARS)
        except KeyError:
            return f"error: no document {docid}"
        ctx.deps.record.add_open(docid)
        return f"[{docid}] page {page} of {pages}\n{text}"

    return agent


class ResearchTeam:
    """Runs one question through a lead and its subagents. Holds no per-question
    state, so one instance is safe to share across concurrent jobs."""

    def __init__(self, corpus: Corpus, base_url: str = BASE_URL) -> None:
        self._corpus = corpus
        self._provider = OpenAIProvider(base_url=base_url, api_key="orla")

    async def research(self, query: str, run: str) -> Research:
        record = Research()
        deps = Deps(corpus=self._corpus, record=record)

        subagent_model = RecordingModel(SUBAGENT_STAGE, run, record, self._provider)
        subagent = _build_subagent(subagent_model)
        toolset = BudgetedSubAgents(
            max_spawns=MAX_SUBAGENTS,
            subagents=[
                SubAgentConfig(
                    name="researcher",
                    description="Researches one constraint of the question against the corpus.",
                    instructions=SUBAGENT_INSTRUCTIONS,
                    agent=subagent,
                    preferred_mode=SUBAGENT_MODE,
                )
            ],
            default_model=subagent_model,
            usage_limits=_subagent_limits,
        )

        lead = Agent[Deps, Answer](
            RecordingModel(LEAD_STAGE, run, record, self._provider),
            deps_type=Deps,
            toolsets=[toolset],
            instructions=LEAD_INSTRUCTIONS,
            output_type=Answer,
        )

        token = _active_loop.set(LEAD_LOOP)
        started = time.monotonic()
        try:
            result = await lead.run(
                query,
                deps=deps,
                usage_limits=UsageLimits(request_limit=LEAD_REQUEST_LIMIT),
            )
        finally:
            _active_loop.reset(token)
            record.add_loop(
                Loop(name=LEAD_LOOP, stage=LEAD_STAGE, seconds=time.monotonic() - started)
            )
        record.answer = result.output.answer.strip()
        record.confidence = result.output.confidence
        record.subagents = toolset.spawned
        return record
