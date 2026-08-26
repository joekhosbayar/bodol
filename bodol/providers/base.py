"""Normalized provider types and the adapter interface.

This module is the port; `gemini.py` / `openai.py` / `anthropic.py` are the
adapters. Nothing here imports from them — the arrow points one way only.

Token accounting contract
-------------------------
The three vendors disagree about what is nested inside what, so `Usage` fixes
one convention and each adapter does the arithmetic to satisfy it:

    input_tokens           ALL billed input, cache reads and writes included
    cached_tokens          the subset of input_tokens served from cache
    cache_write_tokens     the subset written to cache at the default TTL
    cache_write_1h_tokens  the subset written to a 1-hour cache
    output_tokens          ALL billed output, including reasoning/thinking
    reasoning_tokens       the subset of output_tokens spent on reasoning

The three cache subsets are disjoint: a token is read from cache, or written to
one of the two tiers, or neither. What is left over is ordinary uncached input.
`Usage.input_buckets` does that split once, clamped, so pricing and progress
lines cannot disagree about it.

Per-adapter math, verified against tests/fixtures/toolcalls/:

    gemini-3.6-flash
        input     = usage.total_input_tokens
        cached    = usage.total_cached_tokens
        writes    = 0, both tiers. Gemini's implicit cache is the only one it
                    exposes on this API and it carries no write premium — a
                    written token bills as ordinary input. Zero here means
                    "not charged", not "nobody looked".
        output    = usage.total_output_tokens + usage.total_thought_tokens
        reasoning = usage.total_thought_tokens

        Thoughts are EXCLUDED from total_output_tokens. The "Hello World"
        fixture shows 3 + 13 + 259 = 275 = total_tokens. Assigning
        total_output_tokens directly understates that call's output 20x.

    gpt-5.6-luna
        input     = usage.input_tokens          (cached and written included)
        cached    = usage.input_tokens_details.cached_tokens
        write     = usage.input_tokens_details.cache_write_tokens
        write_1h  = 0. There is one write rate and one TTL (30m) to choose.
        output    = usage.output_tokens         (reasoning already included)
        reasoning = usage.output_tokens_details.reasoning_tokens

    claude-haiku-4-5
        input     = usage.input_tokens
                  + usage.cache_read_input_tokens
                  + usage.cache_creation_input_tokens   (both EXCLUDED upstream)
        cached    = usage.cache_read_input_tokens
        write     = usage.cache_creation.ephemeral_5m_input_tokens
        write_1h  = usage.cache_creation.ephemeral_1h_input_tokens
                    The flat cache_creation_input_tokens is the sum of the two,
                    so it stands in for the 5m tier when the split is absent.
        output    = usage.output_tokens         (thinking already included)
        reasoning = usage.output_tokens_details.thinking_tokens, when present

Cache minimums matter when reading a zero
-----------------------------------------
A prefix shorter than the vendor's floor is not cached at all, silently and
without an error. `cached: 0` on a small call is the floor, not a bug:

    openai      1,024 tokens on gpt-5.6 and later
    anthropic   4,096 on Haiku 4.5 · 1,024 on Sonnet 5 · 512 on Opus 5
    gemini      4,096 on 3.x Flash · 2,048 on 2.5

`usage` keys vary by model *within* a vendor — Opus 5 carries
`output_tokens_details`, Haiku 4.5 does not. Read with `.get()`, never `[...]`.

Token counts are not comparable across vendors. The same one-tool request cost
66 input tokens on OpenAI, 89 on Gemini, and 591 on Anthropic, which injects an
unbilled-to-you tool-use system prompt. Compare on cost_usd; treat tokens as a
diagnostic.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal, Protocol, runtime_checkable

# ---------------------------------------------------------------- responses


class FinishReason(StrEnum):
    """Why generation stopped. Derived per adapter — no vendor field maps directly.

    gemini      status == "requires_action"                  -> TOOL_CALLS
                status == "completed"                        -> STOP

    openai      any output[] item of type "function_call"     -> TOOL_CALLS
                status == "completed"                        -> STOP
                status == "incomplete", incomplete_details
                  .reason == "max_output_tokens"             -> MAX_TOKENS

    anthropic   stop_reason == "tool_use"                    -> TOOL_CALLS
                stop_reason in ("end_turn", "stop_sequence") -> STOP
                stop_reason == "max_tokens"                  -> MAX_TOKENS

    OpenAI reports status "completed" on a turn that is waiting on a tool call,
    so its TOOL_CALLS case must be checked before its STOP case.
    """

    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    MAX_TOKENS = "max_tokens"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation the model is asking the runtime to perform.

    `id` is the value the vendor expects echoed back with the result — which is
    not always the block's own identifier. On OpenAI it is `call_id` (`call_…`),
    NOT the item's `id` (`fc_…`).

    `args is None` means the provider sent arguments that did not decode. That
    is a recoverable state, not a failure: the response is still returned with
    its usage intact, and the runtime must NOT dispatch the call. Return a
    ToolResultBlock with is_error=True quoting `raw_args` so the model can
    correct itself on the next step.

    Optional rather than a sentinel `{}` on purpose — mypy forces a None check
    before `**call.args`, so "dispatched a broken call with no arguments"
    becomes a type error instead of a silent wrong tool invocation.
    """

    id: str
    name: str
    args: dict[str, Any] | None
    # The provider's literal wire value, when it sent one. OpenAI ships a JSON
    # string; Gemini and Anthropic ship objects and leave this None. Kept even
    # on success so history can be replayed byte-for-byte.
    raw_args: str | None = None


@dataclass(frozen=True, slots=True)
class InputBuckets:
    """One call's input tokens split into the four classes that bill differently.

    Always sums to `input_tokens`, so a caller can price it or render it without
    re-deriving the remainder and getting a different answer.
    """

    uncached: int
    cached: int
    write: int
    write_1h: int

    @property
    def total(self) -> int:
        return self.uncached + self.cached + self.write + self.write_1h


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts under the containment contract in the module docstring."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cache_write_tokens: int = 0
    cache_write_1h_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Computed, never read from the vendor.

        Anthropic supplies no total at all, and the vendors that do supply one
        don't agree on what it includes.
        """
        return self.input_tokens + self.output_tokens

    @property
    def input_buckets(self) -> InputBuckets:
        """Split input into (uncached, cached, 5m write, 1h write).

        Clamped to `input_tokens` rather than trusted, because the subsets come
        from three different vendors' arithmetic and a sum that overshoots its
        parent must not produce a negative fourth bucket. When it does overshoot,
        the pricier tier is kept whole and the cheaper one absorbs the
        truncation, so an inconsistent report costs more here rather than less.
        """
        remaining = self.input_tokens
        # Claimed in descending order of price: 1h writes at 2x, 5m writes at
        # 1.25x, then cache reads at 0.1x, which is what gets truncated.
        write_1h = min(max(self.cache_write_1h_tokens, 0), remaining)
        remaining -= write_1h
        write = min(max(self.cache_write_tokens, 0), remaining)
        remaining -= write
        cached = min(max(self.cached_tokens, 0), remaining)
        remaining -= cached
        return InputBuckets(uncached=remaining, cached=cached, write=write, write_1h=write_1h)


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """One provider response, normalized.

    `model` is the model that actually served the request, not the one asked
    for — cost from this, since aliases resolve. Note Anthropic returns the
    dated form (`claude-haiku-4-5-20251001`) while Gemini and OpenAI return
    undated IDs, so pricing lookup needs normalization.
    """

    id: str
    model: str
    provider: str
    finish_reason: FinishReason
    usage: Usage
    latency_ms: float
    # None when the model produced no prose at all — the normal case on a
    # tool-call turn, and distinct from the model returning empty prose.
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    # Signed reasoning that has to be replayed alongside `tool_calls` on the
    # next request. Empty for vendors that sign nothing, so the runtime can
    # always echo it unconditionally.
    thoughts: tuple[ThoughtBlock, ...] = ()
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


# ---------------------------------------------------------------- requests

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    """A tool call being replayed back to the model as conversation history.

    A call whose arguments never decoded still has to be replayed — the model's
    turn must be echoed before its error result. Pass the original string as
    `raw_args` and `{}` as `args`; adapters that transmit arguments as a string
    prefer `raw_args`, which also keeps the replayed bytes cache-identical.
    """

    id: str
    name: str
    args: dict[str, Any]
    raw_args: str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    """The runtime's answer to a ToolUseBlock. `call_id` must match its `id`.

    `name` is the tool's registered name. Gemini's `function_result` step
    requires it *alongside* `call_id` and rejects the turn without it, so the
    name rides on the block even though OpenAI and Anthropic match results to
    calls by id alone.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ThoughtBlock:
    """An opaque, provider-signed reasoning step, replayed verbatim.

    Gemini answers 400 when a `function_call` is sent back without the
    `thought` step the model emitted with it: the signature is how the backend
    validates that the pair belongs together. The block is deliberately
    contentless — the signature is conversation state, not something to read,
    summarize, or edit. Adapters with nothing to echo drop it.

    Anthropic's `thinking` blocks are NOT representable here: they carry the
    thinking text next to their signature. See the gap noted in anthropic.py.
    """

    signature: str


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock | ThoughtBlock


@dataclass(frozen=True, slots=True)
class Message:
    """A turn of conversation.

    Adapters map this onto whatever the vendor wants: Anthropic puts tool
    results in a `user` message, Gemini uses a `function_result` step, OpenAI a
    `function_call_output` item. None of them has a `tool` role.
    """

    role: Role
    content: tuple[ContentBlock, ...]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """A tool declaration, rendered per vendor.

    `parameters` is plain JSON Schema. The key it lands under differs:
    `parameters` on Gemini and OpenAI, `input_schema` on Anthropic.

    Keep optional parameters OUT of `required`. OpenAI defaults eligible
    schemas to strict mode and expands `required` to every property, at which
    point the model fabricates values rather than omitting them — the
    openai_weather_0 fixture invented `"unit": "c"` where Gemini and Anthropic
    both omitted it.
    """

    name: str
    description: str
    parameters: dict[str, Any]


# ---------------------------------------------------------------- interface


@runtime_checkable
class Provider(Protocol):
    """What every adapter satisfies. Structural — adapters never name this.

    Conversation state is deliberately not in this signature. Gemini and OpenAI
    can continue server-side from an ID; Anthropic cannot. Putting a
    continuation ID here would make Anthropic unimplementable, so `messages`
    always carries the full history and adapters decide internally what to send.

    Each adapter module should also expose a module-level

        normalize(raw: dict[str, Any], *, latency_ms: float) -> ModelResponse

    as a pure function with no HTTP in it. That is what makes the fixtures in
    tests/fixtures/ testable with no API key and no network.
    """

    name: str
    model: str

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse: ...

    async def aclose(self) -> None: ...
