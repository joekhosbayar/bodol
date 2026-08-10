"""Normalized provider types and the adapter interface.

This module is the port; `gemini.py` / `openai.py` / `anthropic.py` are the
adapters. Nothing here imports from them — the arrow points one way only.

Token accounting contract
-------------------------
The three vendors disagree about what is nested inside what, so `Usage` fixes
one convention and each adapter does the arithmetic to satisfy it:

    input_tokens      ALL billed input, including anything served from cache
    cached_tokens     the subset of input_tokens that was served from cache
    output_tokens     ALL billed output, including reasoning/thinking
    reasoning_tokens  the subset of output_tokens spent on reasoning

Per-adapter math, verified against tests/fixtures/toolcalls/:

    gemini-3.6-flash
        input     = usage.total_input_tokens
        cached    = usage.total_cached_tokens
        output    = usage.total_output_tokens + usage.total_thought_tokens
        reasoning = usage.total_thought_tokens

        Thoughts are EXCLUDED from total_output_tokens. The "Hello World"
        fixture shows 3 + 13 + 259 = 275 = total_tokens. Assigning
        total_output_tokens directly understates that call's output 20x.

    gpt-5.6-luna
        input     = usage.input_tokens          (cached already included)
        cached    = usage.input_tokens_details.cached_tokens
        output    = usage.output_tokens         (reasoning already included)
        reasoning = usage.output_tokens_details.reasoning_tokens

    claude-haiku-4-5
        input     = usage.input_tokens
                  + usage.cache_read_input_tokens
                  + usage.cache_creation_input_tokens   (both EXCLUDED upstream)
        cached    = usage.cache_read_input_tokens
        output    = usage.output_tokens         (thinking already included)
        reasoning = usage.output_tokens_details.thinking_tokens, when present

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
class Usage:
    """Token counts under the containment contract in the module docstring."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """Computed, never read from the vendor.

        Anthropic supplies no total at all, and the vendors that do supply one
        don't agree on what it includes.
        """
        return self.input_tokens + self.output_tokens


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
    """The runtime's answer to a ToolUseBlock. `call_id` must match its `id`."""

    call_id: str
    content: str
    is_error: bool = False


ContentBlock = TextBlock | ToolUseBlock | ToolResultBlock


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
