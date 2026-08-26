"""Gemini adapter — the `interactions` API on generativelanguage.googleapis.com.

Normalization is verified against tests/fixtures/toolcalls/gemini_weather_{0,1}.json
and tests/fixtures/gemini_response.json.

Conversation state: this adapter deliberately does NOT use
`previous_interaction_id`. Bodol owns the transcript and resends it, so the
context manager can truncate and compact history it actually holds. That costs
more tokens than server-side continuation, and it is what keeps ContextPolicy
meaningful and `bodol compare` an apples-to-apples measurement.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import httpx

from bodol import config
from bodol.providers import http
from bodol.providers.base import (
    FinishReason,
    Message,
    ModelResponse,
    TextBlock,
    ThoughtBlock,
    ToolCall,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)

PROVIDER = "gemini"
BASE_URL = "https://generativelanguage.googleapis.com"
ENDPOINT = "/v1beta/interactions"

# Gemini reports turn state on a top-level `status`. Only the two values the
# fixtures exercise are mapped; anything else surfaces as UNKNOWN rather than
# being silently read as "done" — no truncation fixture has been captured yet.
_FINISH_REASONS = {
    "requires_action": FinishReason.TOOL_CALLS,
    "completed": FinishReason.STOP,
}


# ---------------------------------------------------------------- response


def _usage(raw: dict[str, Any]) -> Usage:
    """Fold thought tokens back into output.

    Two claims that are easy to conflate, and only one of them is true:

        "thoughts are excluded from the total_output_tokens FIELD"   <- true
        "thoughts are excluded from BILLING"                         <- false

    Thoughts bill at the standard output rate; they are merely *reported* in a
    sibling field rather than inside total_output_tokens. So the field has to be
    added back to get true billed output — which is exactly why this function
    exists, not an argument that Google gives the reasoning away.

    Gemini's own arithmetic is the proof, from tests/fixtures/gemini_response.json:

        total_input_tokens    3
        total_output_tokens  13
        total_thought_tokens 259
        total_tokens         275   <- only balances if output excludes thoughts

    Assigning total_output_tokens directly reports 13 against a call that was
    billed for 272 tokens of output.
    """
    thoughts: int = raw.get("total_thought_tokens", 0)
    return Usage(
        input_tokens=raw.get("total_input_tokens", 0),
        output_tokens=raw.get("total_output_tokens", 0) + thoughts,
        cached_tokens=raw.get("total_cached_tokens", 0),
        # Both write buckets stay zero, and that is a fact about Gemini rather
        # than a field nobody read: implicit caching is the only cache this API
        # exposes, it reports no creation count, and it charges no write
        # premium. A token written to it bills as ordinary input.
        reasoning_tokens=thoughts,
    )


def normalize(raw: dict[str, Any], *, latency_ms: float = 0.0) -> ModelResponse:
    """Pure: a decoded response body in, a ModelResponse out. No HTTP here."""
    texts: list[str] = []
    tool_calls: list[ToolCall] = []
    thoughts: list[ThoughtBlock] = []

    for step in raw.get("steps") or []:
        match step.get("type"):
            case "model_output":
                for block in step.get("content") or []:
                    if block.get("type") == "text":
                        texts.append(block["text"])
            case "function_call":
                tool_calls.append(
                    ToolCall(
                        id=step["id"],
                        name=step["name"],
                        # Already an object here; only OpenAI returns a string.
                        args=step.get("arguments") or {},
                    )
                )
            case "thought":
                # A signature and nothing else — no text to show a user. It is
                # captured because the next request has to send it back: Gemini
                # rejects a replayed function_call whose thought step is
                # missing. A thought with no signature is not replayable, so it
                # is dropped rather than echoed as an empty one.
                if signature := step.get("signature"):
                    thoughts.append(ThoughtBlock(signature=signature))

    return ModelResponse(
        id=raw.get("id", ""),
        model=raw.get("model", ""),
        provider=PROVIDER,
        finish_reason=_FINISH_REASONS.get(raw.get("status", ""), FinishReason.UNKNOWN),
        usage=_usage(raw.get("usage") or {}),
        latency_ms=latency_ms,
        # None when there were no text blocks at all — distinct from a model
        # that returned an empty one.
        text="".join(texts) if texts else None,
        tool_calls=tuple(tool_calls),
        thoughts=tuple(thoughts),
        raw=raw,
    )


# ---------------------------------------------------------------- request


def _result_payload(content: str) -> Any:
    """Gemini accepted an object for `function_result.result` in the captured
    fixture, so send one when the content parses as JSON. The plain-string
    fallback is not covered by a fixture yet."""
    try:
        return json.loads(content)
    except ValueError:
        return content


def _render_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    # Order is preserved as given, and must be stable across calls within a run:
    # tools render first in the prompt, so a reordering invalidates the cache for
    # the whole conversation. That guarantee belongs to the tool registry.
    return [
        {
            "type": "function",
            "name": t.name,
            "description": t.description,
            "parameters": t.parameters,
        }
        for t in tools
    ]


def _render_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Flatten the transcript into Gemini's flat `input` step list.

    Two rules govern a turn that replays a tool call, both of which Gemini
    enforces with a bare `400 Invalid input received.`:

        1. The `thought` step the model emitted before its `function_call` has
           to be sent back, in front of it. The signature is how the backend
           ties the pair together.
        2. Every `function_result` carries `name` as well as `call_id`.

    Break either one and the request fails whole — there is no partial accept
    and no field-level detail in the response.
    """
    steps: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            raise ValueError("system text goes in the `system=` argument, not `messages`")
        for block in msg.content:
            match block:
                case TextBlock():
                    steps.append(
                        {
                            "type": "model_output" if msg.role == "assistant" else "user_input",
                            "content": [{"type": "text", "text": block.text}],
                        }
                    )
                case ToolUseBlock():
                    steps.append(
                        {
                            "type": "function_call",
                            "id": block.id,
                            "name": block.name,
                            "arguments": block.args,
                        }
                    )
                case ToolResultBlock():
                    steps.append(
                        {
                            "type": "function_result",
                            "call_id": block.call_id,
                            "name": block.name,
                            "result": _result_payload(block.content),
                            "is_error": block.is_error,
                        }
                    )
                case ThoughtBlock():
                    steps.append({"type": "thought", "signature": block.signature})
    return steps


class GeminiAdapter:
    """Satisfies the `Provider` protocol structurally — it never names it."""

    name = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        retry: http.RetryPolicy = http.DEFAULT_RETRY,
        cache: bool = True,
    ) -> None:
        self.model = model
        self._retry = retry
        # Accepted and then ignored, which is the honest implementation rather
        # than an omission: implicit caching is the only cache this API exposes
        # and there is no request field that turns it off. `--no-cache` cannot
        # be honoured here, and the CLI says so instead of pretending.
        self._cache = cache
        self._owns_client = client is None
        self._client = client or http.make_client(
            base_url=BASE_URL,
            headers={"x-goog-api-key": api_key or config.api_key(PROVIDER)},
        )

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": _render_messages(messages),
            "generation_config": {"max_output_tokens": max_tokens},
        }
        if system:
            payload["system_instruction"] = system
        if tools:
            payload["tools"] = _render_tools(tools)

        result = await http.post_json(self._client, ENDPOINT, payload, policy=self._retry)
        return normalize(result.body, latency_ms=result.latency_ms)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
