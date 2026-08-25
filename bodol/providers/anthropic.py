"""Anthropic adapter — the Messages API at api.anthropic.com/v1/messages.

Normalization is verified against tests/fixtures/toolcalls/claude_weather_{0,1}.json.

The stateless one. There is no `previous_response_id` equivalent, so the full
transcript ships on every call — which is why input tokens grew 591 -> 676
across the captured pair while only a tool result was transmitted. Prompt
caching, not server-side state, is what makes that affordable.

Its message model is the closest of the three to Bodol's own: a list of turns,
each holding content blocks. `_render_messages` is nearly an identity mapping.
The one structural oddity is that tool results ride in a `user` message — there
is no `tool` role.

KNOWN GAP: models that return `thinking` blocks (Opus 5 and friends, where
adaptive thinking is on by default) require those blocks echoed back verbatim,
signature included. `ThoughtBlock` is not that block — it carries a signature
and nothing else, while a `thinking` block carries the thinking text the
signature signs, and normalize() here captures neither. So such a model still
fails on the second step of a tool-using conversation. Gemini failed the same
way until the signed step was replayed; the fix here is the same shape, plus a
text field. claude-haiku-4-5 does not emit them, so it is not needed yet.
"""

from __future__ import annotations

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

PROVIDER = "anthropic"
BASE_URL = "https://api.anthropic.com"
ENDPOINT = "/v1/messages"
API_VERSION = "2023-06-01"

# Anthropic states the turn outcome outright — no derivation needed, unlike
# OpenAI. Values deliberately left unmapped so they surface as UNKNOWN rather
# than being read as success: "refusal" (the model declined) and "pause_turn"
# (a server-side tool paused mid-turn).
_FINISH_REASONS = {
    "tool_use": FinishReason.TOOL_CALLS,
    "end_turn": FinishReason.STOP,
    "stop_sequence": FinishReason.STOP,
    "max_tokens": FinishReason.MAX_TOKENS,
}


# ---------------------------------------------------------------- response


def _usage(raw: dict[str, Any]) -> Usage:
    cache_read: int = raw.get("cache_read_input_tokens", 0)
    return Usage(
        # input_tokens EXCLUDES both cache buckets upstream — the opposite of
        # OpenAI, where cached_tokens is already a subset. See base.py.
        input_tokens=(
            raw.get("input_tokens", 0) + cache_read + raw.get("cache_creation_input_tokens", 0)
        ),
        # Thinking is already inside output_tokens.
        output_tokens=raw.get("output_tokens", 0),
        cached_tokens=cache_read,
        # Present on Opus 5, absent entirely on Haiku 4.5 — the usage schema
        # varies by model within this one provider, so never index.
        reasoning_tokens=(raw.get("output_tokens_details") or {}).get("thinking_tokens", 0),
    )


def normalize(raw: dict[str, Any], *, latency_ms: float = 0.0) -> ModelResponse:
    """Pure: a decoded response body in, a ModelResponse out. No HTTP here."""
    texts: list[str] = []
    tool_calls: list[ToolCall] = []

    for block in raw.get("content") or []:
        match block.get("type"):
            case "text":
                texts.append(block["text"])
            case "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block["id"],
                        name=block["name"],
                        # An object on the wire, so it never fails to decode.
                        # `raw_args` stays None; only OpenAI ships a string.
                        args=block.get("input") or {},
                    )
                )
            # Unhandled block types (thinking, and the undocumented "caller"
            # field on tool_use) are left alone. They survive in `raw`.

    return ModelResponse(
        id=raw.get("id", ""),
        model=raw.get("model", ""),
        provider=PROVIDER,
        finish_reason=_FINISH_REASONS.get(raw.get("stop_reason") or "", FinishReason.UNKNOWN),
        usage=_usage(raw.get("usage") or {}),
        latency_ms=latency_ms,
        text="".join(texts) if texts else None,
        tool_calls=tuple(tool_calls),
        raw=raw,
    )


# ---------------------------------------------------------------- request


def _render_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    # `input_schema`, not `parameters`, and no "type": "function" wrapper.
    # Strict mode is not applied here, so optional parameters stay optional —
    # claude_weather_0 omitted `unit` where OpenAI fabricated a value for it.
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.parameters,
        }
        for t in tools
    ]


def _render_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            raise ValueError("system text goes in the `system=` argument, not `messages`")

        content: list[dict[str, Any]] = []
        for block in msg.content:
            match block:
                case TextBlock():
                    content.append({"type": "text", "text": block.text})
                case ToolUseBlock():
                    content.append(
                        {
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.args,
                        }
                    )
                case ToolResultBlock():
                    # Belongs in a user turn — the caller is responsible for
                    # that, and Anthropic rejects it anywhere else.
                    content.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.call_id,
                            "content": block.content,
                            "is_error": block.is_error,
                        }
                    )
                case ThoughtBlock():
                    # Dropped: a bare signature is not a valid `thinking` block
                    # here, and sending one would fail the request rather than
                    # preserve anything. See the KNOWN GAP above.
                    continue
        rendered.append({"role": msg.role, "content": content})
    return rendered


class AnthropicAdapter:
    """Satisfies the `Provider` protocol structurally — it never names it."""

    name = PROVIDER

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        retry: http.RetryPolicy = http.DEFAULT_RETRY,
    ) -> None:
        self.model = model
        self._retry = retry
        self._owns_client = client is None
        self._client = client or http.make_client(
            base_url=BASE_URL,
            headers={
                "x-api-key": api_key or config.api_key(PROVIDER),
                "anthropic-version": API_VERSION,
            },
        )

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse:
        # max_tokens is required here, unlike the other two.
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": _render_messages(messages),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _render_tools(tools)

        result = await http.post_json(self._client, ENDPOINT, payload, policy=self._retry)
        return normalize(result.body, latency_ms=result.latency_ms)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
