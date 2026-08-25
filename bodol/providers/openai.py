"""OpenAI adapter — the Responses API at api.openai.com/v1/responses.

Normalization is verified against tests/fixtures/toolcalls/openai_weather_{0,1}.json.

Two things make this adapter the odd one out:

1. `status` is not a finish reason. It describes the *response object*, not the
   conversation, and reads "completed" on a turn that is blocked waiting for a
   tool result. TOOL_CALLS must therefore be decided by scanning `output[]`,
   and that check has to run before the status check.

2. `arguments` arrives as a JSON **string**, not an object. This is the only
   adapter that parses, so it is the only one with a malformed-JSON failure.

Conversation state: `previous_response_id` is deliberately unused, matching the
Gemini adapter. Bodol owns the transcript and resends it so the context manager
has something to manage.
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

PROVIDER = "openai"
BASE_URL = "https://api.openai.com"
ENDPOINT = "/v1/responses"


# ---------------------------------------------------------------- response


def _usage(raw: dict[str, Any]) -> Usage:
    return Usage(
        # Both details blocks are subsets of their parent, so no arithmetic —
        # the opposite of Gemini and Anthropic. See the contract in base.py.
        input_tokens=raw.get("input_tokens", 0),
        output_tokens=raw.get("output_tokens", 0),
        cached_tokens=(raw.get("input_tokens_details") or {}).get("cached_tokens", 0),
        reasoning_tokens=(raw.get("output_tokens_details") or {}).get("reasoning_tokens", 0),
    )


def _finish_reason(raw: dict[str, Any], has_tool_calls: bool) -> FinishReason:
    # Order is load-bearing: a turn with tool calls reports status "completed".
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    match raw.get("status"):
        case "completed":
            return FinishReason.STOP
        case "incomplete":
            reason = (raw.get("incomplete_details") or {}).get("reason")
            if reason == "max_output_tokens":
                return FinishReason.MAX_TOKENS
            return FinishReason.UNKNOWN
        case _:
            return FinishReason.UNKNOWN


def normalize(raw: dict[str, Any], *, latency_ms: float = 0.0) -> ModelResponse:
    """Pure: a decoded response body in, a ModelResponse out. No HTTP here."""
    texts: list[str] = []
    tool_calls: list[ToolCall] = []

    for item in raw.get("output") or []:
        match item.get("type"):
            case "message":
                for block in item.get("content") or []:
                    if block.get("type") == "output_text":
                        texts.append(block["text"])
            case "function_call":
                arguments = item.get("arguments") or "{}"
                try:
                    args: dict[str, Any] | None = json.loads(arguments)
                except ValueError:
                    # Recoverable, not fatal. The response — and the usage you
                    # were billed for — is kept; the runtime sees args is None,
                    # declines to dispatch, and returns an error tool result so
                    # the model can fix its own output on the next step.
                    args = None
                tool_calls.append(
                    ToolCall(
                        # `call_id` is what gets echoed back, NOT the item's own
                        # `id` (fc_…). Echoing `id` fails opaquely.
                        id=item["call_id"],
                        name=item["name"],
                        args=args,
                        raw_args=arguments,
                    )
                )
            # "reasoning" items carry no content under the default settings.

    return ModelResponse(
        id=raw.get("id", ""),
        model=raw.get("model", ""),
        provider=PROVIDER,
        finish_reason=_finish_reason(raw, bool(tool_calls)),
        usage=_usage(raw.get("usage") or {}),
        latency_ms=latency_ms,
        text="".join(texts) if texts else None,
        tool_calls=tuple(tool_calls),
        raw=raw,
    )


# ---------------------------------------------------------------- request


def _render_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    # Flat shape — `name`/`parameters` at the top level, no nested "function"
    # wrapper (that is Chat Completions, a different endpoint).
    #
    # Optional parameters must stay out of `parameters["required"]`. OpenAI
    # defaults eligible schemas to strict mode, which expands `required` to
    # every property; the model then fabricates values instead of omitting
    # them. openai_weather_0 invented "unit": "c" where the other two vendors
    # omitted it entirely.
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
    items: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            raise ValueError("system text goes in the `system=` argument, not `messages`")
        for block in msg.content:
            match block:
                case TextBlock():
                    items.append({"role": msg.role, "content": block.text})
                case ToolUseBlock():
                    items.append(
                        {
                            "type": "function_call",
                            "call_id": block.id,
                            "name": block.name,
                            # Replay the provider's own bytes when we have them:
                            # it is the only way to echo a call whose arguments
                            # never decoded, and it keeps the prefix cacheable.
                            "arguments": (
                                block.raw_args
                                if block.raw_args is not None
                                else json.dumps(block.args, separators=(",", ":"))
                            ),
                        }
                    )
                case ToolResultBlock():
                    items.append(
                        {
                            "type": "function_call_output",
                            "call_id": block.call_id,
                            # Must be a string here, unlike Gemini's object.
                            "output": block.content,
                        }
                    )
                case ThoughtBlock():
                    # Dropped: nothing here produces one. normalize() does not
                    # capture OpenAI's `reasoning` items, and echoing them would
                    # mean replaying an item id plus its encrypted content, not
                    # a signature — a different shape than this block carries.
                    continue
    return items


class OpenAIAdapter:
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
            headers={"Authorization": f"Bearer {api_key or config.api_key(PROVIDER)}"},
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
            "max_output_tokens": max_tokens,
        }
        if system:
            payload["instructions"] = system
        if tools:
            payload["tools"] = _render_tools(tools)

        result = await http.post_json(self._client, ENDPOINT, payload, policy=self._retry)
        return normalize(result.body, latency_ms=result.latency_ms)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
