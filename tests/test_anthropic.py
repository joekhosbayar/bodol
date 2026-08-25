"""Anthropic adapter tests. Offline — every assertion runs off a captured fixture."""

import json

import httpx
import pytest
import respx

from bodol.providers import anthropic
from bodol.providers.base import (
    FinishReason,
    Message,
    Provider,
    TextBlock,
    ThoughtBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from tests.conftest import load

URL = f"{anthropic.BASE_URL}{anthropic.ENDPOINT}"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


# ---------------------------------------------------------------- normalize


def test_tool_call_turn() -> None:
    r = anthropic.normalize(load("toolcalls/claude_weather_0"), latency_ms=42.0)

    assert r.finish_reason is FinishReason.TOOL_CALLS
    assert r.wants_tools
    assert r.text is None, "a tool-call turn has no prose"
    assert r.model == "claude-haiku-4-5-20251001", "dated form — pricing lookup must normalize"
    assert r.provider == "anthropic"
    assert r.latency_ms == 42.0

    (call,) = r.tool_calls
    assert call.name == "get_weather"
    assert call.id == "toolu_01PF5pZXBYEvTLCYZW7op1xQ"
    assert call.args == {"city": "Ulaanbaatar"}, "optional `unit` omitted, unlike OpenAI"
    assert call.raw_args is None, "arguments arrive as an object, never a string"


def test_final_turn() -> None:
    r = anthropic.normalize(load("toolcalls/claude_weather_1"))

    assert r.finish_reason is FinishReason.STOP
    assert not r.wants_tools
    assert r.text is not None
    assert r.text.startswith("The weather in Ulaanbaatar right now is:")


def test_usage_adds_the_cache_buckets_back_in() -> None:
    """input_tokens excludes both cache buckets upstream — the opposite of OpenAI."""
    raw = load("toolcalls/claude_weather_0")
    raw["usage"] |= {"cache_read_input_tokens": 100, "cache_creation_input_tokens": 20}

    u = anthropic.normalize(raw).usage
    assert u.input_tokens == 591 + 100 + 20
    assert u.cached_tokens == 100
    assert u.output_tokens == 59


def test_usage_as_captured() -> None:
    u = anthropic.normalize(load("toolcalls/claude_weather_0")).usage
    assert (u.input_tokens, u.output_tokens, u.cached_tokens) == (591, 59, 0)
    # No vendor total is supplied at all; ours is computed.
    assert "total_tokens" not in load("toolcalls/claude_weather_0")["usage"]
    assert u.total_tokens == 650


def test_missing_output_tokens_details_is_not_an_error() -> None:
    """Opus 5 carries this key, Haiku 4.5 does not — same provider."""
    assert "output_tokens_details" not in load("toolcalls/claude_weather_0")["usage"]
    assert anthropic.normalize(load("toolcalls/claude_weather_0")).usage.reasoning_tokens == 0


def test_stateless_history_is_resent_and_billed() -> None:
    """Only a tool result was transmitted on turn 2, yet input still grew."""
    first = anthropic.normalize(load("toolcalls/claude_weather_0")).usage.input_tokens
    second = anthropic.normalize(load("toolcalls/claude_weather_1")).usage.input_tokens
    assert (first, second) == (591, 676)


def test_truncation_maps_to_max_tokens() -> None:
    r = anthropic.normalize({"stop_reason": "max_tokens", "content": []})
    assert r.finish_reason is FinishReason.MAX_TOKENS


def test_refusal_does_not_read_as_done() -> None:
    r = anthropic.normalize({"stop_reason": "refusal", "content": []})
    assert r.finish_reason is FinishReason.UNKNOWN


def test_unknown_content_blocks_are_ignored_but_kept_in_raw() -> None:
    raw = load("toolcalls/claude_weather_0")
    assert raw["content"][0]["caller"] == {"type": "direct"}, "undocumented field"
    r = anthropic.normalize(raw)
    assert len(r.tool_calls) == 1
    assert r.raw is raw


def test_responses_compare_by_value() -> None:
    raw = load("toolcalls/claude_weather_0")
    assert anthropic.normalize(raw) == anthropic.normalize(raw)


# ---------------------------------------------------------------- request


def test_renders_a_full_round_trip() -> None:
    rendered = anthropic._render_messages(
        [
            Message("user", (TextBlock("weather in Ulaanbaatar?"),)),
            Message(
                "assistant",
                (ToolUseBlock("toolu_1", "get_weather", {"city": "Ulaanbaatar"}),),
            ),
            Message(
                "user",
                (ToolResultBlock("toolu_1", "get_weather", '{"temperature_c": 12}'),),
            ),
        ]
    )

    assert [m["role"] for m in rendered] == ["user", "assistant", "user"]
    assert rendered[1]["content"][0] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "get_weather",
        "input": {"city": "Ulaanbaatar"},
    }
    # The tool result rides in a *user* turn — there is no "tool" role.
    assert rendered[2]["role"] == "user"
    assert rendered[2]["content"][0]["type"] == "tool_result"
    assert rendered[2]["content"][0]["tool_use_id"] == "toolu_1"


def test_error_tool_results_are_flagged() -> None:
    rendered = anthropic._render_messages(
        [Message("user", (ToolResultBlock("toolu_1", "get_weather", "not valid JSON", True),))]
    )
    assert rendered[0]["content"][0]["is_error"] is True


def test_signed_reasoning_is_dropped() -> None:
    """A bare signature is not an Anthropic `thinking` block, and sending it as
    one would fail the request. See the KNOWN GAP in the adapter's docstring."""
    rendered = anthropic._render_messages(
        [Message("assistant", (ThoughtBlock("EjQKMgERTTIPVtJXOu"), TextBlock("hi")))]
    )
    assert rendered == [{"role": "assistant", "content": [{"type": "text", "text": "hi"}]}]


def test_tools_use_input_schema_not_parameters() -> None:
    (tool,) = anthropic._render_tools([WEATHER_TOOL])
    assert "input_schema" in tool
    assert "parameters" not in tool
    assert "type" not in tool, "custom tools carry no type discriminator"


def test_system_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="system"):
        anthropic._render_messages([Message("system", (TextBlock("be terse"),))])


def test_adapter_satisfies_the_protocol() -> None:
    adapter = anthropic.AnthropicAdapter("claude-haiku-4-5", api_key="k")
    assert isinstance(adapter, Provider)


@respx.mock
async def test_generate_posts_and_normalizes() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=load("toolcalls/claude_weather_0"))
    )

    adapter = anthropic.AnthropicAdapter("claude-haiku-4-5", api_key="test-key")
    try:
        r = await adapter.generate(
            [Message("user", (TextBlock("weather in Ulaanbaatar?"),))],
            system="You are terse.",
            tools=[WEATHER_TOOL],
            max_tokens=512,
        )
    finally:
        await adapter.aclose()

    assert r.finish_reason is FinishReason.TOOL_CALLS

    sent = route.calls.last.request
    assert sent.headers["x-api-key"] == "test-key"
    assert sent.headers["anthropic-version"] == anthropic.API_VERSION

    body = json.loads(sent.content)
    assert body["model"] == "claude-haiku-4-5"
    assert body["system"] == "You are terse."
    assert body["max_tokens"] == 512, "required by this API, unlike the other two"
    assert body["tools"][0]["input_schema"]["type"] == "object"
