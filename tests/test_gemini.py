"""Gemini adapter tests. Offline — every assertion runs off a captured fixture."""

import httpx
import pytest
import respx

from bodol.providers import gemini
from bodol.providers.base import (
    FinishReason,
    Message,
    Provider,
    TextBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
)
from tests.conftest import load

URL = f"{gemini.BASE_URL}{gemini.ENDPOINT}"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


# ---------------------------------------------------------------- normalize


def test_tool_call_turn() -> None:
    r = gemini.normalize(load("toolcalls/gemini_weather_0"), latency_ms=123.4)

    assert r.finish_reason is FinishReason.TOOL_CALLS
    assert r.wants_tools
    assert r.text is None, "a tool-call turn has no prose"
    assert r.model == "gemini-3.6-flash"
    assert r.provider == "gemini"
    assert r.latency_ms == 123.4

    (call,) = r.tool_calls
    assert call.name == "get_weather"
    assert call.id == "vLClgAAM"
    assert call.args == {"city": "Ulaanbaatar"}, "optional `unit` omitted, unlike OpenAI"


def test_final_turn() -> None:
    r = gemini.normalize(load("toolcalls/gemini_weather_1"))

    assert r.finish_reason is FinishReason.STOP
    assert not r.wants_tools
    assert r.text == "The current weather in Ulaanbaatar is clear with a temperature of 12°C."


def test_usage_on_tool_call_turn() -> None:
    u = gemini.normalize(load("toolcalls/gemini_weather_0")).usage

    assert u.input_tokens == 89
    assert u.output_tokens == 19
    assert u.cached_tokens == 0
    assert u.total_tokens == 108


def test_thought_tokens_are_folded_into_output() -> None:
    """The whole reason _usage() exists.

    Gemini excludes thoughts from total_output_tokens but bills them at the
    output rate. Reading the field directly reports 13 instead of 272.
    """
    raw = load("gemini_response")
    assert raw["usage"]["total_output_tokens"] == 13
    assert raw["usage"]["total_thought_tokens"] == 259

    u = gemini.normalize(raw).usage
    assert u.output_tokens == 272
    assert u.reasoning_tokens == 259
    # And the contract's total now agrees with the vendor's own arithmetic.
    assert u.total_tokens == raw["usage"]["total_tokens"] == 275


def test_unknown_status_does_not_read_as_done() -> None:
    r = gemini.normalize({"status": "something_new", "steps": []})
    assert r.finish_reason is FinishReason.UNKNOWN


def test_raw_is_preserved() -> None:
    raw = load("toolcalls/gemini_weather_0")
    assert gemini.normalize(raw).raw is raw


def test_responses_compare_by_value() -> None:
    raw = load("toolcalls/gemini_weather_0")
    assert gemini.normalize(raw) == gemini.normalize(raw)


# ---------------------------------------------------------------- request


def test_renders_a_full_round_trip_as_steps() -> None:
    steps = gemini._render_messages(
        [
            Message("user", (TextBlock("weather in Ulaanbaatar?"),)),
            Message(
                "assistant",
                (ToolUseBlock("vLClgAAM", "get_weather", {"city": "Ulaanbaatar"}),),
            ),
            Message("user", (ToolResultBlock("vLClgAAM", '{"temperature_c": 12}'),)),
        ]
    )

    assert [s["type"] for s in steps] == ["user_input", "function_call", "function_result"]
    assert steps[1]["arguments"] == {"city": "Ulaanbaatar"}
    # JSON-shaped tool output is sent as an object, matching the captured request.
    assert steps[2]["result"] == {"temperature_c": 12}


def test_non_json_tool_result_falls_back_to_string() -> None:
    steps = gemini._render_messages([Message("user", (ToolResultBlock("c1", "12C, clear"),))])
    assert steps[0]["result"] == "12C, clear"


def test_system_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="system"):
        gemini._render_messages([Message("system", (TextBlock("be terse"),))])


def test_adapter_satisfies_the_protocol() -> None:
    adapter = gemini.GeminiAdapter("gemini-3.6-flash", api_key="k")
    assert isinstance(adapter, Provider)


@respx.mock
async def test_generate_posts_and_normalizes() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=load("toolcalls/gemini_weather_0"))
    )

    adapter = gemini.GeminiAdapter("gemini-3.6-flash", api_key="test-key")
    try:
        r = await adapter.generate(
            [Message("user", (TextBlock("weather in Ulaanbaatar?"),))],
            system="You are terse.",
            tools=[WEATHER_TOOL],
        )
    finally:
        await adapter.aclose()

    assert r.finish_reason is FinishReason.TOOL_CALLS
    assert r.latency_ms > 0

    sent = route.calls.last.request
    assert sent.headers["x-goog-api-key"] == "test-key"

    import json as _json

    body = _json.loads(sent.content)
    assert body["model"] == "gemini-3.6-flash"
    assert body["system_instruction"] == "You are terse."
    assert body["tools"][0]["name"] == "get_weather"
    assert body["input"][0]["type"] == "user_input"
