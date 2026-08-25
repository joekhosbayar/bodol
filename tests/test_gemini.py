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
    ThoughtBlock,
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


def test_thought_signature_is_captured_for_replay() -> None:
    """Half of the fix for the first live 400.

    The signature is the model's own proof that its thought and its
    function_call belong together, and Gemini rejects turn 2 when the pair is
    replayed without it. Leaving it buried in `raw` is what broke the first
    tool-using run, so normalize() has to surface it as conversation state.
    """
    raw = load("toolcalls/gemini_weather_0")
    signed = [s["signature"] for s in raw["steps"] if s.get("type") == "thought"]
    assert signed, "fixture proves nothing about signatures without a thought step"

    assert gemini.normalize(raw).thoughts == tuple(ThoughtBlock(s) for s in signed)


def test_unsigned_thought_is_not_replayed() -> None:
    """An empty signature is not replayable, and sending one back is a 400."""
    r = gemini.normalize({"status": "completed", "steps": [{"type": "thought"}]})
    assert r.thoughts == ()


def test_raw_is_preserved() -> None:
    raw = load("toolcalls/gemini_weather_0")
    assert gemini.normalize(raw).raw is raw


def test_responses_compare_by_value() -> None:
    raw = load("toolcalls/gemini_weather_0")
    assert gemini.normalize(raw) == gemini.normalize(raw)


# ---------------------------------------------------------------- request


def test_renders_a_full_round_trip_as_steps() -> None:
    """Turn 2 of a tool call, in the exact shape Gemini accepts.

    Both rules are asserted here because breaking either one fails the request
    whole, with a bare `400 Invalid input received.` and no field named: the
    signed thought must precede its function_call, and the function_result must
    carry the tool's name next to the call id.
    """
    steps = gemini._render_messages(
        [
            Message("user", (TextBlock("weather in Ulaanbaatar?"),)),
            Message(
                "assistant",
                (
                    ThoughtBlock("EjQKMgERTTIPVtJXOu"),
                    ToolUseBlock("vLClgAAM", "get_weather", {"city": "Ulaanbaatar"}),
                ),
            ),
            Message(
                "user",
                (ToolResultBlock("vLClgAAM", "get_weather", '{"temperature_c": 12}'),),
            ),
        ]
    )

    assert [s["type"] for s in steps] == [
        "user_input",
        "thought",
        "function_call",
        "function_result",
    ]
    assert steps[1] == {"type": "thought", "signature": "EjQKMgERTTIPVtJXOu"}
    assert steps[2]["arguments"] == {"city": "Ulaanbaatar"}
    assert steps[3]["call_id"] == "vLClgAAM"
    assert steps[3]["name"] == "get_weather"
    # JSON-shaped tool output is sent as an object, matching the captured request.
    assert steps[3]["result"] == {"temperature_c": 12}


def test_normalized_thought_survives_a_round_trip() -> None:
    """What a live run actually does: normalize a response, replay its turn.

    Asserting against the fixture's own signature is what makes this a
    regression test — the captured response and the next request have to agree.
    """
    response = gemini.normalize(load("toolcalls/gemini_weather_0"))
    (call,) = response.tool_calls
    steps = gemini._render_messages(
        [
            Message("user", (TextBlock("weather in Ulaanbaatar?"),)),
            Message(
                "assistant",
                (*response.thoughts, ToolUseBlock(call.id, call.name, call.args or {})),
            ),
            Message("user", (ToolResultBlock(call.id, call.name, "12C, clear"),)),
        ]
    )

    thought, function_call = steps[1], steps[2]
    assert thought["signature"] == load("toolcalls/gemini_weather_0")["steps"][0]["signature"]
    assert function_call["id"] == call.id


def test_non_json_tool_result_falls_back_to_string() -> None:
    steps = gemini._render_messages(
        [Message("user", (ToolResultBlock("c1", "get_weather", "12C, clear"),))]
    )
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
