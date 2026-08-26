"""OpenAI adapter tests. Offline — every assertion runs off a captured fixture."""

import json

import httpx
import pytest
import respx

from bodol.providers import openai
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

URL = f"{openai.BASE_URL}{openai.ENDPOINT}"

WEATHER_TOOL = ToolSpec(
    name="get_weather",
    description="Get the current weather for a city.",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


# ---------------------------------------------------------------- normalize


def test_completed_status_does_not_mean_the_turn_is_over() -> None:
    """The bug this adapter exists to prevent.

    Both the response and the output item report "completed" while the model is
    blocked on a tool call. Reading `status` as a finish reason would end the
    agent loop with an empty answer and nothing anomalous in the trace.
    """
    raw = load("toolcalls/openai_weather_0")
    assert raw["status"] == "completed"
    assert raw["output"][0]["status"] == "completed"

    r = openai.normalize(raw)
    assert r.finish_reason is FinishReason.TOOL_CALLS
    assert r.wants_tools


def test_tool_call_turn() -> None:
    r = openai.normalize(load("toolcalls/openai_weather_0"), latency_ms=99.0)

    assert r.text is None, "a tool-call turn has no prose"
    assert r.model == "gpt-5.6-luna"
    assert r.provider == "openai"
    assert r.latency_ms == 99.0

    (call,) = r.tool_calls
    assert call.name == "get_weather"
    assert call.id == "call_izeFJBCjtHNrLOeVWuFqpLbW", "echo call_id, never the fc_… id"
    assert not call.id.startswith("fc_")


def test_arguments_are_parsed_from_a_json_string() -> None:
    raw = load("toolcalls/openai_weather_0")
    assert isinstance(raw["output"][0]["arguments"], str), "OpenAI ships a string"

    (call,) = openai.normalize(raw).tool_calls
    assert call.args == {"city": "Ulaanbaatar", "unit": "c"}
    # `unit` was optional in the request. Strict mode promoted it to required
    # and the model invented a value — Gemini and Anthropic both omitted it.
    assert "unit" in call.args


def test_final_turn() -> None:
    r = openai.normalize(load("toolcalls/openai_weather_1"))

    assert r.finish_reason is FinishReason.STOP
    assert not r.wants_tools
    assert r.text == "Ulaanbaatar is currently **12°C (54°F)** and **clear**."


def test_usage_needs_no_arithmetic() -> None:
    u = openai.normalize(load("toolcalls/openai_weather_0")).usage

    assert u.input_tokens == 66
    assert u.output_tokens == 25
    assert u.cached_tokens == 0
    assert u.reasoning_tokens == 0
    assert u.total_tokens == 91 == load("toolcalls/openai_weather_0")["usage"]["total_tokens"]


def test_history_is_replayed_and_billed() -> None:
    """Turn 2 transmitted only the tool result, but input tokens still grew."""
    first = openai.normalize(load("toolcalls/openai_weather_0")).usage.input_tokens
    second = openai.normalize(load("toolcalls/openai_weather_1")).usage.input_tokens
    assert (first, second) == (66, 115)


def _mixed_tool_call_response() -> dict[str, object]:
    """One decodable call and one truncated one, in a single billed turn."""
    return {
        "id": "resp_1",
        "model": "gpt-5.6-luna",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_ok",
                "name": "get_weather",
                "arguments": '{"city":"Ulaanbaatar"}',
            },
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_bad",
                "name": "get_weather",
                "arguments": '{"city": "Ulaanbaa',
            },
        ],
        "usage": {"input_tokens": 66, "output_tokens": 25, "total_tokens": 91},
    }


def test_malformed_arguments_are_recoverable() -> None:
    """A broken tool call must not cost us the response we were billed for."""
    r = openai.normalize(_mixed_tool_call_response())

    assert r.finish_reason is FinishReason.TOOL_CALLS
    assert r.usage.input_tokens == 66, "usage survives a malformed call"
    assert r.usage.output_tokens == 25

    good, bad = r.tool_calls
    assert good.args == {"city": "Ulaanbaatar"}
    assert bad.args is None, "None is the do-not-dispatch signal"
    assert bad.raw_args == '{"city": "Ulaanbaa', "kept so the model can be told what it sent"
    assert bad.id == "call_bad"


def test_valid_calls_also_keep_their_raw_string() -> None:
    (call,) = openai.normalize(load("toolcalls/openai_weather_0")).tool_calls
    assert call.raw_args == '{"city":"Ulaanbaatar","unit":"c"}'


def test_a_malformed_call_can_still_be_replayed_as_history() -> None:
    """Echoing the assistant turn is required before sending its error result."""
    r = openai.normalize(_mixed_tool_call_response())
    bad = r.tool_calls[1]

    items = openai._render_messages(
        [
            Message("assistant", (ToolUseBlock(bad.id, bad.name, {}, raw_args=bad.raw_args),)),
            Message(
                "user",
                (ToolResultBlock(bad.id, bad.name, "arguments were not valid JSON", True),),
            ),
        ]
    )
    # The undecodable bytes go back verbatim — json.dumps({}) would rewrite them.
    assert items[0]["arguments"] == '{"city": "Ulaanbaa'
    assert items[1]["call_id"] == "call_bad"


def test_truncation_maps_to_max_tokens() -> None:
    raw = {
        "id": "resp_1",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output": [],
    }
    assert openai.normalize(raw).finish_reason is FinishReason.MAX_TOKENS


def test_unknown_status_does_not_read_as_done() -> None:
    r = openai.normalize({"status": "queued", "output": []})
    assert r.finish_reason is FinishReason.UNKNOWN


def test_responses_compare_by_value() -> None:
    raw = load("toolcalls/openai_weather_0")
    assert openai.normalize(raw) == openai.normalize(raw)


# ---------------------------------------------------------------- request


def test_renders_a_full_round_trip_as_items() -> None:
    items = openai._render_messages(
        [
            Message("user", (TextBlock("weather in Ulaanbaatar?"),)),
            Message(
                "assistant",
                (ToolUseBlock("call_1", "get_weather", {"city": "Ulaanbaatar"}),),
            ),
            Message(
                "user",
                (ToolResultBlock("call_1", "get_weather", '{"temperature_c": 12}'),),
            ),
        ]
    )

    assert items[0] == {"role": "user", "content": "weather in Ulaanbaatar?"}
    assert items[1]["type"] == "function_call"
    assert items[1]["call_id"] == "call_1"
    # Serialized back to a compact string, matching what the model emits.
    assert items[1]["arguments"] == '{"city":"Ulaanbaatar"}'
    assert items[2] == {
        "type": "function_call_output",
        "call_id": "call_1",
        "output": '{"temperature_c": 12}',
    }


def test_signed_reasoning_is_dropped() -> None:
    """Gemini's signatures mean nothing here, and OpenAI's own reasoning items
    are a different shape (an id plus encrypted content), so the block is
    dropped rather than translated."""
    items = openai._render_messages(
        [Message("assistant", (ThoughtBlock("EjQKMgERTTIPVtJXOu"), TextBlock("hi")))]
    )
    assert items == [{"role": "assistant", "content": "hi"}]


def test_tool_call_args_survive_a_parse_render_round_trip() -> None:
    (call,) = openai.normalize(load("toolcalls/openai_weather_0")).tool_calls
    assert call.args is not None  # mypy: dispatching requires this check
    item = openai._render_messages(
        [Message("assistant", (ToolUseBlock(call.id, call.name, call.args),))]
    )[0]
    assert json.loads(item["arguments"]) == call.args


def test_system_messages_are_rejected() -> None:
    with pytest.raises(ValueError, match="system"):
        openai._render_messages([Message("system", (TextBlock("be terse"),))])


def test_adapter_satisfies_the_protocol() -> None:
    assert isinstance(openai.OpenAIAdapter("gpt-5.6-luna", api_key="k"), Provider)


@respx.mock
async def test_generate_posts_and_normalizes() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=load("toolcalls/openai_weather_0"))
    )

    adapter = openai.OpenAIAdapter("gpt-5.6-luna", api_key="test-key")
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
    assert sent.headers["authorization"] == "Bearer test-key"

    body = json.loads(sent.content)
    assert body["model"] == "gpt-5.6-luna"
    assert body["instructions"] == "You are terse."
    assert body["max_output_tokens"] == 512
    assert body["tools"][0]["name"] == "get_weather", "flat shape, no 'function' wrapper"
    assert "function" not in body["tools"][0]


# ---------------------------------------------------------------- caching


def test_the_cache_key_is_derived_from_the_prefix_not_the_run() -> None:
    """A per-run key would group a run only with itself and throw away the reuse
    the cache lifetime exists to provide."""
    key = openai._cache_key("You are terse.", [WEATHER_TOOL])

    assert key == openai._cache_key("You are terse.", [WEATHER_TOOL]), "stable"
    assert key.startswith("bodol:")
    assert "terse" not in key, "a digest — the key is a field the vendor logs"


def test_changing_the_prefix_changes_the_cache_key() -> None:
    """Tools render ahead of everything, so a different tool set is a different
    prefix and must not be routed to the same entry."""
    other = ToolSpec(name="get_time", description="Time.", parameters={"type": "object"})

    assert openai._cache_key("You are terse.", [WEATHER_TOOL]) != openai._cache_key(
        "You are terse.", [other]
    )
    assert openai._cache_key("You are terse.", []) != openai._cache_key("Be verbose.", [])


@respx.mock
async def test_caching_asks_for_implicit_mode_and_a_routing_key() -> None:
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=load("toolcalls/openai_weather_0"))
    )
    adapter = openai.OpenAIAdapter("gpt-5.6-luna", api_key="k")
    try:
        await adapter.generate(
            [Message("user", (TextBlock("weather?"),))],
            system="You are terse.",
            tools=[WEATHER_TOOL],
        )
    finally:
        await adapter.aclose()

    body = json.loads(route.calls.last.request.content)
    assert body["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}
    assert body["prompt_cache_key"] == openai._cache_key("You are terse.", [WEATHER_TOOL])


@respx.mock
async def test_no_cache_uses_explicit_mode_with_no_breakpoints() -> None:
    """The nearest thing to an off switch: explicit mode with nothing marked
    reads from no cache and writes to none."""
    route = respx.post(URL).mock(
        return_value=httpx.Response(200, json=load("toolcalls/openai_weather_0"))
    )
    adapter = openai.OpenAIAdapter("gpt-5.6-luna", api_key="k", cache=False)
    try:
        await adapter.generate([Message("user", (TextBlock("weather?"),))], system="terse")
    finally:
        await adapter.aclose()

    body = json.loads(route.calls.last.request.content)
    assert body["prompt_cache_options"] == {"mode": "explicit"}
    assert "prompt_cache_key" not in body
    assert "prompt_cache_breakpoint" not in json.dumps(body["input"])


def test_a_cache_write_is_reported_and_no_1h_tier_exists() -> None:
    """Captured live: a 1,881-token prefix over the 1,024 floor, first sighting.

    `cache_write_tokens` sat in every earlier fixture at zero and went unread,
    so this call used to price as plain input at 1.0x instead of 1.25x.
    """
    u = openai.normalize(load("toolcalls/openai_cache_write")).usage

    assert (u.cached_tokens, u.cache_write_tokens) == (0, 1881), "nothing to read yet"
    assert u.cache_write_1h_tokens == 0, "one write rate, one TTL"
    assert u.input_tokens == 1884, "the write is already inside the input total"


def test_a_cache_read_bills_three_ways_at_once() -> None:
    """The same prefix on the next call: read back, with the new tail written."""
    u = openai.normalize(load("toolcalls/openai_cache_read")).usage

    assert (u.cached_tokens, u.cache_write_tokens) == (1881, 70)
    buckets = u.input_buckets
    assert (buckets.uncached, buckets.cached, buckets.write) == (3, 1881, 70)
    assert buckets.total == u.input_tokens == 1954, "the split accounts for every token"
