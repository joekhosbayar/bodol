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
            Message("user", (ToolResultBlock(bad.id, "arguments were not valid JSON", True),)),
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
            Message("user", (ToolResultBlock("call_1", '{"temperature_c": 12}'),)),
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
