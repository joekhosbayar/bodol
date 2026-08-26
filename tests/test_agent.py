"""Tests for the step loop: stop rules, tool dispatch, history, compaction."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from bodol.agent import Agent, Limits, RunResult, StopReason
from bodol.agent import loop as loop_module
from bodol.context import ContextPolicy
from bodol.providers import UnsupportedProviderError
from bodol.providers import http as http_module
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
from bodol.providers.http import PermanentError, QuotaError, TransientError
from bodol.telemetry import events
from bodol.tools import ToolRegistry


def _response(
    *,
    text: str | None = None,
    tool_calls: tuple[ToolCall, ...] = (),
    thoughts: tuple[ThoughtBlock, ...] = (),
    finish_reason: FinishReason | None = None,
    input_tokens: int = 100,
    output_tokens: int = 10,
) -> ModelResponse:
    if finish_reason is None:
        finish_reason = FinishReason.TOOL_CALLS if tool_calls else FinishReason.STOP
    return ModelResponse(
        id="resp",
        model="fake-model",
        provider="fake",
        finish_reason=finish_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
        latency_ms=1.0,
        text=text,
        tool_calls=tool_calls,
        thoughts=thoughts,
    )


def _tool_call(call_id: str = "t1", city: str = "Boston") -> ToolCall:
    return ToolCall(id=call_id, name="get_weather", args={"city": city})


class FakeProvider:
    """Replays scripted responses and records what each call received."""

    name = "fake"
    model = "fake-model"

    def __init__(
        self,
        script: Sequence[ModelResponse] | None = None,
        *,
        repeat: ModelResponse | None = None,
        raises: Exception | None = None,
    ) -> None:
        self._script = list(script or ())
        self._repeat = repeat
        self._raises = raises
        self.calls: list[tuple[list[Message], str | None, tuple[ToolSpec, ...], int]] = []
        self.steps: list[int] = []
        self.closed = 0

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse:
        self.calls.append((list(messages), system, tuple(tools), max_tokens))
        self.steps.append(events.current_step())
        if self._raises is not None:
            raise self._raises
        if self._script:
            return self._script.pop(0)
        if self._repeat is not None:
            return self._repeat
        raise AssertionError("FakeProvider ran out of scripted responses")

    async def aclose(self) -> None:
        self.closed += 1


InstallProvider = Callable[[FakeProvider], FakeProvider]


@pytest.fixture
def install(monkeypatch: pytest.MonkeyPatch) -> InstallProvider:
    """Swap the loop's provider factory for a fake, bypassing HTTP and sinks."""

    def _install(provider: FakeProvider) -> FakeProvider:
        monkeypatch.setattr(loop_module, "create_provider", lambda *a, **kw: provider)
        return provider

    return _install


def _weather_registry(handled: list[str] | None = None) -> ToolRegistry:
    registry = ToolRegistry()

    def get_weather(city: str) -> str:
        if handled is not None:
            handled.append(city)
        return f"sunny in {city}"

    registry.register(
        name="get_weather",
        description="Current weather for a city.",
        parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        handler=get_weather,
    )
    return registry


# ---------------------------------------------------------------- happy paths


async def test_single_turn_finishes(install: InstallProvider) -> None:
    provider = install(FakeProvider([_response(text="42")]))

    result = await Agent("fake:fake-model", system="be terse").run("six times seven")

    assert isinstance(result, RunResult)
    assert result.stop_reason is StopReason.DONE
    assert result.text == "42"
    assert result.steps == 1
    assert result.trace_id is not None
    assert [m.role for m in result.messages] == ["user", "assistant"]

    (messages, system, tools, max_tokens) = provider.calls[0]
    assert system == "be terse"
    assert tools == (), "an empty registry declares no tools"
    assert max_tokens == 4096
    assert messages[0].content == (TextBlock(text="six times seven"),)


async def test_tool_round_trip_builds_history(install: InstallProvider) -> None:
    handled: list[str] = []
    provider = install(
        FakeProvider(
            [
                _response(text="looking that up", tool_calls=(_tool_call(),)),
                _response(text="It is sunny."),
            ]
        )
    )
    agent = Agent("fake:fake-model", tools=_weather_registry(handled))

    result = await agent.run("weather in Boston?")

    assert result.stop_reason is StopReason.DONE
    assert result.steps == 2
    assert handled == ["Boston"], "the handler actually ran"

    assert [m.role for m in result.messages] == ["user", "assistant", "user", "assistant"]
    assert result.messages[1].content == (
        TextBlock(text="looking that up"),
        ToolUseBlock(id="t1", name="get_weather", args={"city": "Boston"}),
    )
    assert result.messages[2].content == (
        ToolResultBlock(call_id="t1", name="get_weather", content="sunny in Boston"),
    )

    second_messages = provider.calls[1][0]
    assert len(second_messages) == 3, "the second call sees the whole round trip"
    assert provider.steps == [1, 2], "one step per model call"


async def test_signed_reasoning_is_replayed_ahead_of_the_call(install: InstallProvider) -> None:
    """The loop's half of the Gemini turn-2 fix.

    Gemini validates a replayed function_call against the thought step that
    preceded it, so the signed blocks lead the echoed assistant turn — ahead of
    prose as well as the calls themselves.
    """
    signed = ThoughtBlock("EjQKMgERTTIPVtJXOu")
    provider = install(
        FakeProvider(
            [
                _response(
                    text="looking that up",
                    tool_calls=(_tool_call(),),
                    thoughts=(signed,),
                ),
                _response(text="It is sunny."),
            ]
        )
    )

    result = await Agent("fake:fake-model", tools=_weather_registry()).run("weather in Boston?")

    assert result.messages[1].content == (
        signed,
        TextBlock(text="looking that up"),
        ToolUseBlock(id="t1", name="get_weather", args={"city": "Boston"}),
    )
    # And the next call is handed that order, which is what the vendor checks.
    assert provider.calls[1][0][1].content[0] is signed


async def test_tool_results_follow_the_models_order(install: InstallProvider) -> None:
    install(
        FakeProvider(
            [
                _response(
                    tool_calls=(
                        _tool_call(call_id="a", city="Boston"),
                        _tool_call(call_id="b", city="Lisbon"),
                    )
                ),
                _response(text="done"),
            ]
        )
    )

    result = await Agent("fake:fake-model", tools=_weather_registry()).run("compare cities")

    blocks = [b for b in result.messages[2].content if isinstance(b, ToolResultBlock)]
    assert [b.call_id for b in blocks] == ["a", "b"]
    assert [b.content for b in blocks] == ["sunny in Boston", "sunny in Lisbon"]


# ---------------------------------------------------------------- tool failures


async def test_unknown_tool_is_answered_not_raised(install: InstallProvider) -> None:
    install(
        FakeProvider(
            [
                _response(
                    tool_calls=(
                        _tool_call(call_id="a"),
                        ToolCall(id="b", name="launch_rocket", args={}),
                    )
                ),
                _response(text="sorry about that"),
            ]
        )
    )

    result = await Agent("fake:fake-model", tools=_weather_registry()).run("do the impossible")

    assert result.stop_reason is StopReason.DONE, "the run recovers instead of aborting"
    good, bad = result.messages[2].content
    assert isinstance(good, ToolResultBlock)
    assert not good.is_error
    assert good.content == "sunny in Boston", "the sibling result is not discarded"
    assert isinstance(bad, ToolResultBlock)
    assert bad.is_error
    assert "launch_rocket" in bad.content
    assert "get_weather" in bad.content, "the model is told what it may call instead"


async def test_malformed_arguments_replay_and_error(install: InstallProvider) -> None:
    install(
        FakeProvider(
            [
                _response(
                    tool_calls=(
                        ToolCall(id="t1", name="get_weather", args=None, raw_args='{"city":'),
                    )
                ),
                _response(text="retrying"),
            ]
        )
    )

    result = await Agent("fake:fake-model", tools=_weather_registry()).run("weather?")

    assert result.messages[1].content == (
        ToolUseBlock(id="t1", name="get_weather", args={}, raw_args='{"city":'),
    ), "a call that never decoded is still echoed before its error"
    (block,) = result.messages[2].content
    assert isinstance(block, ToolResultBlock)
    assert block.is_error
    assert '{"city":' in block.content


# ---------------------------------------------------------------- stop rules


async def test_max_steps_stops_the_loop(install: InstallProvider) -> None:
    provider = install(FakeProvider(repeat=_response(tool_calls=(_tool_call(),))))
    agent = Agent("fake:fake-model", tools=_weather_registry(), limits=Limits(max_steps=3))

    result = await agent.run("loop forever")

    assert result.stop_reason is StopReason.MAX_STEPS
    assert result.steps == 3
    assert len(provider.calls) == 3, "no extra call is made to discover the limit"
    assert result.text is None
    assert result.messages, "the partial transcript comes back"


async def test_max_steps_zero_never_calls_the_provider(install: InstallProvider) -> None:
    provider = install(FakeProvider(repeat=_response(text="hi")))

    result = await Agent("fake:fake-model", limits=Limits(max_steps=0)).run("nothing to do")

    assert result.stop_reason is StopReason.MAX_STEPS
    assert provider.calls == []
    assert provider.closed == 1


async def test_max_cost_stops_the_loop(
    install: InstallProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        events, "rates_for", lambda provider, model: events.Rates(input=1000.0, output=1000.0)
    )
    install(FakeProvider(repeat=_response(tool_calls=(_tool_call(),))))
    agent = Agent("fake:fake-model", tools=_weather_registry(), limits=Limits(max_cost_usd=0.2))

    result = await agent.run("spend money")

    assert result.stop_reason is StopReason.MAX_COST
    assert result.steps == 2, "0.11 a call, so the second crosses 0.2"
    assert result.cost_usd == pytest.approx(0.22)
    assert result.unpriced_calls == 0


async def test_unpriced_model_cannot_trip_max_cost(
    install: InstallProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(events, "rates_for", lambda provider, model: events.Rates())
    install(FakeProvider(repeat=_response(tool_calls=(_tool_call(),))))
    agent = Agent(
        "fake:fake-model",
        tools=_weather_registry(),
        limits=Limits(max_steps=2, max_cost_usd=0.0001),
    )

    result = await agent.run("spend unknown money")

    assert result.stop_reason is StopReason.MAX_STEPS, "unknown cost cannot bound the run"
    assert result.cost_usd == 0.0
    assert result.unpriced_calls == 2


async def test_max_seconds_stops_the_loop(
    install: InstallProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every clock read, in order: the start, then per iteration a limit check and
    # (when a call completed) the progress header, then `result` on the way out.
    # The header's reading has to come after the call rather than share the limit
    # check's, or a step's elapsed time would omit that step's own latency.
    ticks = iter((0.0, 0.0, 1.0, 5.0, 6.0, 11.0, 11.0))
    monkeypatch.setattr(loop_module, "_monotonic", lambda: next(ticks, 999.0))
    install(FakeProvider(repeat=_response(tool_calls=(_tool_call(),))))
    agent = Agent(
        "fake:fake-model",
        tools=_weather_registry(),
        limits=Limits(max_steps=99, max_seconds=10.0),
    )

    result = await agent.run("take too long")

    assert result.stop_reason is StopReason.MAX_SECONDS
    assert result.steps == 2


async def test_each_step_is_announced_as_it_lands(
    install: InstallProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """A run that prints nothing until it finishes is indistinguishable from a
    hung one. The tool arrows come before the results because dispatch gathers:
    five calls start together, so five asks and then five answers is what
    actually happened."""
    install(
        FakeProvider(
            [
                _response(text="Let me check.", tool_calls=(_tool_call(),)),
                _response(text="sunny"),
            ]
        )
    )
    agent = Agent("fake:fake-model", tools=_weather_registry())

    with caplog.at_level("INFO", logger="bodol.agent.loop"):
        await agent.run("weather in Boston?")

    lines = [record.getMessage() for record in caplog.records]
    assert lines[0].startswith("step 1 · ")
    assert lines[1] == '  "Let me check."'
    assert lines[2] == "  → get_weather(city='Boston')"
    assert lines[3] == "  ← get_weather · ok · 15 B"
    assert lines[4].startswith("step 2 · ")


async def test_progress_is_info_so_trouble_can_still_be_filtered_out(
    install: InstallProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """Someone who only wants to hear about problems sets WARNING and should then
    get the retry notices and budget overruns, not a play-by-play."""
    install(FakeProvider([_response(text="done")]))

    with caplog.at_level("WARNING", logger="bodol.agent.loop"):
        await Agent("fake:fake-model").run("hello")

    assert caplog.records == []


async def test_the_overshoot_past_max_seconds_is_named(
    install: InstallProvider, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Stopping for time is in the stop reason; how far past the line is not.

    The gap between the budget and the elapsed time is the size of the hole,
    and it is the only way to see that one call ran long past the deadline.
    """
    ticks = iter((0.0, 0.0, 5.0, 31.0, 31.0))
    monkeypatch.setattr(loop_module, "_monotonic", lambda: next(ticks, 999.0))
    install(FakeProvider(repeat=_response(tool_calls=(_tool_call(),))))
    agent = Agent(
        "fake:fake-model",
        tools=_weather_registry(),
        limits=Limits(max_steps=99, max_seconds=10.0),
    )

    with caplog.at_level("WARNING", logger="bodol.agent.loop"):
        await agent.run("take too long")

    message = caplog.records[0].getMessage()
    assert "31.0s" in message, "what it actually took"
    assert "10s" in message, "against what it was allowed"


async def test_a_failed_run_still_reports_a_blown_time_budget(
    install: InstallProvider, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The real case this exists for.

    A retry sequence inside one call pushed a run past its budget and then the
    call failed. The stop reason is ERROR, the limit check never runs again,
    and without a report on the way out the overrun leaves no trace at all.
    """
    ticks = iter((0.0, 0.0, 130.0))
    monkeypatch.setattr(loop_module, "_monotonic", lambda: next(ticks, 999.0))
    install(FakeProvider(raises=RuntimeError("HTTP 429 after 2 attempts and 72.2s")))
    agent = Agent("fake:fake-model", limits=Limits(max_seconds=120.0))

    result = await agent.run("get rate limited")

    assert result.stop_reason is StopReason.ERROR, "the run ended on the error, not the clock"
    assert "130.0s" in caplog.records[0].getMessage()
    assert "120s" in caplog.records[0].getMessage()


async def test_a_run_inside_its_budget_says_nothing(
    install: InstallProvider, caplog: pytest.LogCaptureFixture
) -> None:
    install(FakeProvider([_response(text="fast")]))

    await Agent("fake:fake-model").run("be quick")

    assert caplog.records == [], "a budget kept is not news"


async def test_usage_is_summed_across_calls(install: InstallProvider) -> None:
    install(
        FakeProvider(
            [
                _response(tool_calls=(_tool_call(),), input_tokens=100, output_tokens=10),
                _response(text="done", input_tokens=180, output_tokens=25),
            ]
        )
    )

    result = await Agent("fake:fake-model", tools=_weather_registry()).run("weather?")

    assert result.usage.input_tokens == 280, "billed input, not final transcript size"
    assert result.usage.output_tokens == 35
    assert result.usage.total_tokens == 315


# ---------------------------------------------------------------- finish reasons


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        (FinishReason.STOP, StopReason.DONE),
        (FinishReason.MAX_TOKENS, StopReason.MAX_TOKENS),
        (FinishReason.UNKNOWN, StopReason.UNKNOWN),
    ],
)
async def test_finish_reason_maps_to_stop_reason(
    install: InstallProvider, finish_reason: FinishReason, expected: StopReason
) -> None:
    install(FakeProvider([_response(text="partial", finish_reason=finish_reason)]))

    result = await Agent("fake:fake-model").run("answer")

    assert result.stop_reason is expected
    assert result.text == "partial"


async def test_textless_final_turn_adds_no_message(install: InstallProvider) -> None:
    install(FakeProvider([_response(text=None)]))

    result = await Agent("fake:fake-model").run("say nothing")

    assert result.text is None
    assert [m.role for m in result.messages] == ["user"], "no phantom empty assistant turn"


# ---------------------------------------------------------------- failures


async def test_provider_failure_becomes_a_stop_reason(install: InstallProvider) -> None:
    provider = install(FakeProvider(raises=RuntimeError("upstream exploded")))

    result = await Agent("fake:fake-model").run("try me")

    assert result.stop_reason is StopReason.ERROR
    assert result.error == "RuntimeError: upstream exploded"
    assert result.steps == 0
    assert [m.role for m in result.messages] == ["user"], "partial history survives"
    assert provider.closed == 1, "the sink is closed on the error path too"


async def test_construction_failure_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> object:
        raise UnsupportedProviderError("nope")

    monkeypatch.setattr(loop_module, "create_provider", boom)

    with pytest.raises(UnsupportedProviderError):
        await Agent("nope:nope").run("go")


# ---------------------------------------------------------------- compaction


async def test_compaction_runs_when_over_budget(install: InstallProvider) -> None:
    provider = install(
        FakeProvider(
            [
                _response(tool_calls=(_tool_call(call_id="a"),), input_tokens=10),
                # Over the 50-token budget, so this round triggers compaction.
                _response(tool_calls=(_tool_call(call_id="b"),), input_tokens=900),
                _response(text="a summary of what came before", input_tokens=20),
                _response(text="all done", input_tokens=20),
            ]
        )
    )
    agent = Agent(
        "fake:fake-model",
        tools=_weather_registry(),
        context=ContextPolicy(max_input_tokens=50, min_recent_turns=1),
    )

    result = await agent.run("weather?")

    assert result.stop_reason is StopReason.DONE
    assert len(provider.calls) == 4
    assert result.steps == 3, "summarization rides inside the step that triggered it"
    assert provider.steps == [1, 2, 2, 3]

    final_messages = provider.calls[3][0]
    assert final_messages[0] is result.messages[0], "the task survives compaction"
    summary = final_messages[1].content[0]
    assert isinstance(summary, TextBlock)
    assert "a summary of what came before" in summary.text
    assert len(final_messages) == 4, "task, summary, and the latest tool pair"
    assert result.messages[:4] == tuple(final_messages)


async def test_compaction_failure_stops_the_run(install: InstallProvider) -> None:
    install(FakeProvider(repeat=_response(tool_calls=(_tool_call(),), input_tokens=900)))
    agent = Agent(
        "fake:fake-model",
        tools=_weather_registry(),
        # min_recent_turns=9 protects the whole transcript, so nothing is evictable.
        context=ContextPolicy(max_input_tokens=50, min_recent_turns=9),
    )

    result = await agent.run("weather?")

    assert result.stop_reason is StopReason.COMPACTION_FAILED
    assert result.error is not None
    assert "protected" in result.error
    assert result.steps == 1


async def test_no_policy_means_no_compaction(install: InstallProvider) -> None:
    provider = install(
        FakeProvider(
            [
                _response(tool_calls=(_tool_call(),), input_tokens=10_000),
                _response(text="fine", input_tokens=10_000),
            ]
        )
    )

    result = await Agent("fake:fake-model", tools=_weather_registry()).run("weather?")

    assert result.stop_reason is StopReason.DONE
    assert len(provider.calls) == 2, "no summarization call was made"


# ---------------------------------------------------------------- tracing


async def test_trace_is_scoped_to_the_run(install: InstallProvider) -> None:
    provider = install(
        FakeProvider([_response(tool_calls=(_tool_call(),)), _response(text="done")])
    )

    assert events.current_trace_id() is None
    result = await Agent("fake:fake-model", tools=_weather_registry()).run("weather?")

    assert result.trace_id is not None
    assert events.current_trace_id() is None, "the trace does not leak past the run"
    assert events.current_step() == 0, "the step counter is restored"
    assert provider.steps == [1, 2]
    assert provider.closed == 1


# ------------------------------------------------------------- step retries


class FlakyProvider(FakeProvider):
    """Fails a set number of times before replaying its script."""

    def __init__(self, failures: int, exc: Exception, *, script: Sequence[ModelResponse]) -> None:
        super().__init__(script)
        self._failures = failures
        self._exc = exc

    async def generate(self, *args: object, **kwargs: object) -> ModelResponse:
        if self._failures > 0:
            self._failures -= 1
            self.calls.append(([], None, (), 0))
            self.steps.append(events.current_step())
            raise self._exc
        return await super().generate(*args, **kwargs)  # type: ignore[arg-type]


async def test_a_transient_failure_is_retried_instead_of_ending_the_run(
    install: InstallProvider,
) -> None:
    """A run given 300s died after 137 of them, with 160 seconds it never spent.

    One 500 from the provider ended the whole run. It is busy, not broken.
    """
    provider = install(
        FlakyProvider(
            2,
            TransientError("HTTP 500", status=500, body=None, url="u"),
            script=[_response(text="done")],
        )
    )

    result = await Agent("fake").run("go")

    assert result.stop_reason is StopReason.DONE
    assert result.text == "done"
    assert len(provider.calls) == 3, "two failures and the success"
    assert result.steps == 1, "one turn, however many attempts it took"


async def test_the_step_number_holds_across_retries(install: InstallProvider) -> None:
    """"Step 3 failed twice then succeeded" is readable; three unrelated
    numbers for the same turn is not."""
    provider = install(
        FlakyProvider(
            2,
            TransientError("HTTP 503", status=503, body=None, url="u"),
            script=[_response(text="done")],
        )
    )

    await Agent("fake").run("go")

    assert provider.steps == [1, 1, 1]


async def test_step_retries_are_bounded(install: InstallProvider) -> None:
    provider = install(
        FlakyProvider(
            5,
            TransientError("HTTP 500", status=500, body=None, url="u"),
            script=[_response(text="never reached")],
        )
    )

    result = await Agent("fake", limits=Limits(max_step_retries=1)).run("go")

    assert result.stop_reason is StopReason.ERROR
    assert result.error is not None and "TransientError" in result.error
    assert len(provider.calls) == 2, "the attempt and its one retry"
    assert result.steps == 0, "no turn ever completed"


async def test_no_retry_once_the_run_is_out_of_time(
    install: InstallProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retrying is only worth it while there is budget left to retry into."""
    ticks = iter([0.0, 0.0])

    def clock() -> float:
        return next(ticks, 1000.0)

    monkeypatch.setattr(loop_module, "_monotonic", clock)
    provider = install(
        FlakyProvider(
            5,
            TransientError("HTTP 500", status=500, body=None, url="u"),
            script=[_response(text="never reached")],
        )
    )

    result = await Agent("fake").run("go")

    assert result.stop_reason is StopReason.ERROR
    assert len(provider.calls) == 1, "no retry into a budget that is already gone"


async def test_a_quota_error_ends_the_run_at_once(install: InstallProvider) -> None:
    """The allowance is spent, and every further attempt spends more of it."""
    provider = install(
        FlakyProvider(
            5,
            QuotaError("out of allowance", status=429, body=None, url="u"),
            script=[_response(text="never reached")],
        )
    )

    result = await Agent("fake").run("go")

    assert result.stop_reason is StopReason.ERROR
    assert result.error is not None and "QuotaError" in result.error
    assert len(provider.calls) == 1


async def test_a_permanent_error_ends_the_run_at_once(install: InstallProvider) -> None:
    provider = install(
        FlakyProvider(
            5,
            PermanentError("HTTP 400", status=400, body=None, url="u"),
            script=[_response(text="never reached")],
        )
    )

    result = await Agent("fake").run("go")

    assert result.stop_reason is StopReason.ERROR
    assert len(provider.calls) == 1


async def test_a_retry_is_announced_while_it_happens(
    install: InstallProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """Silence during a retry is indistinguishable from a hung process."""
    install(
        FlakyProvider(
            1,
            TransientError("HTTP 500", status=500, body=None, url="u"),
            script=[_response(text="done")],
        )
    )

    with caplog.at_level("INFO", logger="bodol.agent.loop"):
        await Agent("fake").run("go")

    notices = [r.getMessage() for r in caplog.records if "retry step" in r.getMessage()]
    assert notices == ["retry step 1 · TransientError · 1 of 2 left · 0s/120s"]


async def test_compaction_is_announced_while_it_happens(
    install: InstallProvider, caplog: pytest.LogCaptureFixture
) -> None:
    """A silent compaction is indistinguishable from the model pausing."""
    install(
        FakeProvider(
            [
                _response(tool_calls=(_tool_call(call_id="a"),), input_tokens=10),
                # Over the 50-token budget, so this round triggers compaction.
                _response(tool_calls=(_tool_call(call_id="b"),), input_tokens=900),
                _response(text="a summary of what came before", input_tokens=20),
                _response(text="all done", input_tokens=20),
            ]
        )
    )

    with caplog.at_level("INFO", logger="bodol.agent.loop"):
        await Agent(
            "fake:fake-model",
            tools=_weather_registry(),
            context=ContextPolicy(max_input_tokens=50, min_recent_turns=1),
        ).run("weather?")

    notices = [r.getMessage() for r in caplog.records if "compact step" in r.getMessage()]
    assert notices == ["compact step 2 · 900 in over 50 budget · 2 messages summarized"]


async def test_the_run_budget_is_published_for_the_retry_layer(
    install: InstallProvider,
) -> None:
    """`--max-seconds 300` reached nothing: the HTTP layer only ever knew its
    own 90s ceiling, so the flag bought no extra time at all."""
    seen: list[float | None] = []

    class Watcher(FakeProvider):
        async def generate(self, *args: object, **kwargs: object) -> ModelResponse:
            seen.append(http_module._run_deadline.get())
            return await super().generate(*args, **kwargs)  # type: ignore[arg-type]

    install(Watcher([_response(text="done")]))

    await Agent("fake", limits=Limits(max_seconds=300.0)).run("go")

    assert seen and seen[0] is not None, "the run's deadline was visible inside the call"
