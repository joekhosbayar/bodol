"""Agent, Limits, RunResult, and the step loop.

The loop is the only place in the framework that owns time: it opens the
telemetry trace, builds the provider inside it, advances the step counter, and
decides when to stop. Everything it calls is a layer that already works on its
own — `create_provider` for a traced adapter, `ToolRegistry` for dispatch,
`ContextManager` for the transcript budget.

Three rules shape the design:

**A run always produces a RunResult.** Hitting a limit, a truncated answer, a
provider that raised mid-call, a compaction that could not proceed — every one
of those comes back as a `stop_reason` with the partial history attached, not as
an exception. The one exception is provider *construction*: a bad spec or a
missing API key is a configuration error, so `create_provider` failures
propagate rather than masquerading as a run outcome.

**Limits are checked before each model call, never after.** Checking after means
discovering the budget is blown by having already blown it. `max_steps=0`
returns immediately without calling anything.

That check cannot prevent every overrun, and pretending otherwise would be the
worse failure. `max_seconds` bounds the gaps between calls; a retry policy
bounds the inside of one call. They are two clocks, so a call that starts at
second 58 of a 120-second budget may legally run to second 131 — and if it ends
on an error, the limit check never runs again to notice. So the elapsed time is
compared against the budget on **every** exit path and an overrun is reported.
Until the run's remaining time is handed down into the call itself, this is a
warning rather than a guarantee, and it says so out loud.

**A step is a model call.** The two provider calls of a tool round-trip share
one step number, which is what makes a trace readable: step 3 is one turn of
thinking, whether or not it also ran four tools.

Cost has a hole in it worth knowing about. `events.cost_usd` returns None for a
model with no published rates, and folding that into the running total as zero
would quietly claim the run was free. Unpriced calls are counted separately
instead, and `max_cost` simply cannot fire for such a model — `max_steps` and
`max_seconds` are the only live bounds there.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from bodol.context import CompactionError, ContextManager, ContextPolicy
from bodol.providers import create_provider
from bodol.providers.base import (
    ContentBlock,
    FinishReason,
    Message,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)
from bodol.telemetry import events
from bodol.tools import ToolRegistry

# Aliased so a test can install a deterministic clock here instead of replacing
# `time.monotonic` globally, which the event loop is also using to schedule.
_monotonic = time.monotonic

logger = logging.getLogger(__name__)


class StopReason(StrEnum):
    """Why the loop stopped.

    Truncation and an unclassifiable stop get their own reasons rather than
    collapsing into DONE: an answer cut off mid-sentence is not a finished
    answer, and a caller that cannot tell the difference will treat it as one.
    """

    DONE = "done"
    MAX_TOKENS = "max_tokens"
    UNKNOWN = "unknown"
    MAX_STEPS = "max_steps"
    MAX_COST = "max_cost"
    MAX_SECONDS = "max_seconds"
    COMPACTION_FAILED = "compaction_failed"
    ERROR = "error"


_FINISH_TO_STOP = {
    FinishReason.STOP: StopReason.DONE,
    FinishReason.MAX_TOKENS: StopReason.MAX_TOKENS,
    FinishReason.UNKNOWN: StopReason.UNKNOWN,
}


@dataclass(frozen=True, slots=True)
class Limits:
    """The three bounds a run cannot cross, plus the per-call output cap.

    `max_steps` counts model calls, not tool calls: one turn that dispatches
    six tools is one step. `max_tokens_per_call` is not exposed on the CLI; it
    mirrors the `Provider.generate` default so the loop is explicit about what
    it asks for.
    """

    max_steps: int = 12
    max_cost_usd: float = 0.25
    max_seconds: float = 120.0
    max_tokens_per_call: int = 4096


# The CLI's documented defaults, shared rather than constructed per call site.
DEFAULT_LIMITS = Limits()


@dataclass(frozen=True, slots=True)
class RunResult:
    """Everything the caller needs to judge a run, finished or not.

    `messages` is the full transcript as it stood at the stop — post-compaction
    if compaction ran, since that is what the model actually had in front of
    it. `cost_usd` covers the priced calls only; `unpriced_calls` says how many
    calls it does not account for.
    """

    text: str | None
    stop_reason: StopReason
    steps: int
    cost_usd: float
    unpriced_calls: int
    usage: Usage
    trace_id: str | None
    messages: tuple[Message, ...] = ()
    error: str | None = None


def _merge_usage(total: Usage, add: Usage) -> Usage:
    """Sum billed tokens across calls.

    Input tokens are summed as billed, so a long run's total exceeds the size
    of its final transcript. That is the number that was paid for.
    """
    return Usage(
        input_tokens=total.input_tokens + add.input_tokens,
        output_tokens=total.output_tokens + add.output_tokens,
        cached_tokens=total.cached_tokens + add.cached_tokens,
        reasoning_tokens=total.reasoning_tokens + add.reasoning_tokens,
    )


def _assistant_turn(response: ModelResponse) -> Message:
    """Replay the model's tool-calling turn back into history.

    Signed reasoning is echoed first, ahead of prose and tool calls. Gemini
    validates a replayed `function_call` against the `thought` step that
    preceded it and rejects the whole request when that step is missing, so the
    original order has to survive the round trip.

    A call whose arguments never decoded still has to be echoed before its
    error result, with `{}` as args and the original string as `raw_args`.
    """
    blocks: list[ContentBlock] = list(response.thoughts)
    if response.text:
        blocks.append(TextBlock(text=response.text))
    blocks.extend(
        ToolUseBlock(id=call.id, name=call.name, args=call.args or {}, raw_args=call.raw_args)
        for call in response.tool_calls
    )
    return Message(role="assistant", content=tuple(blocks))


class Agent:
    """One task, one trace, one provider, run to a stop condition."""

    def __init__(
        self,
        provider: str,
        *,
        default_family: str | None = None,
        tools: ToolRegistry | None = None,
        limits: Limits = DEFAULT_LIMITS,
        system: str | None = None,
        context: ContextPolicy | None = None,
    ) -> None:
        self.spec = provider
        self.default_family = default_family
        self.tools = tools if tools is not None else ToolRegistry()
        self.limits = limits
        self.system = system
        self.context_policy = context

    async def run(self, task: str) -> RunResult:
        """Drive the task to a stop condition and report what happened."""
        messages: list[Message] = [Message(role="user", content=(TextBlock(text=task),))]
        manager = ContextManager(self.context_policy) if self.context_policy else None
        started = _monotonic()
        steps = 0
        cost = 0.0
        unpriced = 0
        usage = Usage()

        with events.start_trace() as trace_id:

            def result(
                reason: StopReason,
                *,
                text: str | None = None,
                error: str | None = None,
            ) -> RunResult:
                # Every stop path funnels through here, which is the point: the
                # overshoot has to be reported even when the run ended for some
                # other reason. A 429 sequence that pushed a run to 130s of a
                # 120s budget stops on ERROR, never re-reaches the limit check,
                # and would otherwise leave no sign that the budget was passed.
                elapsed = _monotonic() - started
                if elapsed > self.limits.max_seconds:
                    logger.warning(
                        "time budget exceeded: the run took %.1fs against a %gs "
                        "limit. Limits are checked between model calls, so a "
                        "call that starts inside the budget can finish outside "
                        "it — retries and their waits happen inside one call.",
                        elapsed,
                        self.limits.max_seconds,
                    )
                return RunResult(
                    text=text,
                    stop_reason=reason,
                    steps=steps,
                    cost_usd=round(cost, 10),
                    unpriced_calls=unpriced,
                    usage=usage,
                    trace_id=trace_id,
                    messages=tuple(messages),
                    error=error,
                )

            # Before the adapter exists, so a construction failure is not
            # charged against the trace as a run outcome.
            provider = create_provider(self.spec, default_family=self.default_family)
            try:
                while True:
                    breach = self._breached(steps=steps, cost=cost, started=started)
                    if breach is not None:
                        return result(breach)

                    events.advance_step()
                    try:
                        response = await provider.generate(
                            messages,
                            system=self.system,
                            tools=self.tools.specs,
                            max_tokens=self.limits.max_tokens_per_call,
                        )
                    except Exception as exc:
                        # TracedProvider has already written the error record;
                        # the loop's job is to report it, not to re-raise it.
                        return result(
                            StopReason.ERROR, error=f"{type(exc).__name__}: {exc}"
                        )

                    steps += 1
                    usage = _merge_usage(usage, response.usage)
                    call_cost = events.cost_usd(
                        response.usage, events.rates_for(response.provider, response.model)
                    )
                    if call_cost is None:
                        unpriced += 1
                    else:
                        cost += call_cost
                    if manager is not None:
                        manager.observe(response)

                    if not response.wants_tools:
                        if response.text is not None:
                            messages.append(
                                Message(
                                    role="assistant", content=(TextBlock(text=response.text),)
                                )
                            )
                        reason = _FINISH_TO_STOP.get(response.finish_reason, StopReason.UNKNOWN)
                        return result(reason, text=response.text)

                    messages.append(_assistant_turn(response))
                    messages.append(
                        Message(role="user", content=await self._dispatch(response.tool_calls))
                    )

                    if manager is not None and manager.over_budget():
                        # Summarization goes through the same traced provider,
                        # so it lands in the trace under the step that caused
                        # it, and is not counted against max_steps.
                        try:
                            messages = await manager.compact(
                                messages, provider=provider, system=self.system
                            )
                        except CompactionError as exc:
                            return result(StopReason.COMPACTION_FAILED, error=str(exc))
            finally:
                # Closes the JSONL sink too, so the trace file is complete even
                # when the run ended on an exception.
                await provider.aclose()

    def _breached(self, *, steps: int, cost: float, started: float) -> StopReason | None:
        """First limit crossed, in a fixed order so runs are reproducible."""
        if steps >= self.limits.max_steps:
            return StopReason.MAX_STEPS
        if cost >= self.limits.max_cost_usd:
            return StopReason.MAX_COST
        if _monotonic() - started >= self.limits.max_seconds:
            return StopReason.MAX_SECONDS
        return None

    async def _dispatch(self, calls: Sequence[ToolCall]) -> tuple[ContentBlock, ...]:
        """Results for every call, in the order the model asked for them.

        Unregistered tools cannot go through `dispatch_all`: it gathers, so one
        `UnknownToolError` would discard the results of the calls that were
        fine. They are answered with an error block instead, which is the same
        shape the registry already returns for undecodable arguments — the
        model sees what it got wrong and can correct itself on the next step.
        """
        known = {spec.name for spec in self.tools.specs}
        dispatchable = [call for call in calls if call.name in known]
        results: dict[str, ContentBlock] = {
            block.call_id: block for block in await self.tools.dispatch_all(dispatchable)
        }
        available = ", ".join(sorted(known)) or "none"
        for call in calls:
            if call.name not in known:
                results[call.id] = ToolResultBlock(
                    call_id=call.id,
                    name=call.name,
                    content=f"Unknown tool {call.name!r}. Available tools: {available}",
                    is_error=True,
                )
        return tuple(results[call.id] for call in calls)


__all__ = ["DEFAULT_LIMITS", "Agent", "Limits", "RunResult", "StopReason"]
