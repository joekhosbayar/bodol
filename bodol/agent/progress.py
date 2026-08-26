"""Rendered lines for a run in flight, and for one that ended with nothing to show.

Why this exists: a run that took 124 seconds across 8 model calls printed its
first character after it was already over, and the summary was `(no text)`. The
work had happened — 14 greps, two file reads, $0.056 — and every trace of it was
either in a JSONL file nobody had opened yet or discarded outright.

Two things are deliberately *not* the answer here.

**Token streaming is not what was missing.** Across all 8 of those calls the model
emitted no prose at all: `text` was None every time, and the output tokens were
90% signed reasoning, which is an opaque blob by design. There was nothing to
stream a character at a time. What carried the signal was which tools it kept
reaching for, and how the clock was running.

**A progress line is not an answer.** Everything rendered here goes to stderr, and
partial prose is labelled as partial. A sentence the model wrote on its way to
somewhere else must never be printed where a caller redirecting stdout would
collect it as the result.

These are pure functions returning strings, with no logging and no I/O, so the
wording is testable without running an agent. A name ending in `_line` returns a
whole display line, quoting included; anything else returns a value for a caller
to place.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from bodol.providers.base import (
    Message,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
)

# One argument value, then the whole argument list. Two limits rather than one
# because a single huge value should not push every other argument off the line —
# `grep(pattern='…', path=…)` with the path missing is worse than useless, since
# the path is what tells you whether the model is searching where you expect.
MAX_VALUE = 40
MAX_ARGS = 80

# Prose and error text share a width: both are read, not scanned.
MAX_TEXT = 100


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _one_line(text: str, limit: int) -> str:
    """Collapse to a single line before clipping.

    Model prose arrives with newlines in it, and a multi-line log record breaks
    the alignment that makes a column of progress lines scannable.
    """
    return _clip(" ".join(text.split()), limit)


def step_line(
    step: int,
    response: ModelResponse,
    *,
    elapsed: float,
    budget: float,
    cost_total: float,
    tokens_total: int,
) -> str:
    """The header for one completed model call.

    Carries four numbers that answer four different questions: how slow was this
    call, how close is the run to being cut off, what has it cost, and how big has
    the transcript grown. The elapsed/budget pair is the one that earns its place
    twice over — a run that dies at `max_seconds` should have been visibly walking
    toward that number for a minute beforehand.
    """
    return (
        f"step {step} · {response.latency_ms / 1000:.1f}s"
        f" · {elapsed:.0f}s/{budget:g}s"
        f" · ${cost_total:.4f} · {tokens_total:,} tokens"
    )


def retry_line(
    step: int,
    error: str,
    retries_left: int,
    retries_allowed: int,
    elapsed: float,
    budget: float,
) -> str:
    """A turn about to be attempted again after the provider failed transiently.

    Carries the step number rather than an attempt number so it lines up with the
    `step N` header that eventually follows — the whole point of not advancing the
    step on a retry is that this reads as "step 3, again", not as a new step.

    The elapsed/budget pair is here for the same reason it is on `step_line`, and
    matters more: retries are the mechanism by which a run's remaining time
    disappears without anything visibly happening.
    """
    return (
        f"retry step {step} · {error}"
        f" · {retries_left} of {retries_allowed} left"
        f" · {elapsed:.0f}s/{budget:g}s"
    )


def compact_line(
    step: int,
    *,
    input_tokens: int,
    budget: int,
    before: int,
    after: int,
) -> str:
    """The context manager summarized the transcript to fit the budget.

    Announced because it is silent otherwise: the summarization call goes
    through the traced provider and lands in the JSONL, but a user watching
    the terminal has no way to know the transcript was rewritten under them.
    """
    evicted = before - after + 1  # the summary replaces the evicted middle
    return (
        f"compact step {step}"
        f" · {input_tokens:,} in over {budget:,} budget"
        f" · {evicted} messages summarized"
    )


def text_line(response: ModelResponse) -> str | None:
    """The model's prose, clipped to one line. None when it wrote none.

    Quoted, because it is the only part of a progress block that is the model
    talking rather than the framework reporting.
    """
    if not response.text or not response.text.strip():
        return None
    return f'"{_one_line(response.text, MAX_TEXT)}"'


def call_line(call: ToolCall) -> str:
    """A tool the model asked for, with its arguments, before it runs."""
    return f"→ {call.name}({_args(call)})"


def _args(call: ToolCall) -> str:
    if call.args is None:
        # Undecodable arguments are shown raw. The wire value is the whole
        # diagnostic: it is what the model actually emitted.
        return _clip(call.raw_args or "<missing>", MAX_ARGS)
    rendered = ", ".join(
        f"{key}={_clip(repr(value), MAX_VALUE)}" for key, value in call.args.items()
    )
    return _clip(rendered, MAX_ARGS)


def result_line(block: ToolResultBlock) -> str:
    """What a tool returned.

    Named even though the matching `call_line` just named it: tools dispatch
    concurrently, so a batch of five greps prints five arrows and then five
    results, and without the name the results cannot be attributed.

    A success is reported as a size because its content is for the model, not for
    you. A failure is reported verbatim, because that one is for you.
    """
    if block.is_error:
        return f"← {block.name} · error · {_one_line(block.content, MAX_TEXT)}"
    return f"← {block.name} · ok · {_size(block.content)}"


def _size(text: str) -> str:
    """Bytes as sent, not characters: a tool result is billed as tokens over the
    wire, and a multibyte file reads smaller than it costs when counted in `len`."""
    count = len(text.encode())
    if count < 1024:
        return f"{count} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KB"
    return f"{count / (1024 * 1024):.1f} MB"


def trail(messages: Sequence[Message]) -> str:
    """What the run actually did, as a count per tool, busiest first.

    This is the salvage line for a run that produced no answer. Nine greps and a
    file read is not an answer, but it is evidence — both of what the model was
    attempting and, when the same tool repeats a dozen times, that it was stuck.
    """
    counts = Counter(
        block.name
        for message in messages
        for block in message.content
        if isinstance(block, ToolUseBlock)
    )
    return ", ".join(
        f"{name} ×{count}" if count > 1 else name for name, count in counts.most_common()
    )


def last_words(messages: Sequence[Message]) -> str | None:
    """The most recent prose the model wrote, clipped. None when it wrote none.

    Searched backwards, and only in assistant turns: a user turn holds the task
    and the tool results, neither of which the model said.
    """
    for message in reversed(messages):
        if message.role != "assistant":
            continue
        for block in reversed(message.content):
            if isinstance(block, TextBlock) and block.text.strip():
                return _one_line(block.text, MAX_TEXT)
    return None
