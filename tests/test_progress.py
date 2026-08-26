"""Progress line rendering.

These are the lines a person reads while a run is in flight, so the tests are
about wording and width as much as correctness. Pure functions, so none of this
needs an agent, a provider, or a clock.
"""

from __future__ import annotations

from bodol.agent import progress
from bodol.providers.base import (
    FinishReason,
    Message,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
    ToolUseBlock,
    Usage,
)


def _response(*, text: str | None = None, latency_ms: float = 1234.0) -> ModelResponse:
    return ModelResponse(
        id="resp_1",
        model="gemini-3.6-flash",
        provider="gemini",
        finish_reason=FinishReason.TOOL_CALLS,
        usage=Usage(input_tokens=100, output_tokens=20),
        latency_ms=latency_ms,
        text=text,
    )


# ---------------------------------------------------------------- step header


def test_the_step_header_answers_four_questions_at_once() -> None:
    line = progress.step_line(
        3,
        _response(latency_ms=6188.0),
        elapsed=51.4,
        budget=120.0,
        cost_total=0.0186,
        tokens_total=23110,
    )

    assert line == "step 3 · 6.2s · 51s/120s · $0.0186 · 23,110 tokens"


def test_the_budget_is_shown_without_a_pointless_decimal() -> None:
    """`max_seconds` is a float carrying a whole number. "120.0s" reads like a
    measurement; the budget is a setting."""
    line = progress.step_line(
        1, _response(), elapsed=0.4, budget=120.0, cost_total=0.0, tokens_total=0
    )

    assert "0s/120s" in line

    fractional = progress.step_line(
        1, _response(), elapsed=0.4, budget=0.5, cost_total=0.0, tokens_total=0
    )

    assert "0s/0.5s" in fractional, "a real fraction still survives"


# ------------------------------------------------------------- compaction


def test_the_compact_line_says_what_triggered_it_and_what_it_did() -> None:
    line = progress.compact_line(
        2,
        input_tokens=983,
        budget=800,
        before=7,
        after=3,
    )

    assert line == "compact step 2 · 983 in over 800 budget · 5 messages summarized"


def test_the_compact_line_formats_large_numbers_with_commas() -> None:
    line = progress.compact_line(
        14,
        input_tokens=128_400,
        budget=120_000,
        before=22,
        after=8,
    )

    assert "128,400 in over 120,000 budget" in line


# ---------------------------------------------------------------- model prose


def test_prose_is_quoted_and_flattened_to_one_line() -> None:
    line = progress.text_line(_response(text="Now checking\n  the largest file"))

    assert line == '"Now checking the largest file"'


def test_prose_is_clipped_to_a_readable_width() -> None:
    line = progress.text_line(_response(text="x" * 500))

    assert line is not None
    assert len(line) == progress.MAX_TEXT + 2, "plus the two quotes"
    assert line.endswith('…"')


def test_no_prose_is_reported_as_nothing_rather_than_empty_quotes() -> None:
    assert progress.text_line(_response(text=None)) is None
    assert progress.text_line(_response(text="   \n ")) is None, "whitespace is not prose"


# ---------------------------------------------------------------- tool calls


def test_a_call_shows_the_arguments_it_will_run_with() -> None:
    call = ToolCall(id="c1", name="grep", args={"pattern": "def ", "path": "bodol"})

    assert progress.call_line(call) == "→ grep(pattern='def ', path='bodol')"


def test_one_huge_argument_cannot_crowd_out_the_others() -> None:
    call = ToolCall(id="c1", name="grep", args={"pattern": "x" * 200, "path": "bodol"})

    line = progress.call_line(call)

    assert "path=" in line, "the argument that says where it looked survives"
    assert "…" in line


def test_undecodable_arguments_are_shown_as_the_model_sent_them() -> None:
    """The wire value is the entire diagnostic here."""
    call = ToolCall(id="c1", name="grep", args=None, raw_args='{"pattern": "def ",,}')

    assert progress.call_line(call) == '→ grep({"pattern": "def ",,})'


# ---------------------------------------------------------------- tool results


def test_a_result_is_named_so_concurrent_calls_stay_attributable() -> None:
    block = ToolResultBlock(call_id="c1", name="grep", content="x" * 2150)

    assert progress.result_line(block) == "← grep · ok · 2.1 KB"


def test_a_small_result_is_sized_in_bytes() -> None:
    block = ToolResultBlock(call_id="c1", name="grep", content="no matches")

    assert progress.result_line(block) == "← grep · ok · 10 B"


def test_size_is_bytes_on_the_wire_not_characters() -> None:
    """A tool result is billed as tokens over the wire. Counting `len` on text
    that is half multibyte under-reports what it actually cost to send."""
    block = ToolResultBlock(call_id="c1", name="file_read", content="—" * 400)

    assert progress.result_line(block) == "← file_read · ok · 1.2 KB"


def test_a_failed_tool_shows_its_reason_verbatim() -> None:
    """A success is summarized because its content is for the model. A failure is
    quoted because it is for the person watching."""
    block = ToolResultBlock(
        call_id="c1",
        name="ls",
        content="Unknown tool 'ls'. Available tools: calculator, file_read, grep",
        is_error=True,
    )

    line = progress.result_line(block)

    assert line.startswith("← ls · error · Unknown tool 'ls'.")


# ---------------------------------------------------------------- the salvage


def _transcript() -> list[Message]:
    return [
        Message("user", (TextBlock("how many python files?"),)),
        Message(
            "assistant",
            (
                TextBlock("Let me look."),
                ToolUseBlock(id="c1", name="grep", args={}),
                ToolUseBlock(id="c2", name="grep", args={}),
            ),
        ),
        Message("user", (ToolResultBlock(call_id="c1", name="grep", content="hits"),)),
        Message(
            "assistant",
            (
                TextBlock("I still need the sizes."),
                ToolUseBlock(id="c3", name="file_read", args={}),
            ),
        ),
    ]


def test_the_trail_counts_repeats_busiest_first() -> None:
    assert progress.trail(_transcript()) == "grep ×2, file_read"


def test_the_trail_of_a_run_that_called_nothing_is_empty() -> None:
    assert progress.trail([Message("user", (TextBlock("hi"),))]) == ""


def test_last_words_finds_the_most_recent_thing_the_model_said() -> None:
    assert progress.last_words(_transcript()) == "I still need the sizes."


def test_last_words_ignores_user_turns() -> None:
    """Tool results live in user turns. They are not the model talking, and
    echoing one back as "last words" would attribute it to the wrong speaker."""
    messages = [
        Message("user", (TextBlock("the task"),)),
        Message("assistant", (ToolUseBlock(id="c1", name="grep", args={}),)),
        Message("user", (ToolResultBlock(call_id="c1", name="grep", content="hits"),)),
    ]

    assert progress.last_words(messages) is None
