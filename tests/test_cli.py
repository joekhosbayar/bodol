"""Tests for the CLI shell: parse, delegate, print.

No test here reaches a provider. `bodol.cli.Agent` is replaced by a stub that
records what it was constructed with and returns a canned RunResult, which is
exactly the boundary cli.py owns.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bodol import cli, config
from bodol.agent import Limits, RunResult, StopReason
from bodol.context import ContextPolicy
from bodol.providers import ProviderCredentialError
from bodol.providers.base import Message, TextBlock, ToolResultBlock, ToolUseBlock, Usage

runner = CliRunner()


def _result(
    *,
    text: str | None = "the answer",
    stop_reason: StopReason = StopReason.DONE,
    steps: int = 2,
    cost_usd: float = 0.0042,
    unpriced_calls: int = 0,
    error: str | None = None,
    messages: tuple[Message, ...] | None = None,
) -> RunResult:
    if messages is None:
        messages = (Message(role="user", content=(TextBlock(text="task"),)),)
    return RunResult(
        text=text,
        stop_reason=stop_reason,
        steps=steps,
        cost_usd=cost_usd,
        unpriced_calls=unpriced_calls,
        usage=Usage(input_tokens=1000, output_tokens=234),
        trace_id="tr_abc123",
        messages=messages,
        error=error,
    )


class StubAgent:
    """Captures constructor arguments; never touches a provider."""

    last: StubAgent | None = None

    def __init__(self, provider: str, **kwargs: Any) -> None:
        self.provider = provider
        self.kwargs = kwargs
        self.tasks: list[str] = []
        self.result = _result()
        self.raises: Exception | None = None
        # Stands in for the HTTP layer announcing a retry mid-run.
        self.warns: str | None = None
        # Stands in for the loop announcing a completed step mid-run.
        self.notes: str | None = None
        StubAgent.last = self

    async def run(self, task: str) -> RunResult:
        self.tasks.append(task)
        if self.notes is not None:
            logging.getLogger("bodol.agent.loop").info(self.notes)
        if self.warns is not None:
            logging.getLogger("bodol.providers.http").warning(self.warns)
        if self.raises is not None:
            raise self.raises
        return self.result


@pytest.fixture
def agent(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Install the stub and hand back a hook for shaping its outcome."""
    created: list[StubAgent] = []

    def factory(provider: str, **kwargs: Any) -> StubAgent:
        stub = StubAgent(provider, **kwargs)
        if shape["result"] is not None:
            stub.result = shape["result"]
        if shape["raises"] is not None:
            stub.raises = shape["raises"]
        stub.warns = shape["warns"]
        stub.notes = shape["notes"]
        created.append(stub)
        return stub

    shape: dict[str, Any] = {"result": None, "raises": None, "warns": None, "notes": None}
    monkeypatch.setattr(cli, "Agent", factory)
    return type("Hook", (), {"shape": shape, "created": created})()


# ---------------------------------------------------------------- shell basics


def test_version() -> None:
    result = runner.invoke(cli.app, ["--version"])
    assert result.exit_code == 0
    assert "bodol" in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "eval", "compare", "cost", "replay"):
        assert cmd in result.stdout


def test_unimplemented_commands_still_exit_2(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text("cases: []\n", encoding="utf-8")

    assert runner.invoke(cli.app, ["eval", str(suite)]).exit_code == 2
    assert runner.invoke(cli.app, ["cost", str(tmp_path)]).exit_code == 2


def test_run_no_longer_offers_a_trace_flag() -> None:
    """create_provider owns sink creation, so the flag could not be honored."""
    result = runner.invoke(cli.app, ["run", "--help"])
    assert "--trace" not in result.stdout
    assert "--json" in result.stdout


# ---------------------------------------------------------------- delegation


def test_run_passes_parsed_options_to_the_agent(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(
        cli.app,
        [
            "run",
            "count the todos",
            "--provider",
            "anthropic:claude-haiku-4-5",
            "--max-steps",
            "5",
            "--max-cost",
            "0.10",
            "--max-seconds",
            "30",
        ],
    )

    assert result.exit_code == 0
    (stub,) = agent.created
    assert stub.provider == "anthropic:claude-haiku-4-5"
    assert stub.tasks == ["count the todos"]
    assert stub.kwargs["limits"] == Limits(max_steps=5, max_cost_usd=0.10, max_seconds=30.0)


def test_run_loads_the_prompt_and_registers_the_builtins(agent) -> None:  # type: ignore[no-untyped-def]
    runner.invoke(cli.app, ["run", "hello"])

    (stub,) = agent.created
    assert "bodol" in stub.kwargs["system"], "the v1 prompt text is passed, not the version"
    assert [s.name for s in stub.kwargs["tools"].specs] == [
        "calculator",
        "file_read",
        "grep",
        "list_files",
    ]


def test_run_derives_the_context_budget_from_the_model(agent) -> None:  # type: ignore[no-untyped-def]
    runner.invoke(cli.app, ["run", "hello", "--provider", "gemini:gemini-3.7-flash"])

    (stub,) = agent.created
    policy = stub.kwargs["context"]
    assert isinstance(policy, ContextPolicy)
    assert policy.max_input_tokens == int(1_048_576 * 0.6)


def test_unknown_window_disables_compaction_and_says_so(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(cli.app, ["run", "hello", "--provider", "openai:gpt-5.6-luna"])

    (stub,) = agent.created
    assert stub.kwargs["context"] is None
    assert "compaction disabled" in result.stderr


def test_max_input_tokens_overrides_the_derived_budget(agent) -> None:  # type: ignore[no-untyped-def]
    """A manual budget replaces the window-derived one, however large the window."""
    runner.invoke(
        cli.app,
        ["run", "hello", "--provider", "gemini:gemini-3.7-flash", "--max-input-tokens", "2000"],
    )

    (stub,) = agent.created
    policy = stub.kwargs["context"]
    assert isinstance(policy, ContextPolicy)
    assert policy.max_input_tokens == 2000
    assert policy.min_recent_turns == 2


def test_max_input_tokens_enables_compaction_on_an_unknown_window(agent) -> None:  # type: ignore[no-untyped-def]
    """The whole point: compaction is testable on any provider, not just ones
    with a verified context window."""
    result = runner.invoke(
        cli.app,
        ["run", "hello", "--provider", "openai:gpt-5.6-luna", "--max-input-tokens", "2000"],
    )

    (stub,) = agent.created
    policy = stub.kwargs["context"]
    assert isinstance(policy, ContextPolicy)
    assert policy.max_input_tokens == 2000
    assert "compaction disabled" not in result.stderr


def test_min_recent_turns_is_passed_through(agent) -> None:  # type: ignore[no-untyped-def]
    runner.invoke(
        cli.app,
        ["run", "hello", "--max-input-tokens", "2000", "--min-recent-turns", "4"],
    )

    (stub,) = agent.created
    policy = stub.kwargs["context"]
    assert isinstance(policy, ContextPolicy)
    assert policy.min_recent_turns == 4


def test_min_recent_turns_overrides_the_derived_policy(agent) -> None:  # type: ignore[no-untyped-def]
    """The flag also adjusts a window-derived policy, not just a manual one."""
    runner.invoke(
        cli.app,
        ["run", "hello", "--provider", "gemini:gemini-3.7-flash", "--min-recent-turns", "3"],
    )

    (stub,) = agent.created
    policy = stub.kwargs["context"]
    assert isinstance(policy, ContextPolicy)
    assert policy.min_recent_turns == 3
    assert policy.max_input_tokens == int(1_048_576 * 0.6), "budget unchanged"


# ---------------------------------------------------------------- output


def test_answer_on_stdout_summary_on_stderr(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(cli.app, ["run", "hello"])

    assert result.stdout.strip() == "the answer", "redirecting stdout yields just the answer"
    assert "done" in result.stderr
    assert "2 steps" in result.stderr
    assert "$0.0042" in result.stderr
    assert "1,234 tokens" in result.stderr
    assert str(config.TRACE_DIR / "tr_abc123.jsonl") in result.stderr


def test_a_retry_is_visible_while_the_run_is_still_going(agent) -> None:  # type: ignore[no-untyped-def]
    """A silent five-minute retry sequence looks exactly like a hung process.

    The HTTP layer logs because a library must not own stdio; the CLI is what
    decides those lines reach a terminal, and in whose voice.
    """
    agent.shape["warns"] = "retry 1/4 · HTTP 500 · waiting 2.0s · high demand"

    result = runner.invoke(cli.app, ["run", "hello"])

    assert "  retry 1/4 · HTTP 500 · waiting 2.0s · high demand" in result.stderr


def test_step_progress_reaches_the_terminal_while_the_run_is_going(agent) -> None:  # type: ignore[no-untyped-def]
    """Progress is INFO, not WARNING: it is not trouble. The CLI still shows it,
    because the question "is this thing still working" is the same question."""
    agent.shape["notes"] = "step 1 · 6.2s · 51s/120s · $0.0186 · 23,110 tokens"

    result = runner.invoke(cli.app, ["run", "hello"])

    assert "  step 1 · 6.2s · 51s/120s · $0.0186 · 23,110 tokens" in result.stderr


def _stuck_transcript() -> tuple[Message, ...]:
    """Eight steps of grepping, no answer — the shape of a real cut-short run."""
    return (
        Message(role="user", content=(TextBlock(text="how many python files?"),)),
        Message(
            role="assistant",
            content=(
                TextBlock(text="I still need the sizes."),
                ToolUseBlock(id="c1", name="grep", args={}),
                ToolUseBlock(id="c2", name="grep", args={}),
                ToolUseBlock(id="c3", name="file_read", args={}),
            ),
        ),
        Message(
            role="user",
            content=(ToolResultBlock(call_id="c1", name="grep", content="hits"),),
        ),
    )


def test_a_run_with_no_answer_still_shows_what_it_did(agent) -> None:  # type: ignore[no-untyped-def]
    """$0.056 of tool work used to print the words "(no text)" and nothing else.

    The transcript was in the RunResult the whole time.
    """
    agent.shape["result"] = _result(
        text=None, stop_reason=StopReason.MAX_SECONDS, messages=_stuck_transcript()
    )

    result = runner.invoke(cli.app, ["run", "hello"])

    assert "  did · grep ×2, file_read" in result.stderr
    assert '  last words · "I still need the sizes."' in result.stderr
    assert result.stdout.strip() == "", "a partial thought is not an answer and stdout is answers"


def test_salvage_lines_are_skipped_when_there_is_nothing_to_salvage(agent) -> None:  # type: ignore[no-untyped-def]
    agent.shape["result"] = _result(text=None)

    result = runner.invoke(cli.app, ["run", "hello"])

    assert "(no text)" in result.stderr
    assert "did ·" not in result.stderr
    assert "last words" not in result.stderr


def test_an_answered_run_says_nothing_about_the_route_it_took(agent) -> None:  # type: ignore[no-untyped-def]
    agent.shape["result"] = _result(text="42 files", messages=_stuck_transcript())

    result = runner.invoke(cli.app, ["run", "hello"])

    assert result.stdout.strip() == "42 files"
    assert "did ·" not in result.stderr, "the trail is a consolation, not a habit"


def test_the_log_handler_does_not_outlive_the_run(agent) -> None:  # type: ignore[no-untyped-def]
    """A StreamHandler captures the stream it was built with.

    Leaving one attached to a module-level logger means the next writer in the
    process ends up writing into a pipe that has already closed.
    """
    runner.invoke(cli.app, ["run", "hello"])

    assert logging.getLogger("bodol").handlers == []


def test_missing_text_is_reported_without_pretending(agent) -> None:  # type: ignore[no-untyped-def]
    agent.shape["result"] = _result(text=None)

    result = runner.invoke(cli.app, ["run", "hello"])

    assert result.stdout.strip() == ""
    assert "(no text)" in result.stderr


def test_unpriced_calls_qualify_the_cost(agent) -> None:  # type: ignore[no-untyped-def]
    agent.shape["result"] = _result(steps=3, unpriced_calls=2, cost_usd=0.001)

    result = runner.invoke(cli.app, ["run", "hello"])

    assert "cost covers 1 of 3 calls" in result.stderr


def test_singular_step_reads_correctly(agent) -> None:  # type: ignore[no-untyped-def]
    agent.shape["result"] = _result(steps=1)

    assert "1 step ·" in runner.invoke(cli.app, ["run", "hello"]).stderr


def test_json_output_carries_the_whole_result(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(cli.app, ["run", "hello", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["text"] == "the answer"
    assert payload["stop_reason"] == "done"
    assert payload["trace_id"] == "tr_abc123"
    assert payload["usage"]["total_tokens"] == 1234, "a property, added explicitly"
    assert payload["messages"], "the transcript is included"


# ---------------------------------------------------------------- exit codes


@pytest.mark.parametrize(
    "stop_reason",
    [
        StopReason.MAX_STEPS,
        StopReason.MAX_COST,
        StopReason.MAX_SECONDS,
        StopReason.MAX_TOKENS,
        StopReason.COMPACTION_FAILED,
        StopReason.ERROR,
    ],
)
def test_every_run_outcome_exits_zero(agent, stop_reason: StopReason) -> None:  # type: ignore[no-untyped-def]
    """The run happened. Its outcome is on stderr, not in the exit code."""
    agent.shape["result"] = _result(stop_reason=stop_reason, error="something went wrong")

    result = runner.invoke(cli.app, ["run", "hello"])

    assert result.exit_code == 0
    assert str(stop_reason) in result.stderr
    assert "something went wrong" in result.stderr


def test_unknown_prompt_version_exits_2_with_the_path(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(cli.app, ["run", "hello", "--system", "v99"])

    assert result.exit_code == 2
    assert "v99.md" in result.stderr
    assert "available prompt versions: v1" in result.stderr
    assert agent.created == [], "the agent is never constructed"


def test_path_like_prompt_version_exits_2(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(cli.app, ["run", "hello", "--system", "../.env"])

    assert result.exit_code == 2
    assert "bare name" in result.stderr


def test_malformed_provider_spec_exits_2(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(cli.app, ["run", "hello", "--provider", "gemini:"])

    assert result.exit_code == 2
    assert agent.created == [], "the spec is rejected before the agent is built"


def test_configuration_failure_exits_2(agent) -> None:  # type: ignore[no-untyped-def]
    agent.shape["raises"] = ProviderCredentialError("no credential for 'gemini'")

    result = runner.invoke(cli.app, ["run", "hello"])

    assert result.exit_code == 2, "the run never started, so this is not an outcome"
    assert "no credential" in result.stderr
