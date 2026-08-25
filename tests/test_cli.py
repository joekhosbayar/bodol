"""Tests for the CLI shell: parse, delegate, print.

No test here reaches a provider. `bodol.cli.Agent` is replaced by a stub that
records what it was constructed with and returns a canned RunResult, which is
exactly the boundary cli.py owns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from bodol import cli, config
from bodol.agent import Limits, RunResult, StopReason
from bodol.context import ContextPolicy
from bodol.providers import ProviderCredentialError
from bodol.providers.base import Message, TextBlock, Usage

runner = CliRunner()


def _result(
    *,
    text: str | None = "the answer",
    stop_reason: StopReason = StopReason.DONE,
    steps: int = 2,
    cost_usd: float = 0.0042,
    unpriced_calls: int = 0,
    error: str | None = None,
) -> RunResult:
    return RunResult(
        text=text,
        stop_reason=stop_reason,
        steps=steps,
        cost_usd=cost_usd,
        unpriced_calls=unpriced_calls,
        usage=Usage(input_tokens=1000, output_tokens=234),
        trace_id="tr_abc123",
        messages=(Message(role="user", content=(TextBlock(text="task"),)),),
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
        StubAgent.last = self

    async def run(self, task: str) -> RunResult:
        self.tasks.append(task)
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
        created.append(stub)
        return stub

    shape: dict[str, Any] = {"result": None, "raises": None}
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
    assert [s.name for s in stub.kwargs["tools"].specs] == ["calculator", "file_read", "grep"]


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


# ---------------------------------------------------------------- output


def test_answer_on_stdout_summary_on_stderr(agent) -> None:  # type: ignore[no-untyped-def]
    result = runner.invoke(cli.app, ["run", "hello"])

    assert result.stdout.strip() == "the answer", "redirecting stdout yields just the answer"
    assert "done" in result.stderr
    assert "2 steps" in result.stderr
    assert "$0.0042" in result.stderr
    assert "1,234 tokens" in result.stderr
    assert str(config.TRACE_DIR / "tr_abc123.jsonl") in result.stderr


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
