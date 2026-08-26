"""Thin shell over the Python API. No logic lives here — parse, delegate, print.

Two conventions the commands share:

**Exit code 2 means the run never started.** A bad provider spec, a missing API
key, an unknown prompt version — configuration, fixable by editing the command
or the environment. Everything the agent itself reports, including hitting a
limit or a provider that failed mid-run, exits 0: the run happened, and its
outcome is on stderr and in `--json`. Scripts branch on the printed stop reason,
not on a numeric ladder nobody remembers.

**Answers go to stdout, everything else to stderr.** `bodol run ... > out.txt`
should leave the model's answer in the file and the cost summary on the
terminal.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from bodol import __version__, config
from bodol.agent import Agent, Limits, RunResult, StopReason, progress, prompts
from bodol.context import ContextPolicy, policy_for
from bodol.providers import ProviderRegistryError
from bodol.tools.builtins import register_builtins

app = typer.Typer(
    name="bodol",
    help="A model-agnostic AI agent framework.",
    no_args_is_help=True,
    add_completion=False,
)


def _todo(module: str, what: str) -> NoReturn:
    """Placeholder until the core module exists."""
    typer.secho(f"not implemented: {what}", fg=typer.colors.YELLOW, err=True)
    typer.secho(f"  build it in {module}", fg=typer.colors.BRIGHT_BLACK, err=True)
    raise typer.Exit(code=2)


def _die(message: str) -> NoReturn:
    """Configuration failure: the run never started."""
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=2)


def _version(value: bool) -> None:
    if value:
        typer.echo(f"bodol {__version__}")
        raise typer.Exit


_STOP_COLOR = {
    StopReason.DONE: typer.colors.GREEN,
    StopReason.ERROR: typer.colors.RED,
    StopReason.COMPACTION_FAILED: typer.colors.RED,
}


def _payload(result: RunResult) -> dict[str, Any]:
    """The whole RunResult as JSON-ready data, transcript included."""
    data = dataclasses.asdict(result)
    # total_tokens is a property, so asdict skips it. It is the number most
    # readers of this output actually want.
    data["usage"]["total_tokens"] = result.usage.total_tokens
    return data


def _report(result: RunResult) -> None:
    """Answer to stdout, one summary line to stderr."""
    if result.text:
        typer.echo(result.text)
    else:
        # A run with no answer is not a run with nothing in it. Eight steps of
        # tool work and a partial thought are worth showing — on stderr, and
        # labelled, because a sentence written on the way to somewhere else is
        # not an answer and must not land in a redirected stdout as if it were.
        typer.secho("(no text)", fg=typer.colors.BRIGHT_BLACK, err=True)
        did = progress.trail(result.messages)
        if did:
            typer.secho(f"  did · {did}", fg=typer.colors.BRIGHT_BLACK, err=True)
        words = progress.last_words(result.messages)
        if words:
            typer.secho(f'  last words · "{words}"', fg=typer.colors.BRIGHT_BLACK, err=True)

    parts = [
        f"{result.steps} step{'s' if result.steps != 1 else ''}",
        f"${result.cost_usd:.4f}",
        f"{result.usage.total_tokens:,} tokens",
    ]
    if result.unpriced_calls:
        parts.append(
            f"cost covers {result.steps - result.unpriced_calls} of {result.steps} calls"
        )
    if result.trace_id:
        parts.append(str(config.TRACE_DIR / f"{result.trace_id}.jsonl"))

    color = _STOP_COLOR.get(result.stop_reason, typer.colors.YELLOW)
    typer.secho(f"  {result.stop_reason}", fg=color, nl=False, err=True)
    typer.secho(" · " + " · ".join(parts), fg=typer.colors.BRIGHT_BLACK, err=True)
    if result.error:
        typer.secho(f"  {result.error}", fg=typer.colors.RED, err=True)


@contextlib.contextmanager
def _live_notices() -> Iterator[None]:
    """Show the library's progress and warnings on stderr while the run is going.

    The layers below cannot print: a library that owns stdio is a library you
    cannot embed. So they log, and this — the one place that does own the
    terminal — decides what is worth watching and what it looks like.

    INFO carries progress, WARNING carries trouble, and both are shown, because
    both answer the same question: is this thing still working? A consumer that
    only wants the trouble filters to WARNING and gets exactly the retry notices
    and the budget overruns.

    Scoped to the run rather than installed once at startup, because a
    `StreamHandler` captures the stream it was built with. A handler that
    outlives the command it was built for goes on writing into a pipe that has
    since closed, which is how this was found.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("  %(message)s"))
    library = logging.getLogger("bodol")
    level, propagate = library.level, library.propagate
    library.addHandler(handler)
    library.setLevel(logging.INFO)
    # Ours is the only handler that should print these; an embedding app's root
    # configuration would otherwise show each line twice.
    library.propagate = False
    try:
        yield
    finally:
        library.removeHandler(handler)
        library.setLevel(level)
        library.propagate = propagate
        handler.close()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    config.ensure_dirs()


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="The task for the agent.")],
    provider: Annotated[
        str, typer.Option("--provider", "-p", help='e.g. "gemini:gemini-3.7-flash"')
    ] = config.DEFAULT_PROVIDER,
    prompt_version: Annotated[
        str, typer.Option("--system", "-s", help="Prompt version to load, e.g. v2.")
    ] = "v1",
    max_steps: Annotated[int, typer.Option(help="Stop after N steps.")] = 12,
    max_cost: Annotated[float, typer.Option(help="Stop after $N.")] = 0.25,
    max_seconds: Annotated[int, typer.Option(help="Stop after N seconds.")] = 120,
    max_input_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-input-tokens",
            help=(
                "Override the context budget. Compaction fires when the last"
                " call's input tokens exceed this. Works on any provider, even"
                " one with no known window. Set small (e.g. 2000) to test"
                " compaction on a short run."
            ),
        ),
    ] = None,
    min_recent_turns: Annotated[
        int,
        typer.Option(
            "--min-recent-turns",
            help="Turns kept verbatim at the tail when compaction fires.",
        ),
    ] = 2,
    json_output: Annotated[
        bool, typer.Option("--json", help="Emit the whole RunResult as JSON.")
    ] = False,
) -> None:
    """Run the agent on a single task."""
    try:
        system = prompts.load(prompt_version)
    except prompts.PromptError as exc:
        known = ", ".join(prompts.available()) or "none found"
        _die(f"{exc}\n  available prompt versions: {known}")

    # A manual budget overrides the derived one, and works even when the
    # model's window is unknown — which is the point: it makes compaction
    # testable on any provider without editing the context-window table.
    if max_input_tokens is not None:
        policy: ContextPolicy | None = ContextPolicy(
            max_input_tokens=max_input_tokens,
            min_recent_turns=min_recent_turns,
        )
    else:
        try:
            policy = policy_for(provider)
        except ProviderRegistryError as exc:
            _die(str(exc))
        if policy is not None and min_recent_turns != 2:
            policy = ContextPolicy(
                max_input_tokens=policy.max_input_tokens,
                summary_max_tokens=policy.summary_max_tokens,
                min_recent_turns=min_recent_turns,
            )
        if policy is None and not json_output:
            typer.secho(
                f"  no context window on record for {provider}; compaction disabled",
                fg=typer.colors.BRIGHT_BLACK,
                err=True,
            )

    agent = Agent(
        provider,
        tools=register_builtins(),
        limits=Limits(
            max_steps=max_steps, max_cost_usd=max_cost, max_seconds=float(max_seconds)
        ),
        system=system,
        context=policy,
    )
    try:
        with _live_notices():
            result = asyncio.run(agent.run(task))
    except ProviderRegistryError as exc:
        _die(str(exc))

    if json_output:
        typer.echo(json.dumps(_payload(result), indent=2, default=str))
    else:
        _report(result)


@app.command("eval")
def eval_(
    suite: Annotated[Path, typer.Argument(exists=True, help="Path to a suite YAML.")],
    repeat: Annotated[int, typer.Option("--repeat", "-r", help="Runs per case.")] = 5,
    batch: Annotated[bool, typer.Option("--batch", help="Use the provider batch API.")] = False,
    provider: Annotated[str, typer.Option("--provider", "-p")] = config.DEFAULT_PROVIDER,
    concurrency: Annotated[int, typer.Option(help="Parallel runs when not batching.")] = 4,
    out: Annotated[Path | None, typer.Option(help="Where to write results.")] = None,
) -> None:
    """Run an eval suite."""
    _todo("bodol/evals/harness.py", "suite execution")


@app.command()
def compare(
    suite: Annotated[Path, typer.Option("--suite", exists=True, help="Path to a suite YAML.")],
    models: Annotated[str, typer.Option("--models", help="Comma-separated provider strings.")],
    repeat: Annotated[int, typer.Option("--repeat", "-r")] = 5,
    out: Annotated[Path | None, typer.Option(help="Where to write the table.")] = None,
) -> None:
    """Compare pass rate, cost, and latency across models or prompt versions."""
    _todo("bodol/evals/harness.py", "comparison table")


@app.command()
def cost(
    path: Annotated[
        Path, typer.Argument(exists=True, help="A trace file or a directory of them.")
    ] = config.TRACE_DIR,
    by: Annotated[str, typer.Option(help="Group by: model, tool, step, or run.")] = "model",
) -> None:
    """Summarize spend across traces."""
    _todo("bodol/telemetry/events.py", "cost rollup")


@app.command()
def replay(
    trace: Annotated[Path, typer.Argument(exists=True, help="Path to traces/<id>.jsonl")],
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show full payloads.")] = False,
) -> None:
    """Replay a recorded run step by step."""
    _todo("bodol/telemetry/writer.py", "trace replay")


if __name__ == "__main__":
    app()
