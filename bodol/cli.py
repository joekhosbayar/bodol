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
import dataclasses
import json
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer

from bodol import __version__, config
from bodol.agent import Agent, Limits, RunResult, StopReason, prompts
from bodol.context import policy_for
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
        typer.secho("(no text)", fg=typer.colors.BRIGHT_BLACK, err=True)

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
        str, typer.Option("--provider", "-p", help='e.g. "gemini:gemini-2.0-flash"')
    ] = config.DEFAULT_PROVIDER,
    prompt_version: Annotated[
        str, typer.Option("--system", "-s", help="Prompt version to load, e.g. v2.")
    ] = "v1",
    max_steps: Annotated[int, typer.Option(help="Stop after N steps.")] = 12,
    max_cost: Annotated[float, typer.Option(help="Stop after $N.")] = 0.25,
    max_seconds: Annotated[int, typer.Option(help="Stop after N seconds.")] = 120,
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

    # No known window means no verified budget, so compaction stays off rather
    # than being sized from a guess. This is also where a malformed spec first
    # surfaces, before anything has been constructed.
    try:
        policy = policy_for(provider)
    except ProviderRegistryError as exc:
        _die(str(exc))
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
