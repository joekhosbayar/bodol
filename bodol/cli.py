"""Thin shell over the Python API. No logic lives here — parse, delegate, print."""

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from bodol import __version__, config

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


def _version(value: bool) -> None:
    if value:
        typer.echo(f"bodol {__version__}")
        raise typer.Exit


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
    system: Annotated[
        str, typer.Option("--system", "-s", help="Prompt version to load, e.g. v2.")
    ] = "v1",
    max_steps: Annotated[int, typer.Option(help="Stop after N steps.")] = 12,
    max_cost: Annotated[float, typer.Option(help="Stop after $N.")] = 0.25,
    max_seconds: Annotated[int, typer.Option(help="Stop after N seconds.")] = 120,
    trace: Annotated[bool, typer.Option(help="Write a JSONL trace.")] = True,
) -> None:
    """Run the agent on a single task."""
    _todo("bodol/agent/loop.py", "Agent.run")


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
