"""Smoke tests for the CLI shell. Delete or rewrite these as the core lands."""

from typer.testing import CliRunner

from bodol.cli import app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "bodol" in result.stdout


def test_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("run", "eval", "compare", "cost", "replay"):
        assert cmd in result.stdout


def test_run_is_wired_but_unimplemented() -> None:
    result = runner.invoke(app, ["run", "count the todos"])
    assert result.exit_code == 2
