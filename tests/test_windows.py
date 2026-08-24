"""Tests for context window lookup and the budget derived from it."""

from __future__ import annotations

from pathlib import Path

import pytest

from bodol import config
from bodol.context import windows


@pytest.fixture(autouse=True)
def _clear_table_cache() -> None:
    """The window table is process-cached; tests that repoint the file must clear it."""
    windows._window_table.cache_clear()


def test_known_window_is_found() -> None:
    assert windows.context_window("gemini", "gemini-2.0-flash") == 1_048_576
    assert windows.context_window("anthropic", "claude-haiku-4-5") == 200_000


def test_dated_anthropic_id_resolves() -> None:
    """Anthropic returns claude-haiku-4-5-20251001; the table is keyed undated."""
    assert windows.context_window("anthropic", "claude-haiku-4-5-20251001") == 200_000


def test_family_is_case_insensitive() -> None:
    assert windows.context_window("GEMINI", "gemini-2.0-flash") == 1_048_576


def test_null_and_unknown_are_both_none() -> None:
    assert windows.context_window("openai", "gpt-5.6-luna") is None, "explicit null"
    assert windows.context_window("gemini", "not-a-model") is None, "absent key"
    assert windows.context_window("nope", "nope") is None, "absent family"


def test_policy_is_the_configured_fraction_of_the_window() -> None:
    policy = windows.policy_for("gemini:gemini-2.0-flash")

    assert policy is not None
    assert policy.max_input_tokens == int(1_048_576 * 0.6)
    assert policy.summary_max_tokens == 1024
    assert policy.min_recent_turns == 2


def test_policy_ratio_is_overridable() -> None:
    policy = windows.policy_for("anthropic:claude-haiku-4-5", ratio=0.5)

    assert policy is not None
    assert policy.max_input_tokens == 100_000


def test_policy_accepts_a_bare_model_with_a_default_family() -> None:
    policy = windows.policy_for("gemini-2.0-flash", default_family="gemini")

    assert policy is not None
    assert policy.max_input_tokens == int(1_048_576 * 0.6)


def test_unknown_window_yields_no_policy() -> None:
    """No verified window means no budget, rather than a budget from a guess."""
    assert windows.policy_for("openai:gpt-5.6-luna") is None
    assert windows.policy_for("gemini:not-a-model") is None


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_impossible_ratios_are_refused(ratio: float) -> None:
    with pytest.raises(ValueError, match="ratio must be"):
        windows.policy_for("gemini:gemini-2.0-flash", ratio=ratio)


def test_malformed_spec_still_raises_from_the_parser() -> None:
    from bodol.providers import InvalidProviderSpecError

    with pytest.raises(InvalidProviderSpecError):
        windows.policy_for("bare-model-no-family")


def test_missing_table_file_is_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "CONTEXT_WINDOW_FILE", tmp_path / "absent.yaml")
    windows._window_table.cache_clear()

    assert windows.context_window("gemini", "gemini-2.0-flash") is None
    assert windows.policy_for("gemini:gemini-2.0-flash") is None


def test_non_integer_entries_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = tmp_path / "windows.yaml"
    table.write_text(
        "gemini:\n  a: null\n  b: 0\n  c: not-a-number\n  d: 128000\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "CONTEXT_WINDOW_FILE", table)
    windows._window_table.cache_clear()

    assert windows.context_window("gemini", "a") is None
    assert windows.context_window("gemini", "b") is None, "zero is not a usable window"
    assert windows.context_window("gemini", "c") is None
    assert windows.context_window("gemini", "d") == 128_000


def test_shipped_table_covers_every_priced_model() -> None:
    """A model worth pricing is a model worth budgeting; drift here is a bug."""
    import yaml

    pricing = yaml.safe_load(config.PRICING_FILE.read_text())
    shipped = yaml.safe_load(config.CONTEXT_WINDOW_FILE.read_text())

    missing = [
        f"{family}:{model}"
        for family, models in pricing.items()
        for model in (models or {})
        if model not in (shipped.get(family) or {})
    ]
    assert missing == [], f"models priced but not sized: {missing}"
