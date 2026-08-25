"""Tests for versioned system prompt loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from bodol import config
from bodol.agent import prompts


def test_ships_a_v1_prompt() -> None:
    """The default --system value must resolve against the repo as shipped."""
    text = prompts.load("v1")
    assert text.strip()
    assert "v1" in prompts.available()


def test_load_reads_the_named_version(tmp_path: Path) -> None:
    (tmp_path / "v2.md").write_text("be terse", encoding="utf-8")

    assert prompts.load("v2", directory=tmp_path) == "be terse"
    assert prompts.load("  v2  ", directory=tmp_path) == "be terse", "names are trimmed"


def test_missing_version_names_the_path_it_looked_for(tmp_path: Path) -> None:
    with pytest.raises(prompts.PromptNotFoundError, match=r"v9\.md"):
        prompts.load("v9", directory=tmp_path)


def test_missing_prompt_directory_is_not_a_crash(tmp_path: Path) -> None:
    absent = tmp_path / "nope"

    with pytest.raises(prompts.PromptNotFoundError):
        prompts.load("v1", directory=absent)
    assert prompts.available(absent) == (), "no directory means no versions, not an error"


@pytest.mark.parametrize("name", ["../.env", "sub/v1", "/etc/passwd", "..", ""])
def test_path_like_versions_are_refused(name: str, tmp_path: Path) -> None:
    """--system is user input being turned into a path, so it is validated."""
    with pytest.raises(prompts.InvalidPromptNameError):
        prompts.load(name, directory=tmp_path)


def test_traversal_cannot_reach_a_real_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "secret.md"
    secret.write_text("API_KEY=hunter2", encoding="utf-8")
    promptdir = tmp_path / "prompts"
    promptdir.mkdir()
    monkeypatch.setattr(config, "PROMPT_DIR", promptdir)

    with pytest.raises(prompts.InvalidPromptNameError):
        prompts.load("../secret")


def test_empty_prompt_file_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "blank.md").write_text("   \n\n", encoding="utf-8")

    with pytest.raises(prompts.PromptError, match="empty"):
        prompts.load("blank", directory=tmp_path)


def test_available_is_sorted_and_stem_only(tmp_path: Path) -> None:
    for name in ("v3.md", "v1.md", "v2.md", "notes.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    assert prompts.available(tmp_path) == ("v1", "v2", "v3"), "only .md, sorted, no suffix"
