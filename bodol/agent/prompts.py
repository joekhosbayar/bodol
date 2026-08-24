"""Versioned system prompts, loaded from disk.

Prompts are files rather than string literals so a prompt change is a diff, and
so `bodol compare --models` can hold the model fixed and vary the prompt. The
version is the filename: `--system v1` reads `prompts/v1.md`.

`config.ensure_dirs()` deliberately does not create the prompt directory.
Traces and the scratchpad are runtime state the CLI may invent; prompts are
source, and a run whose prompt silently went missing should fail loudly rather
than proceed with an empty brief.
"""

from __future__ import annotations

from pathlib import Path

from bodol import config

SUFFIX = ".md"


class PromptError(ValueError):
    """Base class for prompt loading failures."""


class InvalidPromptNameError(PromptError):
    """The requested version is not a usable filename."""


class PromptNotFoundError(PromptError):
    """No prompt file exists for the requested version."""


def load(name: str, *, directory: Path | None = None) -> str:
    """Return the text of prompt version `name`.

    The name is validated rather than trusted: it arrives from `--system`, and
    joining unvalidated user input onto a path is how `v1` becomes
    `../../.env`.
    """
    version = name.strip()
    if not version:
        raise InvalidPromptNameError("prompt version cannot be empty")
    if version != Path(version).name or version in {".", ".."}:
        raise InvalidPromptNameError(
            f"prompt version {name!r} must be a bare name, not a path"
        )

    root = directory if directory is not None else config.PROMPT_DIR
    path = root / f"{version}{SUFFIX}"
    if not path.is_file():
        raise PromptNotFoundError(f"no prompt file at {path}")

    text = path.read_text(encoding="utf-8")
    if not text.strip():
        # An empty system prompt is indistinguishable from no system prompt at
        # the provider, which makes an accidentally-truncated file invisible.
        raise PromptError(f"prompt file {path} is empty")
    return text


def available(directory: Path | None = None) -> tuple[str, ...]:
    """Version names that `load` would accept, sorted. Empty when none exist."""
    root = directory if directory is not None else config.PROMPT_DIR
    if not root.is_dir():
        return ()
    return tuple(sorted(p.stem for p in root.glob(f"*{SUFFIX}") if p.is_file()))


__all__ = [
    "InvalidPromptNameError",
    "PromptError",
    "PromptNotFoundError",
    "available",
    "load",
]
