"""Shared fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point trace output at a temp dir so tests never touch traces/."""
    from bodol import config

    d = tmp_path / "traces"
    d.mkdir()
    monkeypatch.setattr(config, "TRACE_DIR", d)
    yield d
