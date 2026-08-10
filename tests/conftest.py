"""Shared fixtures."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str) -> dict[str, Any]:
    """Read a captured provider response, e.g. load("toolcalls/gemini_weather_0")."""
    data: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    return data


@pytest.fixture
def tmp_trace_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point trace output at a temp dir so tests never touch traces/."""
    from bodol import config

    d = tmp_path / "traces"
    d.mkdir()
    monkeypatch.setattr(config, "TRACE_DIR", d)
    yield d
