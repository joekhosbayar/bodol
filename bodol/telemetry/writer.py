"""JSONL sinks.

One file per trace, at `traces/<trace_id>.jsonl` — the layout `bodol replay
traces/<id>.jsonl` and `bodol cost traces/` expect.

Records are flushed on every write. A run that dies at step 9 of 12 should still
have nine rows on disk; buffering them away is how you lose exactly the run you
most wanted to look at.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, runtime_checkable

from bodol import config


@runtime_checkable
class Sink(Protocol):
    """Where trace records go. Structural, so a test can pass a list-backed fake."""

    def emit(self, record: Mapping[str, Any]) -> None: ...

    def close(self) -> None: ...


class NullSink:
    """Tracing off. Used when the CLI is run with --no-trace."""

    def emit(self, record: Mapping[str, Any]) -> None:
        return None

    def close(self) -> None:
        return None


class MemorySink:
    """Collects records in a list. For tests and for `bodol run` summaries."""

    def __init__(self) -> None:
        self.records: list[Mapping[str, Any]] = []

    def emit(self, record: Mapping[str, Any]) -> None:
        self.records.append(record)

    def close(self) -> None:
        return None


class JsonlSink:
    """Append-only JSONL, one file per trace."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")

    @classmethod
    def for_trace(cls, trace_id: str, *, directory: Path | None = None) -> JsonlSink:
        return cls((directory or config.TRACE_DIR) / f"{trace_id}.jsonl")

    def emit(self, record: Mapping[str, Any]) -> None:
        # default=str so an unexpected non-serializable value degrades to a
        # string instead of killing the run it was supposed to be observing.
        self._fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> JsonlSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Load a trace file back. Used by `bodol replay` and `bodol cost`."""
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records
