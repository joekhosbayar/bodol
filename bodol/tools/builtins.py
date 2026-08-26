"""Built-in tools: calculator, file read, grep, file listing.

All four are read-only and confined to one root directory, `Path.cwd()` by
default. Every path a model supplies is resolved and checked against that root,
which rejects `../` traversal and symlinks pointing outward in the same test.

**The sandbox permits dotfiles, and that includes `.env`.** A model that asks
for it will be handed your API keys, and the provider you are handing them to
is the one that asked. This is a deliberate trade-off in favor of a tool that
can read `.gitignore` and `.github/workflows/*` without special-casing; it is
written down here so that whoever tightens it later knows it was a decision.
Confine the root or add a denylist if that trade is wrong for your use.

Handlers are synchronous on purpose: `ToolRegistry` rejects coroutines and runs
these in `asyncio.to_thread`, so blocking file IO does not stall the loop. They
raise `ValueError` for anything a model can fix, because the registry turns a
handler exception into an `is_error` result the model sees and can correct from.
"""

from __future__ import annotations

import ast
import fnmatch
import json
import operator
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from bodol.tools.registry import ToolRegistry

# Directories that are never worth walking: enormous, generated, or both. A
# model asking to grep a repository means its source, not 40k objects in .git.
SKIP_DIRS = frozenset({".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache",
                       ".pytest_cache", ".ruff_cache", "dist", "build", ".DS_Store"})

MAX_READ_BYTES = 64_000
MAX_GREP_MATCHES = 100
# Files above this are almost certainly data or binaries; scanning them line by
# line burns the step budget for nothing.
MAX_GREP_FILE_BYTES = 2_000_000
# The listing is bounded so a monorepo cannot flood the next model call, but
# the count in the response is always exact — a truncated list with a true
# count still answers "how many".
MAX_LIST_RESULTS = 200

# ---------------------------------------------------------------- calculator

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], object]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# `2 ** 3 ** 999999` is a few characters that pin a core for minutes. Bound the
# exponent rather than trusting the model to be reasonable.
MAX_EXPONENT = 1024


def _eval_node(node: ast.expr) -> float | int:
    """Evaluate one whitelisted arithmetic node. No names, no calls, no eval."""
    match node:
        case ast.Constant(value=bool()):
            # bool is an int subclass; arithmetic on True is a typo, not a sum.
            raise ValueError("booleans are not numbers")
        case ast.Constant(value=int() | float() as value):
            return value
        case ast.UnaryOp(op=ast.USub(), operand=operand):
            return -_eval_node(operand)
        case ast.UnaryOp(op=ast.UAdd(), operand=operand):
            return +_eval_node(operand)
        case ast.BinOp(left=left, op=op, right=right) if type(op) in _BIN_OPS:
            lhs = _eval_node(left)
            rhs = _eval_node(right)
            if isinstance(op, ast.Pow) and abs(rhs) > MAX_EXPONENT:
                raise ValueError(f"exponent {rhs} exceeds the limit of {MAX_EXPONENT}")
            try:
                computed = _BIN_OPS[type(op)](lhs, rhs)
            except ZeroDivisionError:
                raise ValueError("division by zero") from None
            except OverflowError:
                raise ValueError("result is too large to represent") from None
            # A negative base with a fractional exponent returns a complex
            # number, which is not an answer to an arithmetic question.
            if isinstance(computed, bool) or not isinstance(computed, int | float):
                raise ValueError(f"{ast.unparse(node)} is not a real number")
            return computed
        case _:
            raise ValueError(
                f"unsupported expression element {type(node).__name__}; "
                "only numbers and + - * / // % ** are allowed"
            )


def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression and return the result as text."""
    if not expression.strip():
        raise ValueError("expression cannot be empty")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse {expression!r}: {exc.msg}") from None
    return str(_eval_node(tree.body))


# ---------------------------------------------------------------- filesystem


def _resolved(root: Path, raw: str) -> Path:
    """Resolve `raw` against `root` and refuse anything that leaves it.

    Resolution happens before the containment check, so a symlink pointing
    outside the root fails the same test as `../../etc/passwd`.
    """
    if not raw.strip():
        raise ValueError("path cannot be empty")
    candidate = (root / raw).resolve() if not Path(raw).is_absolute() else Path(raw).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path {raw!r} is outside the allowed root {root}")
    return candidate


def _read_file(root: Path, path: str, max_bytes: int = MAX_READ_BYTES) -> str:
    target = _resolved(root, path)
    if target.is_dir():
        raise ValueError(f"{path!r} is a directory, not a file")
    if not target.is_file():
        raise ValueError(f"no such file: {path!r}")

    limit = max(1, min(int(max_bytes), MAX_READ_BYTES))
    raw = target.read_bytes()
    truncated = len(raw) > limit
    try:
        text = raw[:limit].decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{path!r} is not UTF-8 text") from None
    if truncated:
        text += f"\n[truncated at {limit} bytes of {len(raw)}]"
    return text


def _walk_files(start: Path, max_bytes: int | None = MAX_GREP_FILE_BYTES) -> Iterator[Path]:
    """Yield files under `start`, skipping generated and oversized ones.

    `max_bytes=None` lists everything: a file too large to *grep* line by line
    still counts when the question is how many files there are.
    """
    if start.is_file():
        yield start
        return
    for parent, dirnames, filenames in start.walk(on_error=lambda _: None):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in sorted(filenames):
            candidate = parent / name
            try:
                if candidate.is_file() and (
                    max_bytes is None or candidate.stat().st_size <= max_bytes
                ):
                    yield candidate
            except OSError:
                continue


def _list_files(
    root: Path,
    pattern: str,
    path: str = ".",
    max_results: int = MAX_LIST_RESULTS,
) -> str:
    """List files matching a glob, with an exact count and byte sizes.

    Returns JSON so `count` stays a number the model can trust instead of a
    sentence it has to parse. The list itself may be truncated; the count never
    is.
    """
    if not pattern.strip():
        raise ValueError("pattern cannot be empty")
    start = _resolved(root, path)
    if not start.exists():
        raise ValueError(f"no such path: {path!r}")

    limit = max(1, min(int(max_results), MAX_LIST_RESULTS))
    matched: list[tuple[str, int]] = []
    for file in _walk_files(start, max_bytes=None):
        rel = file.relative_to(root).as_posix() if root in file.parents else file.name
        # Matched against the full relative path and the bare name, so `*.py`
        # finds files at any depth and `bodol/*.py` still scopes. fnmatchcase,
        # because fnmatch lowercases on macOS and "*.PY" should not match.
        if fnmatch.fnmatchcase(rel, pattern) or fnmatch.fnmatchcase(file.name, pattern):
            try:
                matched.append((rel, file.stat().st_size))
            except OSError:
                continue
    matched.sort()

    listed = matched[:limit]
    return json.dumps(
        {
            "pattern": pattern,
            "path": path,
            "count": len(matched),
            "returned": len(listed),
            "truncated": len(matched) > len(listed),
            "files": [{"path": rel, "bytes": size} for rel, size in listed],
        }
    )


def _grep(
    root: Path,
    pattern: str,
    path: str = ".",
    max_matches: int = MAX_GREP_MATCHES,
) -> str:
    if not pattern:
        raise ValueError("pattern cannot be empty")
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"invalid regular expression {pattern!r}: {exc}") from None

    start = _resolved(root, path)
    if not start.exists():
        raise ValueError(f"no such path: {path!r}")

    limit = max(1, min(int(max_matches), MAX_GREP_MATCHES))
    hits: list[str] = []
    for file in _walk_files(start):
        try:
            content = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                rel = file.relative_to(root) if root in file.parents else file.name
                hits.append(f"{rel}:{lineno}:{line.strip()[:200]}")
                if len(hits) >= limit:
                    return "\n".join(hits) + f"\n[stopped at {limit} matches]"
    return "\n".join(hits) if hits else f"no matches for {pattern!r}"


# ---------------------------------------------------------------- registration

_CALCULATOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "Arithmetic only, e.g. '(1200 * 1.08) / 3'.",
        }
    },
    "required": ["expression"],
}

_FILE_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path relative to the working directory.",
        },
        "max_bytes": {
            "type": "integer",
            "description": f"Truncate after this many bytes. Max {MAX_READ_BYTES}.",
        },
    },
    "required": ["path"],
}

_GREP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {"type": "string", "description": "Python regular expression."},
        "path": {
            "type": "string",
            "description": "File or directory to search. Defaults to the whole tree.",
        },
        "max_matches": {
            "type": "integer",
            "description": f"Stop after this many hits. Max {MAX_GREP_MATCHES}.",
        },
    },
    "required": ["pattern"],
}

_LIST_FILES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": (
                "Filename glob, matched at any depth, e.g. '*.py' or 'test_*.py'. "
                "Use '*' to list every file."
            ),
        },
        "path": {
            "type": "string",
            "description": "Directory to list, relative to the working directory. "
            "Defaults to the whole tree.",
        },
        "max_results": {
            "type": "integer",
            "description": (
                f"Return at most this many files. Max {MAX_LIST_RESULTS}. "
                "The count in the response is exact even when the list is truncated."
            ),
        },
    },
    "required": ["pattern"],
}


def register_builtins(
    registry: ToolRegistry | None = None, *, root: Path | None = None
) -> ToolRegistry:
    """Register the read-only tool set and return the registry.

    `root` bounds every path these tools will touch; it defaults to the process
    working directory, which is what makes `bodol run` operate on the project
    the user invoked it from.
    """
    registry = registry if registry is not None else ToolRegistry()
    base = (root if root is not None else Path.cwd()).resolve()

    registry.register(
        name="calculator",
        description="Evaluate an arithmetic expression exactly. Use instead of mental math.",
        parameters=_CALCULATOR_SCHEMA,
        handler=calculate,
    )
    registry.register(
        name="file_read",
        description=(
            "Read a UTF-8 text file from the working directory tree. "
            "Returns the file contents, truncated if large."
        ),
        parameters=_FILE_READ_SCHEMA,
        handler=lambda path, max_bytes=MAX_READ_BYTES: _read_file(base, path, max_bytes),
    )
    registry.register(
        name="grep",
        description=(
            "Search file contents by regular expression. "
            "Returns 'path:line:text' hits. Use this to locate code before reading it."
        ),
        parameters=_GREP_SCHEMA,
        handler=lambda pattern, path=".", max_matches=MAX_GREP_MATCHES: _grep(
            base, pattern, path, max_matches
        ),
    )
    registry.register(
        name="list_files",
        description=(
            "List files by name pattern, with byte sizes and an exact count. "
            "Use this to see what exists before guessing a path, and to answer "
            "questions about how many files there are or which is largest."
        ),
        parameters=_LIST_FILES_SCHEMA,
        handler=lambda pattern, path=".", max_results=MAX_LIST_RESULTS: _list_files(
            base, pattern, path, max_results
        ),
    )
    return registry


__all__ = [
    "MAX_GREP_MATCHES",
    "MAX_LIST_RESULTS",
    "MAX_READ_BYTES",
    "calculate",
    "register_builtins",
]
