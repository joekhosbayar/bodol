"""Tests for the read-only built-in tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from bodol.tools.builtins import MAX_LIST_RESULTS, MAX_READ_BYTES, calculate, register_builtins
from bodol.tools.registry import ToolRegistry


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A small project tree with something to find and something to refuse."""
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text(
        "import os\n\n\ndef start():\n    return os.getcwd()\n", encoding="utf-8"
    )
    (tmp_path / "README.md").write_text("# demo\nstart the app\n", encoding="utf-8")
    (tmp_path / ".env").write_text("GEMINI_API_KEY=sk-test\n", encoding="utf-8")
    (tmp_path / "logo.bin").write_bytes(b"\x00\x01\x02\xff\xfe")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "COMMIT_EDITMSG").write_text("start the app\n", encoding="utf-8")
    return tmp_path


def _handler(registry: ToolRegistry, name: str):  # type: ignore[no-untyped-def]
    return next(t for n, t in registry._tools.items() if n == name).handler


# ---------------------------------------------------------------- calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 2", "4"),
        ("2 + 3 * 4", "14"),
        ("(2 + 3) * 4", "20"),
        ("7 / 2", "3.5"),
        ("7 // 2", "3"),
        ("7 % 2", "1"),
        ("2 ** 10", "1024"),
        ("-5 + 1", "-4"),
        ("+3", "3"),
        ("1200 * 1.08", "1296.0"),
    ],
)
def test_calculator_arithmetic(expression: str, expected: str) -> None:
    assert calculate(expression) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "os.getcwd()",
        "x + 1",
        "[1, 2, 3]",
        "{'a': 1}",
        "'a' * 3",
        "lambda: 1",
        "True + 1",
    ],
)
def test_calculator_refuses_everything_that_is_not_arithmetic(expression: str) -> None:
    """No eval anywhere in the path, so none of this is reachable."""
    with pytest.raises(ValueError):
        calculate(expression)


def test_calculator_bounds_the_exponent() -> None:
    with pytest.raises(ValueError, match="exponent"):
        calculate("2 ** 99999")


def test_calculator_reports_division_by_zero_as_an_error() -> None:
    with pytest.raises(ValueError, match="division by zero"):
        calculate("1 / 0")


def test_calculator_refuses_complex_results() -> None:
    with pytest.raises(ValueError, match="not a real number"):
        calculate("(-8) ** 0.5")


def test_calculator_rejects_empty_and_unparseable() -> None:
    with pytest.raises(ValueError, match="empty"):
        calculate("   ")
    with pytest.raises(ValueError, match="could not parse"):
        calculate("2 +")


# ---------------------------------------------------------------- file_read


def test_file_read_returns_contents(tree: Path) -> None:
    read = _handler(register_builtins(root=tree), "file_read")

    assert "def start()" in read(path="app/main.py")


def test_file_read_allows_dotfiles(tree: Path) -> None:
    """A documented trade-off: the sandbox is the tree, dotfiles included."""
    read = _handler(register_builtins(root=tree), "file_read")

    assert "GEMINI_API_KEY" in read(path=".env")


@pytest.mark.parametrize("path", ["../outside.txt", "../../etc/passwd", "/etc/hosts"])
def test_file_read_refuses_paths_outside_the_root(tree: Path, path: str) -> None:
    (tree.parent / "outside.txt").write_text("secret", encoding="utf-8")
    read = _handler(register_builtins(root=tree), "file_read")

    with pytest.raises(ValueError, match="outside the allowed root"):
        read(path=path)


def test_file_read_refuses_a_symlink_pointing_out(tree: Path) -> None:
    """Resolution happens before the containment check, so this is the same test."""
    (tree.parent / "outside.txt").write_text("secret", encoding="utf-8")
    (tree / "escape.txt").symlink_to(tree.parent / "outside.txt")
    read = _handler(register_builtins(root=tree), "file_read")

    with pytest.raises(ValueError, match="outside the allowed root"):
        read(path="escape.txt")


def test_file_read_refuses_directories_and_missing_files(tree: Path) -> None:
    read = _handler(register_builtins(root=tree), "file_read")

    with pytest.raises(ValueError, match="is a directory"):
        read(path="app")
    with pytest.raises(ValueError, match="no such file"):
        read(path="app/nope.py")


def test_file_read_refuses_binary(tree: Path) -> None:
    read = _handler(register_builtins(root=tree), "file_read")

    with pytest.raises(ValueError, match="not UTF-8"):
        read(path="logo.bin")


def test_file_read_marks_truncation(tree: Path) -> None:
    (tree / "long.txt").write_text("x" * 500, encoding="utf-8")
    read = _handler(register_builtins(root=tree), "file_read")

    out = read(path="long.txt", max_bytes=100)
    assert out.startswith("x" * 100)
    assert "[truncated at 100 bytes of 500]" in out


def test_file_read_caps_max_bytes_at_the_module_limit(tree: Path) -> None:
    (tree / "big.txt").write_text("y" * (MAX_READ_BYTES + 50), encoding="utf-8")
    read = _handler(register_builtins(root=tree), "file_read")

    out = read(path="big.txt", max_bytes=10_000_000)
    assert f"[truncated at {MAX_READ_BYTES} bytes" in out


# ---------------------------------------------------------------- grep


def test_grep_reports_path_line_and_text(tree: Path) -> None:
    grep = _handler(register_builtins(root=tree), "grep")

    hits = grep(pattern=r"def start").splitlines()
    assert hits == ["app/main.py:4:def start():"]


def test_grep_skips_generated_directories(tree: Path) -> None:
    grep = _handler(register_builtins(root=tree), "grep")

    hits = grep(pattern="start the app")
    assert "README.md" in hits
    assert ".git" not in hits, "walking .git returns noise, not answers"


def test_grep_scopes_to_a_subtree(tree: Path) -> None:
    grep = _handler(register_builtins(root=tree), "grep")

    assert "no matches" in grep(pattern="demo", path="app")
    assert "README.md" in grep(pattern="demo", path=".")


def test_grep_stops_at_max_matches(tree: Path) -> None:
    (tree / "many.txt").write_text("hit\n" * 20, encoding="utf-8")
    grep = _handler(register_builtins(root=tree), "grep")

    out = grep(pattern="hit", max_matches=3)
    assert len(out.splitlines()) == 4, "three hits plus the stopped-at marker"
    assert "[stopped at 3 matches]" in out


def test_grep_skips_binary_without_failing(tree: Path) -> None:
    grep = _handler(register_builtins(root=tree), "grep")

    assert "no matches" in grep(pattern=r"\xff")


def test_grep_reports_a_bad_regex(tree: Path) -> None:
    grep = _handler(register_builtins(root=tree), "grep")

    with pytest.raises(ValueError, match="invalid regular expression"):
        grep(pattern="(unclosed")


def test_grep_refuses_paths_outside_the_root(tree: Path) -> None:
    grep = _handler(register_builtins(root=tree), "grep")

    with pytest.raises(ValueError, match="outside the allowed root"):
        grep(pattern="x", path="..")


# ---------------------------------------------------------------- list_files


def _listed(tree: Path, **kwargs: Any) -> Any:
    list_files = _handler(register_builtins(root=tree), "list_files")
    return json.loads(list_files(**kwargs))


def test_list_files_matches_a_pattern_at_any_depth(tree: Path) -> None:
    """The canonical question: how many Python files, without guessing a layout."""
    (tree / "app" / "util.py").write_text("def helper():\n    return 1\n", encoding="utf-8")

    out = _listed(tree, pattern="*.py")

    assert out["count"] == 2
    assert out["truncated"] is False
    by_path = {f["path"]: f["bytes"] for f in out["files"]}
    assert set(by_path) == {"app/main.py", "app/util.py"}
    assert by_path["app/main.py"] == len("import os\n\n\ndef start():\n    return os.getcwd()\n")


def test_list_files_skips_generated_directories(tree: Path) -> None:
    out = _listed(tree, pattern="*")

    paths = [f["path"] for f in out["files"]]
    assert ".git/COMMIT_EDITMSG" not in paths
    assert out["count"] == len(paths), "the count never covers what the walker skipped"


def test_list_files_scopes_to_a_subtree(tree: Path) -> None:
    out = _listed(tree, pattern="*.py", path="app")

    assert out["count"] == 1
    assert out["files"][0]["path"] == "app/main.py", "paths stay relative to the root"


def test_list_files_matches_a_prefixed_pattern(tree: Path) -> None:
    out = _listed(tree, pattern="app/*.py")

    assert out["count"] == 1


def test_list_files_reports_zero_matches_as_data_not_an_error(tree: Path) -> None:
    out = _listed(tree, pattern="*.rs")

    assert out["count"] == 0
    assert out["files"] == []


def test_list_files_count_stays_exact_when_the_list_is_truncated(tree: Path) -> None:
    """A monorepo must not flood the next call, but the answer must stay true."""
    for i in range(5):
        (tree / "app" / f"mod_{i}.py").write_text(f"# {i}\n", encoding="utf-8")

    out = _listed(tree, pattern="*.py", max_results=2)

    assert out["count"] == 6
    assert out["returned"] == 2
    assert out["truncated"] is True


def test_list_files_caps_max_results(tree: Path) -> None:
    out = _listed(tree, pattern="*", max_results=10_000_000)

    assert out["count"] <= MAX_LIST_RESULTS or out["returned"] <= MAX_LIST_RESULTS


def test_list_files_includes_oversized_files_in_the_count(tree: Path) -> None:
    """A file too large to grep line by line still counts as a file."""
    big = tree / "app" / "generated.py"
    big.write_text("x" * 3_000_000, encoding="utf-8")

    out = _listed(tree, pattern="*.py")

    assert out["count"] == 2
    assert next(f for f in out["files"] if f["path"] == "app/generated.py")["bytes"] == 3_000_000


def test_list_files_refuses_paths_outside_the_root(tree: Path) -> None:
    list_files = _handler(register_builtins(root=tree), "list_files")

    with pytest.raises(ValueError, match="outside the allowed root"):
        list_files(pattern="*", path="..")


def test_list_files_refuses_a_symlinked_start_pointing_out(tree: Path) -> None:
    (tree.parent / "outside").mkdir(exist_ok=True)
    (tree / "escape").symlink_to(tree.parent / "outside", target_is_directory=True)
    list_files = _handler(register_builtins(root=tree), "list_files")

    with pytest.raises(ValueError, match="outside the allowed root"):
        list_files(pattern="*", path="escape")


def test_list_files_reports_missing_paths_and_empty_patterns(tree: Path) -> None:
    list_files = _handler(register_builtins(root=tree), "list_files")

    with pytest.raises(ValueError, match="no such path"):
        list_files(pattern="*", path="nope")
    with pytest.raises(ValueError, match="cannot be empty"):
        list_files(pattern="  ")


# ---------------------------------------------------------------- registration


def test_register_builtins_declares_four_usable_tools(tree: Path) -> None:
    registry = register_builtins(root=tree)

    assert [spec.name for spec in registry.specs] == [
        "calculator",
        "file_read",
        "grep",
        "list_files",
    ]
    for spec in registry.specs:
        assert spec.description.strip()
        assert spec.parameters["type"] == "object"
        assert spec.parameters["required"], "every tool has at least one required argument"


async def test_registered_tools_dispatch_through_the_registry(tree: Path) -> None:
    """End to end through the machinery the agent loop actually uses."""
    from bodol.providers.base import ToolCall

    registry = register_builtins(root=tree)

    ok, bad = await registry.dispatch_all(
        [
            ToolCall(id="a", name="calculator", args={"expression": "6 * 7"}),
            ToolCall(id="b", name="file_read", args={"path": "../escape"}),
        ]
    )

    assert (ok.content, ok.is_error) == ("42", False)
    assert bad.is_error and "outside the allowed root" in bad.content
