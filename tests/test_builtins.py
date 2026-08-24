"""Tests for the read-only built-in tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from bodol.tools.builtins import MAX_READ_BYTES, calculate, register_builtins
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


# ---------------------------------------------------------------- registration


def test_register_builtins_declares_three_usable_tools(tree: Path) -> None:
    registry = register_builtins(root=tree)

    assert [spec.name for spec in registry.specs] == ["calculator", "file_read", "grep"]
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
