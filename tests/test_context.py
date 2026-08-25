"""Tests for the usage-feedback context manager."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from bodol.context import (
    CompactionError,
    ContextManager,
    ContextPolicy,
)
from bodol.providers.base import (
    FinishReason,
    Message,
    ModelResponse,
    TextBlock,
    ThoughtBlock,
    ToolResultBlock,
    ToolSpec,
    ToolUseBlock,
    Usage,
)


def _response(input_tokens: int = 100, text: str | None = "ok") -> ModelResponse:
    return ModelResponse(
        id="resp",
        model="fake-model",
        provider="fake",
        finish_reason=FinishReason.STOP,
        usage=Usage(input_tokens=input_tokens, output_tokens=10),
        latency_ms=1.0,
        text=text,
    )


def _text(role: str, text: str) -> Message:
    return Message(role=role, content=(TextBlock(text=text),))  # type: ignore[arg-type]


def _tool_turn() -> list[Message]:
    """An assistant tool_use turn followed by its results — an atomic pair."""
    return [
        Message(
            role="assistant",
            content=(ToolUseBlock(id="t1", name="get_weather", args={"city": "Boston"}),),
        ),
        Message(
            role="user",
            content=(ToolResultBlock(call_id="t1", name="get_weather", content="sunny"),),
        ),
    ]


class FakeProvider:
    """Captures the summarization request and returns a canned summary."""

    name = "fake"
    model = "fake-model"

    def __init__(self, *, text: str | None = "summary text", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls: list[tuple[list[Message], str | None, int]] = []

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse:
        self.calls.append((list(messages), system, max_tokens))
        if self.fail:
            raise RuntimeError("boom")
        return _response(text=self.text)

    async def aclose(self) -> None:
        return None


# ---------------------------------------------------------------- observation


def test_over_budget_tracks_last_observed_input() -> None:
    manager = ContextManager(ContextPolicy(max_input_tokens=500))

    assert not manager.over_budget(), "no observation yet means not over budget"

    manager.observe(_response(input_tokens=500))
    assert not manager.over_budget(), "at budget is not over budget"

    manager.observe(_response(input_tokens=501))
    assert manager.over_budget()

    manager.observe(_response(input_tokens=10))
    assert not manager.over_budget(), "a smaller later observation clears it"


# ---------------------------------------------------------------- compaction


async def test_compact_keeps_task_and_recent_turns() -> None:
    provider = FakeProvider()
    manager = ContextManager(ContextPolicy(max_input_tokens=100, min_recent_turns=2))
    messages = [
        _text("user", "the original task"),
        _text("assistant", "old reply 1"),
        _text("user", "old followup"),
        _text("assistant", "recent reply"),
        _text("user", "latest question"),
    ]

    result = await manager.compact(messages, provider=provider, system="be terse")

    assert result[0] is messages[0], "initial task is always first"
    assert result[1].role == "user"
    (summary_block,) = result[1].content
    assert isinstance(summary_block, TextBlock)
    assert "summary text" in summary_block.text
    assert result[2:] == messages[-2:], "recent turns survive untouched"

    (prompt_messages, system, max_tokens) = provider.calls[0]
    assert system == "be terse"
    assert max_tokens == 1024
    (prompt_block,) = prompt_messages[0].content
    assert isinstance(prompt_block, TextBlock)
    assert "old reply 1" in prompt_block.text
    assert "old followup" in prompt_block.text
    assert "the original task" not in prompt_block.text, "task is kept, not summarized"
    assert "recent reply" not in prompt_block.text, "recent turns are kept"


async def test_signatures_are_not_summarized() -> None:
    """A signature is conversation state, not content.

    It has nothing a summary could carry, and putting it in the prompt would
    spend tokens asking the model to describe a base64 blob it cannot read.
    """
    provider = FakeProvider()
    manager = ContextManager(ContextPolicy(max_input_tokens=100, min_recent_turns=2))
    messages = [
        _text("user", "task"),
        Message(
            role="assistant",
            content=(ThoughtBlock("EjQKMgERTTIPVtJXOu"), TextBlock(text="early reply")),
        ),
        _text("user", "old followup"),
        _text("assistant", "recent reply"),
        _text("user", "latest"),
    ]

    await manager.compact(messages, provider=provider)

    (prompt_messages, _, _) = provider.calls[0]
    (prompt_block,) = prompt_messages[0].content
    assert isinstance(prompt_block, TextBlock)
    assert "early reply" in prompt_block.text
    assert "EjQKMgERTTIPVtJXOu" not in prompt_block.text


async def test_compact_never_splits_tool_pair() -> None:
    provider = FakeProvider()
    manager = ContextManager(ContextPolicy(max_input_tokens=100, min_recent_turns=2))
    messages = [
        _text("user", "task"),
        _text("assistant", "early reply"),
        *_tool_turn(),
        _text("user", "latest"),
    ]
    # Default cut (len - 2 = 3) would orphan the tool results in the kept
    # tail while their tool_use is evicted. The boundary must walk left so
    # the pair survives intact on one side of the cut.
    result = await manager.compact(messages, provider=provider)

    assert result[2:] == messages[2:], "tool_use, its results, and latest turn kept together"
    (prompt_messages, _, _) = provider.calls[0]
    (prompt_block,) = prompt_messages[0].content
    assert isinstance(prompt_block, TextBlock)
    assert "early reply" in prompt_block.text
    assert "tool_use" not in prompt_block.text, "kept pair is not summarized"


async def test_compact_never_strands_tool_use_in_evicted_region() -> None:
    provider = FakeProvider()
    manager = ContextManager(ContextPolicy(max_input_tokens=100, min_recent_turns=1))
    messages = [
        _text("user", "task"),
        _text("assistant", "old"),
        *_tool_turn(),
    ]
    # Default cut (len - 1 = 3) would evict the tool_use while its results
    # stay in the kept tail. The boundary must walk left past the pair.
    result = await manager.compact(messages, provider=provider)

    assert result[2:] == messages[2:], "the whole pair is kept, not split"
    (prompt_messages, _, _) = provider.calls[0]
    (prompt_block,) = prompt_messages[0].content
    assert isinstance(prompt_block, TextBlock)
    assert "old" in prompt_block.text
    assert "tool_use" not in prompt_block.text


async def test_compact_raises_when_nothing_evictable() -> None:
    manager = ContextManager(ContextPolicy(max_input_tokens=100, min_recent_turns=3))
    messages = [_text("user", "task"), _text("assistant", "a"), _text("user", "b")]

    with pytest.raises(CompactionError, match="protected"):
        await manager.compact(messages, provider=FakeProvider())


async def test_compact_raises_on_short_transcript() -> None:
    manager = ContextManager(ContextPolicy(max_input_tokens=100))
    with pytest.raises(CompactionError, match="too short"):
        await manager.compact([_text("user", "task")], provider=FakeProvider())


async def test_compact_wraps_provider_failure() -> None:
    manager = ContextManager(ContextPolicy(max_input_tokens=100, min_recent_turns=1))
    messages = [_text("user", "task"), _text("assistant", "old"), _text("user", "new")]

    with pytest.raises(CompactionError, match="summarization call failed"):
        await manager.compact(messages, provider=FakeProvider(fail=True))


async def test_compact_raises_on_empty_summary() -> None:
    manager = ContextManager(ContextPolicy(max_input_tokens=100, min_recent_turns=1))
    messages = [_text("user", "task"), _text("assistant", "old"), _text("user", "new")]

    with pytest.raises(CompactionError, match="no text"):
        await manager.compact(messages, provider=FakeProvider(text=None))
