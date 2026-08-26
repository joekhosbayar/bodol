"""ContextPolicy and the manager that enforces it.

Token accounting comes from the provider, not a tokenizer: every
`ModelResponse` reports `usage.input_tokens` for the transcript actually sent,
so `observe()` after each call is the ground truth for how large the context
currently is. When the last observed input exceeds the budget, `compact()`
replaces the middle of the transcript with a model-written summary.

Two invariants survive every compaction:

- The system prompt is never in `messages` at all (adapters take it
  separately), so it cannot be evicted.
- The initial user task is always kept, so a compacted run cannot drift away
  from what it was asked to do.

The cut point never splits a tool-use/tool-result pair: providers reject a
history that ends an assistant tool_use turn without its results (or starts
with orphaned results), so the boundary walks to a safe position before
anything is evicted.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from bodol.providers.base import (
    Message,
    ModelResponse,
    Provider,
    TextBlock,
    ThoughtBlock,
    ToolResultBlock,
    ToolUseBlock,
)


class ContextManagerError(RuntimeError):
    """Base class for context management failures."""


class CompactionError(ContextManagerError):
    """The transcript could not be compacted."""


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Budget and compaction knobs.

    `max_input_tokens` is compared against the last observed
    `Usage.input_tokens`. `min_recent_turns` keeps the tail of the
    conversation live so the model always sees its latest exchange.
    """

    max_input_tokens: int
    summary_max_tokens: int = 1024
    min_recent_turns: int = 2


_SUMMARY_INSTRUCTION = """\
Summarize the following agent transcript prefix for continuation. Preserve:
the goal, decisions made, tool calls and their results, and any outstanding
work. Be terse; this replaces the evicted history verbatim.
"""


def _render_turn(message: Message) -> str:
    """Plain-text rendering of one turn for the summarization prompt."""
    parts: list[str] = []
    for block in message.content:
        match block:
            case TextBlock(text=text):
                parts.append(text)
            case ToolUseBlock(name=name, args=args):
                parts.append(f"[tool_use {name} args={args}]")
            case ToolResultBlock(content=content, is_error=is_error):
                tag = "tool_error" if is_error else "tool_result"
                parts.append(f"[{tag} {content}]")
            case ThoughtBlock():
                # Nothing to summarize: the block holds a signature, no text.
                # It is also dropped from the compacted transcript, which is
                # correct — a signature outlives neither the turn it signs nor
                # the tool call it was paired with.
                continue
    return f"{message.role}: " + "\n".join(parts)


def _starts_with_results(message: Message) -> bool:
    return any(isinstance(block, ToolResultBlock) for block in message.content)


def _ends_with_tool_use(message: Message) -> bool:
    return any(isinstance(block, ToolUseBlock) for block in message.content)


class ContextManager:
    """Tracks context size from usage feedback and compacts on demand."""

    def __init__(self, policy: ContextPolicy) -> None:
        self.policy = policy
        self._last_input_tokens: int | None = None

    @property
    def last_input_tokens(self) -> int | None:
        """Input size of the most recent model call, or None before the first."""
        return self._last_input_tokens

    def observe(self, response: ModelResponse) -> None:
        """Record the input size of the call that produced `response`."""
        self._last_input_tokens = response.usage.input_tokens

    def over_budget(self) -> bool:
        """True once the last observed input exceeded the budget."""
        return (
            self._last_input_tokens is not None
            and self._last_input_tokens > self.policy.max_input_tokens
        )

    async def compact(
        self,
        messages: Sequence[Message],
        *,
        provider: Provider,
        system: str | None = None,
    ) -> list[Message]:
        """Replace the middle of the transcript with a model-written summary.

        Returns a new list; the caller owns swapping it in. Raises
        CompactionError when there is nothing safe to evict or the
        summarization call fails.
        """
        if len(messages) < 2:
            raise CompactionError("nothing to compact: transcript is too short")

        cut = self._cut_point(messages)
        if cut <= 1:
            raise CompactionError(
                "nothing to compact: all turns are protected by the policy"
            )

        evicted = messages[1:cut]
        transcript = "\n\n".join(_render_turn(m) for m in evicted)
        prompt = Message(
            role="user",
            content=(TextBlock(text=_SUMMARY_INSTRUCTION + "\n\n" + transcript),),
        )
        try:
            response = await provider.generate(
                [prompt], system=system, tools=(), max_tokens=self.policy.summary_max_tokens
            )
        except Exception as exc:
            raise CompactionError(f"summarization call failed: {exc}") from exc

        if not response.text:
            raise CompactionError("summarization returned no text")

        summary = Message(
            role="user",
            content=(
                TextBlock(
                    text="[Summary of earlier conversation]\n" + response.text
                ),
            ),
        )
        return [messages[0], summary, *messages[cut:]]

    def _cut_point(self, messages: Sequence[Message]) -> int:
        """Index splitting [evicted | kept]. Never splits a tool pair.

        Index 0 (the initial task) is always kept, so eviction starts at 1.
        The default cut leaves `min_recent_turns` at the tail; the boundary
        then walks left while it would orphan tool results in the kept region
        or strand a tool_use in the evicted region.
        """
        cut = max(1, len(messages) - self.policy.min_recent_turns)
        while cut > 1 and (
            _starts_with_results(messages[cut]) or _ends_with_tool_use(messages[cut - 1])
        ):
            cut -= 1
        return cut
