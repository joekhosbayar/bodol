"""Registration and dispatch for the agent's model-facing tools."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from bodol.providers.base import ToolCall, ToolResultBlock, ToolSpec


class ToolRegistryError(ValueError):
    """Base class for tool registration and lookup failures."""


class InvalidToolError(ToolRegistryError):
    """A tool definition is missing required metadata."""


class DuplicateToolError(ToolRegistryError):
    """A tool name is already registered."""


class UnknownToolError(ToolRegistryError):
    """A model requested a tool that is not registered."""


@dataclass(frozen=True, slots=True)
class RegisteredTool:
    """A model-facing declaration paired with its local implementation."""

    spec: ToolSpec
    handler: Callable[..., object]


class ToolRegistry:
    """Own tool declarations and execute synchronous handlers safely."""

    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., object],
    ) -> ToolSpec:
        """Register one explicit tool definition and return its model spec."""
        name = name.strip()
        if not name or not description.strip():
            raise InvalidToolError("tool name and description cannot be empty")
        if not parameters:
            raise InvalidToolError(f"tool {name!r} must provide a JSON Schema")
        if inspect.iscoroutinefunction(handler):
            raise InvalidToolError(f"tool {name!r} must use a synchronous callable")
        if name in self._tools:
            raise DuplicateToolError(f"tool {name!r} is already registered")

        spec = ToolSpec(name=name, description=description.strip(), parameters=dict(parameters))
        self._tools[name] = RegisteredTool(spec=spec, handler=handler)
        return spec

    @property
    def specs(self) -> tuple[ToolSpec, ...]:
        """Return declarations in registration order for provider calls."""
        return tuple(tool.spec for tool in self._tools.values())

    async def dispatch(self, call: ToolCall) -> ToolResultBlock:
        """Execute one call and return a result block for conversation history."""
        tool = self._tools.get(call.name)
        if tool is None:
            raise UnknownToolError(f"tool {call.name!r} is not registered")
        if call.args is None:
            raw = call.raw_args or "<missing>"
            return ToolResultBlock(
                call_id=call.id,
                name=call.name,
                content=f"Invalid arguments for tool {call.name!r}: {raw}",
                is_error=True,
            )

        try:
            result = await asyncio.to_thread(tool.handler, **call.args)
        except Exception as exc:
            return ToolResultBlock(
                call_id=call.id,
                name=call.name,
                content=f"Tool {call.name!r} failed: {exc}",
                is_error=True,
            )
        return ToolResultBlock(call_id=call.id, name=call.name, content=_render_result(result))

    async def dispatch_all(self, calls: Sequence[ToolCall]) -> tuple[ToolResultBlock, ...]:
        """Execute independent calls concurrently, preserving input order."""
        results = await asyncio.gather(*(self.dispatch(call) for call in calls))
        return tuple(results)


def _render_result(result: object) -> str:
    """Convert a handler result into the text content expected by providers."""
    if isinstance(result, str):
        return result
    return json.dumps(result, default=str)
