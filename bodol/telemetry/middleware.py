"""The provider-shaped wrapper that emits one record per call.

    provider = TracedProvider(GeminiAdapter("gemini-3.6-flash"), sink)

`TracedProvider` satisfies the same `Provider` protocol as the thing it wraps, so
nothing downstream knows it is there. Composition rather than a mixin or a
decorator inside each adapter: telemetry is written once and works for every
provider added later, including ones that do not exist yet.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from bodol.providers.base import Message, ModelResponse, ToolSpec
from bodol.telemetry import events
from bodol.telemetry.writer import Sink


class TracedProvider:
    """Wraps a Provider and emits a trace record for every call, success or not."""

    def __init__(self, inner: object, sink: Sink) -> None:
        # `inner` is typed loosely on purpose: Provider is a runtime-checkable
        # Protocol, and requiring it here would force adapters to import it.
        self._inner = inner
        self._sink = sink

    @property
    def name(self) -> str:
        return str(getattr(self._inner, "name", "unknown"))

    @property
    def model(self) -> str:
        return str(getattr(self._inner, "model", "unknown"))

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse:
        started = time.perf_counter()
        try:
            response: ModelResponse = await self._inner.generate(  # type: ignore[attr-defined]
                messages, system=system, tools=tools, max_tokens=max_tokens
            )
        except BaseException as exc:
            # A failed call still consumed wall-clock time, and may still have
            # been billed. Record it rather than losing the step entirely.
            elapsed = (time.perf_counter() - started) * 1000
            self._sink.emit(events.error_record(self.name, self.model, exc, latency_ms=elapsed))
            raise

        self._sink.emit(events.call_record(response))
        return response

    async def aclose(self) -> None:
        try:
            await self._inner.aclose()  # type: ignore[attr-defined]
        finally:
            self._sink.close()
