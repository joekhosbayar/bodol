"""Resolve provider specifications and construct traced adapters."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import cast

import httpx

from bodol import config
from bodol.providers import anthropic, gemini, openai
from bodol.providers.base import Provider
from bodol.providers.http import DEFAULT_RETRY, Attempt, RetryPolicy
from bodol.telemetry import events
from bodol.telemetry.middleware import TracedProvider
from bodol.telemetry.writer import JsonlSink, Sink


class ProviderRegistryError(ValueError):
    """Base class for provider specification and construction failures."""


class InvalidProviderSpecError(ProviderRegistryError):
    """The provider specification is missing or malformed."""


class UnsupportedProviderError(ProviderRegistryError):
    """The specification names a provider family without an adapter."""


class NoActiveTraceError(ProviderRegistryError):
    """A provider was requested outside an active telemetry trace."""


class ProviderCredentialError(ProviderRegistryError):
    """The selected provider has no usable credential."""


def parse_provider_spec(
    value: str,
    *,
    default_family: str | None = None,
) -> tuple[str, str]:
    """Parse ``family:model`` or a model with an explicit default family."""
    spec = value.strip()
    if not spec:
        raise InvalidProviderSpecError("provider specification cannot be empty")

    if ":" in spec:
        family, model = spec.split(":", 1)
    else:
        family, model = default_family or "", spec

    family = family.strip().lower()
    model = model.strip()
    if not family:
        raise InvalidProviderSpecError(
            "bare model IDs require an explicit default_family"
        )
    if not model:
        raise InvalidProviderSpecError(
            f"provider specification {value!r} is missing a model ID"
        )
    return family, model


def _make_sink(trace_id: str) -> Sink:
    """Create the production sink for a provider's active trace."""
    return JsonlSink.for_trace(trace_id)


def _retry_recorder(sink: Sink, provider: str, model: str) -> Callable[[Attempt], None]:
    """Put every failed HTTP attempt in the trace.

    This is the one place that can: `TracedProvider` wraps `generate()` and only
    ever sees the outcome, while the attempts happen a layer below it inside
    `post_json`. The HTTP layer holds no sink, so the sink is handed to it here
    as a callback on the retry policy.
    """

    def record(attempt: Attempt) -> None:
        sink.emit(
            events.retry_record(
                provider,
                model,
                attempt=attempt.number,
                of=attempt.of,
                status=attempt.status,
                latency_ms=attempt.latency_ms,
                delay_s=attempt.delay_s,
                detail=attempt.detail,
            )
        )

    return record


def create_provider(
    value: str,
    *,
    default_family: str | None = None,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
    cache: bool = True,
) -> Provider:
    """Construct the selected adapter and atomically wrap it for telemetry.

    `cache` asks the vendor to reuse the prompt prefix. On by default, matching
    what OpenAI and Gemini do whether asked or not; Anthropic caches only when
    asked, and Gemini cannot be asked to stop.
    """
    family, model = parse_provider_spec(value, default_family=default_family)
    trace_id = events.current_trace_id()
    if trace_id is None:
        raise NoActiveTraceError(
            "provider creation requires an active trace; use events.start_trace() first"
        )

    adapters: dict[str, Callable[..., Provider]] = {
        "gemini": gemini.GeminiAdapter,
        "openai": openai.OpenAIAdapter,
        "anthropic": anthropic.AnthropicAdapter,
    }
    adapter_type = adapters.get(family)
    if adapter_type is None:
        supported = ", ".join(sorted(adapters))
        raise UnsupportedProviderError(
            f"unsupported provider family {family!r}; choose one of: {supported}"
        )

    sink = _make_sink(trace_id)
    if retry.on_attempt is None:
        retry = dataclasses.replace(retry, on_attempt=_retry_recorder(sink, family, model))
    try:
        adapter = adapter_type(model, api_key=api_key, client=client, retry=retry, cache=cache)
    except config.MissingCredential as exc:
        sink.close()
        raise ProviderCredentialError(
            f"no credential available for provider family {family!r}"
        ) from exc

    return cast(Provider, TracedProvider(adapter, sink))
