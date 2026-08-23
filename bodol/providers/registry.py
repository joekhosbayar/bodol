"""Resolve provider specifications and construct traced adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import httpx

from bodol import config
from bodol.providers import anthropic, gemini, openai
from bodol.providers.base import Provider
from bodol.providers.http import DEFAULT_RETRY, RetryPolicy
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


def create_provider(
    value: str,
    *,
    default_family: str | None = None,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
    retry: RetryPolicy = DEFAULT_RETRY,
) -> Provider:
    """Construct the selected adapter and atomically wrap it for telemetry."""
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
    try:
        adapter = adapter_type(model, api_key=api_key, client=client, retry=retry)
    except config.MissingCredential as exc:
        sink.close()
        raise ProviderCredentialError(
            f"no credential available for provider family {family!r}"
        ) from exc

    return cast(Provider, TracedProvider(adapter, sink))
