from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from bodol.providers import (
    InvalidProviderSpecError,
    NoActiveTraceError,
    ProviderCredentialError,
    UnsupportedProviderError,
    create_provider,
    gemini,
    parse_provider_spec,
)
from bodol.providers.base import FinishReason, Message, ModelResponse, ToolSpec, Usage
from bodol.providers.http import RetryPolicy
from bodol.telemetry import events
from bodol.telemetry.middleware import TracedProvider
from bodol.telemetry.writer import MemorySink
from tests.conftest import load


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        ("gemini:gemini-3.6-flash", None, ("gemini", "gemini-3.6-flash")),
        (" OPENAI:gpt-5.6-luna ", None, ("openai", "gpt-5.6-luna")),
        ("claude-haiku-4-5", "anthropic", ("anthropic", "claude-haiku-4-5")),
        ("openai:model:variant", None, ("openai", "model:variant")),
    ],
)
def test_parse_provider_spec(
    value: str,
    default: str | None,
    expected: tuple[str, str],
) -> None:
    assert parse_provider_spec(value, default_family=default) == expected


@pytest.mark.parametrize("value", ["", " :model", "family:", "model"])
def test_parse_provider_spec_rejects_invalid_values(value: str) -> None:
    with pytest.raises(InvalidProviderSpecError):
        parse_provider_spec(value)


def test_parse_provider_spec_rejects_unknown_family() -> None:
    with pytest.raises(UnsupportedProviderError), events.start_trace("tr_registry_unknown"):
        create_provider("local:model", api_key="key")


def test_create_provider_requires_active_trace() -> None:
    with pytest.raises(NoActiveTraceError):
        create_provider("gemini:model", api_key="key")


@pytest.mark.parametrize("spec", ["gemini:model", "openai:model", "anthropic:model"])
async def test_create_provider_always_returns_traced_provider(
    spec: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = MemorySink()
    monkeypatch.setattr("bodol.providers.registry._make_sink", lambda _: sink)
    client = httpx.AsyncClient()

    with events.start_trace("tr_registry_factory"):
        provider = create_provider(spec, api_key="key", client=client)
        await provider.aclose()

    assert isinstance(provider, TracedProvider)
    assert provider.name == spec.split(":", 1)[0]
    assert provider.model == "model"


def test_create_provider_wraps_missing_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_trace_dir: Path,
) -> None:
    # The sink is created before credentials are checked, so without a
    # redirected trace dir this test writes a stray file into traces/.
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    with events.start_trace("tr_registry_credentials"), pytest.raises(
        ProviderCredentialError
    ) as error:
        create_provider("gemini:model")

    assert "gemini" in str(error.value)


class _FakeProvider:
    name = "gemini"
    model = "model"

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse:
        return ModelResponse(
            id="resp",
            model=self.model,
            provider=self.name,
            finish_reason=FinishReason.STOP,
            usage=Usage(),
            latency_ms=1,
        )

    async def aclose(self) -> None:
        return None


async def test_factory_sink_receives_telemetry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = MemorySink()
    monkeypatch.setattr("bodol.providers.registry._make_sink", lambda _: sink)
    monkeypatch.setattr(
        "bodol.providers.registry.gemini.GeminiAdapter",
        lambda *args, **kwargs: _FakeProvider(),
    )

    with events.start_trace("tr_registry_emit"):
        events.advance_step()
        provider = create_provider("gemini:model", api_key="key")
        await provider.generate([])
        await provider.aclose()

    assert len(sink.records) == 1
    assert sink.records[0]["event"] == "call"


@respx.mock
async def test_retried_attempts_land_in_the_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    """The only place this wiring can happen.

    `TracedProvider` wraps `generate()` and sees one outcome, while the attempts
    happen a layer below inside `post_json`. Without this, a call that spent five
    minutes across four attempts left a single row carrying the total.
    """
    sink = MemorySink()
    monkeypatch.setattr("bodol.providers.registry._make_sink", lambda _: sink)
    respx.post(f"{gemini.BASE_URL}{gemini.ENDPOINT}").mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "high demand"}}),
            httpx.Response(200, json=load("toolcalls/gemini_weather_1")),
        ]
    )

    with events.start_trace("tr_registry_retry"):
        events.advance_step()
        provider = create_provider(
            "gemini:gemini-3.6-flash",
            api_key="key",
            # No jitter and no wait: this test is about the record, not the clock.
            retry=RetryPolicy(initial_backoff=0.0, jitter=0.0),
        )
        await provider.generate([])
        await provider.aclose()

    assert [r["event"] for r in sink.records] == ["retry", "call"]
    retry = sink.records[0]
    assert retry["attempt"] == 1
    assert retry["of"] == 4
    assert retry["error_status"] == 503
    assert retry["detail"] == "high demand"
    assert retry["retry_in_s"] == 0.0, "recorded the wait it was about to take"
    assert retry["cost_usd"] is None, "a failed attempt may still have been billed"
    assert retry["step"] == 1, "attributed to the step that caused it"


def test_a_callers_own_attempt_hook_is_not_overwritten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tracing is the default, not an imposition."""
    seen: list[object] = []
    # Bound once: `seen.append` is a fresh object on every attribute access.
    hook = seen.append
    captured: dict[str, Any] = {}

    def adapter(*args: Any, **kwargs: Any) -> _FakeProvider:
        captured.update(kwargs)
        return _FakeProvider()

    monkeypatch.setattr("bodol.providers.registry._make_sink", lambda _: MemorySink())
    monkeypatch.setattr("bodol.providers.registry.gemini.GeminiAdapter", adapter)

    with events.start_trace("tr_registry_hook"):
        create_provider("gemini:model", api_key="key", retry=RetryPolicy(on_attempt=hook))

    assert captured["retry"].on_attempt is hook
