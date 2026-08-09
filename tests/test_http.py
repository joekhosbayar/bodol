"""Tests for the shared HTTP plumbing. No network, no real sleeps."""

import asyncio
from collections.abc import Iterator

import httpx
import pytest
import respx

from bodol.providers import http

URL = "https://example.test/v1/generate"


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    """Record backoff delays instead of waiting them out."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    yield slept


@respx.mock
async def test_success_returns_decoded_body() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"ok": True}))

    async with http.make_client() as client:
        result = await http.post_json(client, URL, {"prompt": "hi"})

    assert result.status == 200
    assert result.body == {"ok": True}
    assert result.attempts == 1
    assert result.latency_ms >= 0


@respx.mock
async def test_retries_transient_then_succeeds(no_sleep: list[float]) -> None:
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
    )

    async with http.make_client() as client:
        result = await http.post_json(client, URL, {})

    assert result.attempts == 3
    assert len(no_sleep) == 2


@respx.mock
async def test_permanent_error_is_not_retried(no_sleep: list[float]) -> None:
    respx.post(URL).mock(return_value=httpx.Response(400, json={"error": "bad model"}))

    async with http.make_client() as client:
        with pytest.raises(http.PermanentError) as exc_info:
            await http.post_json(client, URL, {})

    assert exc_info.value.status == 400
    assert exc_info.value.body == {"error": "bad model"}
    assert no_sleep == []


@respx.mock
async def test_retries_exhausted_raises_transient(no_sleep: list[float]) -> None:
    respx.post(URL).mock(return_value=httpx.Response(429))
    policy = http.RetryPolicy(max_attempts=3)

    async with http.make_client() as client:
        with pytest.raises(http.TransientError):
            await http.post_json(client, URL, {}, policy=policy)

    assert len(no_sleep) == 2


@respx.mock
async def test_retry_after_header_wins_over_backoff(no_sleep: list[float]) -> None:
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "7"}),
            httpx.Response(200, json={}),
        ]
    )

    async with http.make_client() as client:
        await http.post_json(client, URL, {})

    assert no_sleep == [7.0]


@respx.mock
async def test_retry_after_is_capped(no_sleep: list[float]) -> None:
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "9999"}),
            httpx.Response(200, json={}),
        ]
    )
    policy = http.RetryPolicy(max_retry_after=30.0)

    async with http.make_client() as client:
        await http.post_json(client, URL, {}, policy=policy)

    assert no_sleep == [30.0]


@respx.mock
async def test_transport_failure_retries_then_raises(no_sleep: list[float]) -> None:
    respx.post(URL).mock(side_effect=httpx.ConnectError("refused"))
    policy = http.RetryPolicy(max_attempts=2)

    async with http.make_client() as client:
        with pytest.raises(http.TransientError):
            await http.post_json(client, URL, {}, policy=policy)

    assert len(no_sleep) == 1


@respx.mock
async def test_non_json_error_body_falls_back_to_text() -> None:
    respx.post(URL).mock(return_value=httpx.Response(400, text="<html>nope</html>"))

    async with http.make_client() as client:
        with pytest.raises(http.PermanentError) as exc_info:
            await http.post_json(client, URL, {})

    assert exc_info.value.body == "<html>nope</html>"
