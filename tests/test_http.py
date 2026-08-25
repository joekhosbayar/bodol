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
async def test_a_retry_delay_stated_in_prose_is_honored(no_sleep: list[float]) -> None:
    """Gemini sends no Retry-After header and no structured RetryInfo.

    Without reading the message, backoff waits ~0.5s against a limit that wants
    a minute — and burns another request from the exhausted quota to find out.
    """
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                429,
                json={
                    "error": {
                        "message": (
                            "You exceeded your current quota. * Quota exceeded for metric: "
                            "generate_content_free_tier_requests, limit: 20 "
                            "Please retry in 12.5s."
                        )
                    }
                },
            ),
            httpx.Response(200, json={}),
        ]
    )

    async with http.make_client() as client:
        await http.post_json(client, URL, {})

    assert no_sleep == [12.5]


@respx.mock
async def test_a_header_still_wins_over_prose(no_sleep: list[float]) -> None:
    """The header is the standard; prose is only the fallback."""
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(
                429,
                headers={"retry-after": "3"},
                json={"error": {"message": "Please retry in 45s."}},
            ),
            httpx.Response(200, json={}),
        ]
    )

    async with http.make_client() as client:
        await http.post_json(client, URL, {})

    assert no_sleep == [3.0]


def test_a_prose_delay_is_capped_like_the_header() -> None:
    policy = http.RetryPolicy(max_retry_after=30.0)
    assert http._retry_hint_seconds("Please retry in 600s.", policy) == 30.0
    assert http._retry_hint_seconds("no delay stated here", policy) is None


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


@respx.mock
async def test_vendor_error_message_reaches_the_exception_text() -> None:
    """The CLI prints one line, so the vendor's reason has to be in it.

    The first live Gemini failure read `HTTP 400 from /v1beta/interactions` and
    nothing more — identical to what a bad key or a retired model would print,
    while the body sitting on the exception said `Invalid input received.`
    """
    respx.post(URL).mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "Invalid input received.", "code": "invalid_request"}},
        )
    )

    async with http.make_client() as client:
        with pytest.raises(http.PermanentError, match="Invalid input received") as exc_info:
            await http.post_json(client, URL, {})

    assert "400" in str(exc_info.value)
    assert exc_info.value.body["error"]["code"] == "invalid_request", "full body still attached"


@respx.mock
async def test_exhausted_retries_also_say_why(no_sleep: list[float]) -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(429, json={"error": {"message": "quota exceeded"}})
    )

    async with http.make_client() as client:
        with pytest.raises(http.TransientError, match="after 2 attempts: quota exceeded"):
            await http.post_json(client, URL, {}, policy=http.RetryPolicy(max_attempts=2))


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"error": {"message": "Invalid input received."}}, "Invalid input received."),
        ({"error": "bad model"}, "bad model"),
        ({"message": "quota exceeded"}, "quota exceeded"),
        ({"error": {"status": "PERMISSION_DENIED"}}, "PERMISSION_DENIED"),
        # No words anywhere in it: better a bare status line than "400: 400".
        ({"error": {"code": 400}}, ""),
        ({}, ""),
        (None, ""),
        ("<html>\n  nope\n</html>", "<html> nope </html>"),
    ],
)
def test_detail_reads_the_error_shapes_the_three_vendors_send(body: object, expected: str) -> None:
    assert http._detail(body) == expected


def test_html_error_pages_are_clamped_to_one_line() -> None:
    """A proxy 502 is an HTML document, and it must not become the error text."""
    detail = http._detail("<html>" + "nope " * 500 + "</html>")
    assert len(detail) <= http.MAX_DETAIL
    assert detail.endswith("…")


def test_the_actionable_end_of_a_google_quota_message_survives() -> None:
    """Regression on the clamp width.

    Google's 429 opens with 240 characters of boilerplate and two doc links,
    then names the quota. A 300-character clamp cut the message exactly one word
    before the metric — the only part that tells you what to change.
    """
    detail = http._detail(
        {
            "error": {
                "message": (
                    "You exceeded your current quota, please check your plan and billing "
                    "details. For more information on this error, head to: "
                    "https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your "
                    "current usage, head to: https://ai.dev/rate-limit. * Quota exceeded "
                    "for metric: generativelanguage.googleapis.com/"
                    "generate_content_free_tier_requests, limit: 20"
                )
            }
        }
    )
    assert detail.endswith("limit: 20")
