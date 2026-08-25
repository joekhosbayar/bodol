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
        with pytest.raises(http.TransientError, match=r"after 2 attempts and [\d.]+s: quota"):
            await http.post_json(client, URL, {}, policy=http.RetryPolicy(max_attempts=2))


@respx.mock
async def test_the_retry_budget_refuses_a_wait_it_cannot_afford(no_sleep: list[float]) -> None:
    """A provider asking for more time than the call is allowed to spend.

    Honoring the 60s hint here would blow the budget and then still have three
    attempts to go. Better to stop at once and say why: the caller learns the
    real problem in a second instead of discovering it four minutes later.
    """
    respx.post(URL).mock(
        return_value=httpx.Response(429, json={"error": {"message": "Please retry in 60s."}})
    )
    policy = http.RetryPolicy(max_total_seconds=10.0)

    async with http.make_client() as client:
        with pytest.raises(http.TransientError, match="would exceed the 10s retry budget"):
            await http.post_json(client, URL, {}, policy=policy)

    assert no_sleep == [], "gave up instead of sleeping past the budget"


@respx.mock
async def test_each_attempt_is_bounded_by_the_budget_that_is_left() -> None:
    """The 313-second run: four attempts a provider held open for ~78s each.

    A deadline that only gates whether to *start* an attempt cannot stop that,
    so the remaining budget is pushed down into the request's own timeout.
    """
    seen: list[dict[str, float | None]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={})

    respx.post(URL).mock(side_effect=capture)

    async with http.make_client() as client:
        await http.post_json(client, URL, {}, policy=http.RetryPolicy(max_total_seconds=5.0))

    read = seen[0]["read"]
    assert read is not None and read <= 5.0, "read timeout clamped to the budget"
    connect = seen[0]["connect"]
    assert connect is not None and connect <= 10.0, "a short connect timeout is not widened"


@respx.mock
async def test_no_budget_leaves_the_client_timeout_alone(no_sleep: list[float]) -> None:
    seen: list[dict[str, float | None]] = []

    def capture(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return httpx.Response(200, json={})

    respx.post(URL).mock(side_effect=capture)

    async with http.make_client() as client:
        await http.post_json(client, URL, {}, policy=http.RetryPolicy(max_total_seconds=None))

    assert seen[0]["read"] == http.DEFAULT_TIMEOUT.read


@respx.mock
async def test_every_failed_attempt_is_reported(no_sleep: list[float]) -> None:
    """What the trace needs: per-attempt status and latency, not just a total."""
    attempts: list[http.Attempt] = []
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(503, json={"error": {"message": "high demand"}}),
            httpx.Response(503, json={"error": {"message": "high demand"}}),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    policy = http.RetryPolicy(on_attempt=attempts.append)

    async with http.make_client() as client:
        result = await http.post_json(client, URL, {}, policy=policy)

    assert result.attempts == 3
    assert [a.number for a in attempts] == [1, 2], "only failures are reported"
    assert [a.status for a in attempts] == [503, 503]
    assert [a.detail for a in attempts] == ["high demand", "high demand"]
    assert all(a.delay_s is not None for a in attempts), "each one is about to be retried"
    assert all(a.of == 4 for a in attempts)


@respx.mock
async def test_the_last_attempt_reports_no_delay(no_sleep: list[float]) -> None:
    """`delay_s is None` marks the attempt the caller's exception came from."""
    attempts: list[http.Attempt] = []
    respx.post(URL).mock(return_value=httpx.Response(500, json={"error": "boom"}))
    policy = http.RetryPolicy(max_attempts=2, on_attempt=attempts.append)

    async with http.make_client() as client:
        with pytest.raises(http.TransientError):
            await http.post_json(client, URL, {}, policy=policy)

    assert [a.delay_s is None for a in attempts] == [False, True]


@respx.mock
async def test_a_permanent_failure_is_reported_once(no_sleep: list[float]) -> None:
    attempts: list[http.Attempt] = []
    respx.post(URL).mock(return_value=httpx.Response(400, json={"error": "bad model"}))

    policy = http.RetryPolicy(on_attempt=attempts.append)
    async with http.make_client() as client:
        with pytest.raises(http.PermanentError):
            await http.post_json(client, URL, {}, policy=policy)

    assert len(attempts) == 1
    assert attempts[0].delay_s is None
    assert no_sleep == []


@respx.mock
async def test_a_wait_is_announced_but_a_final_failure_is_not(
    no_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """Five silent minutes is indistinguishable from a hung process.

    The final failure stays quiet: it is about to be raised, and the caller
    reports it in its own words.
    """
    respx.post(URL).mock(
        return_value=httpx.Response(503, json={"error": {"message": "high demand"}})
    )

    with caplog.at_level("WARNING", logger="bodol.providers.http"):
        async with http.make_client() as client:
            with pytest.raises(http.TransientError):
                await http.post_json(client, URL, {}, policy=http.RetryPolicy(max_attempts=2))

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "retry 1/2" in message
    assert "HTTP 503" in message
    assert "high demand" in message


@respx.mock
async def test_the_live_notice_stays_one_line(
    no_sleep: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    """Google's quota message is a paragraph. It belongs in the trace, not
    wrapped across four terminal rows while the user waits."""
    respx.post(URL).mock(
        return_value=httpx.Response(
            429, json={"error": {"message": "You exceeded your current quota. " * 20}}
        )
    )

    with caplog.at_level("WARNING", logger="bodol.providers.http"):
        async with http.make_client() as client:
            with pytest.raises(http.TransientError):
                await http.post_json(client, URL, {}, policy=http.RetryPolicy(max_attempts=2))

    notice = caplog.records[0].getMessage()
    assert len(notice) < http.MAX_DETAIL, "shorter than what the exception carries"
    assert notice.endswith("…")


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
