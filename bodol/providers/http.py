"""Shared HTTP plumbing for provider adapters.

Deliberately knows nothing about any vendor: no auth, no request bodies, no
response parsing. Adapters build the payload and headers, call `post_json`, and
normalize what comes back.

The vendor SDKs are dev-only dependencies. Import `google.genai.types` in a REPL
to check your request shapes against the real schema, but keep them out of the
runtime path — the whole point of hand-rolling this layer is that the wire
format stays visible.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
import textwrap
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

# A library does not own stdio. This layer reports a retry and leaves the
# decision about whether a human should see it to whoever configured logging —
# `bodol/cli.py` attaches a stderr handler; an embedding application need not.
logger = logging.getLogger(__name__)

# Generous read timeout: a long tool-calling turn on a large context can sit
# quiet for a while. Connect stays short — a slow connect is a dead host.
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=180.0, write=30.0, pool=10.0)

DEFAULT_LIMITS = httpx.Limits(max_connections=32, max_keepalive_connections=16)

# 408/409 are rare here but safe to retry; the 5xx family and 429 are the ones
# that actually fire in practice.
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class ProviderHTTPError(Exception):
    """Base for HTTP-layer failures. Carries enough to log without re-reading the response."""

    def __init__(self, message: str, *, status: int | None, body: Any, url: str) -> None:
        super().__init__(message)
        self.status = status
        self.body = body
        self.url = url


class TransientError(ProviderHTTPError):
    """Retryable, but retries were exhausted or disallowed."""


class PermanentError(ProviderHTTPError):
    """Not worth retrying — bad request, bad auth, model not found."""


@dataclass(frozen=True, slots=True)
class Attempt:
    """One HTTP attempt that failed, reported as it happens.

    `delay_s` is the wait before the next attempt, or None when there will not
    be one — so a None here marks the attempt the caller's exception came from.
    """

    number: int
    of: int
    status: int | None
    latency_ms: float
    delay_s: float | None
    detail: str


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 4
    initial_backoff: float = 0.5
    max_backoff: float = 30.0
    multiplier: float = 2.0
    # Fraction of the delay to randomize by, +/-. Prevents a batch of parallel
    # runs from re-colliding in lockstep after a shared 429.
    jitter: float = 0.3
    retry_on: frozenset[int] = RETRYABLE_STATUS
    # Cap on a single wait, whether the provider asked for it by header or in
    # prose. Providers occasionally hand back minutes.
    max_retry_after: float = 60.0
    # Ceiling on the whole sequence: attempts, waits, and any attempt still in
    # flight. Deliberately sits between the two numbers around it — above
    # `max_retry_after`, so one honored provider delay can still complete, and
    # below the CLI's 120s default `max_seconds`, so a single call cannot eat a
    # whole run's time budget.
    #
    # It bounds each attempt's timeout too, not just the decision to start one.
    # Without that, a provider holding a request open for 78 seconds (Gemini
    # does, under load) overruns the budget inside one attempt: four of those
    # took a nominally 120-second run to 313 seconds.
    #
    # None disables the ceiling for a caller that wants the older behavior.
    max_total_seconds: float | None = 90.0
    # Called for every failed attempt. Kept on the policy rather than threaded
    # through three adapters as an argument: `create_provider` already builds
    # this object and already holds the trace sink.
    on_attempt: Callable[[Attempt], None] | None = None


DEFAULT_RETRY = RetryPolicy()


@dataclass(frozen=True, slots=True)
class HTTPResult:
    status: int
    body: Any
    headers: Mapping[str, str]
    # Wall time of the attempt that succeeded.
    latency_ms: float
    # Wall time across every attempt including backoff sleeps. When this and
    # latency_ms diverge, retries are eating your wall-clock budget.
    total_ms: float
    attempts: int


def make_client(
    *,
    base_url: str = "",
    headers: Mapping[str, str] | None = None,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    limits: httpx.Limits = DEFAULT_LIMITS,
) -> httpx.AsyncClient:
    """Build an async client. Adapters own the lifecycle — use it as a context manager
    or close it explicitly, and reuse one per adapter so connections stay pooled."""
    return httpx.AsyncClient(
        base_url=base_url,
        headers=dict(headers or {}),
        timeout=timeout,
        limits=limits,
        follow_redirects=False,
    )


def _parse_body(response: httpx.Response) -> Any:
    """Error responses aren't always JSON — a proxy 502 is usually HTML."""
    try:
        return response.json()
    except ValueError:
        return response.text


# Wide enough to survive a vendor that buries the point. Google's 429 spends its
# first ~240 characters on boilerplate and two doc links before naming the quota
# that was exceeded, and the quota is the only part worth reading. Still bounded,
# because an HTML error page must not become the error text.
MAX_DETAIL = 500

# The live retry notice keeps only enough to recognize the failure: it already
# states the wait separately, and it competes with a terminal's width.
NOTICE_DETAIL = 120


def _clamp(text: str, width: int = MAX_DETAIL) -> str:
    """Collapse whitespace and cut long bodies down to one readable line."""
    return textwrap.shorten(text.strip(), width=width, placeholder=" …")


def _detail(body: Any) -> str:
    """The vendor's own description of the failure, when it sent one.

    Without it the operator gets `HTTP 400 from /v1beta/interactions` and has to
    guess between a bad key, a retired model, and a malformed request. Gemini's
    "Invalid input received." is barely more, but it is the difference between
    an unexplained rejection and a named one, and the full body still hangs off
    `.body` for anything richer.

    Shapes handled: {"error": {"message": ...}} (all three vendors), a bare
    {"error": "..."}, a top-level {"message": ...}, and non-JSON text — a proxy
    5xx is usually an HTML page, so the string branch is clamped.
    """
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, str):
            return _clamp(error)
        if isinstance(error, dict):
            for key in ("message", "detail", "status"):
                value = error.get(key)
                if isinstance(value, str) and value.strip():
                    return _clamp(value)
        return ""
    if isinstance(body, str):
        return _clamp(body)
    return ""


# "Please retry in 59.993057449s." — Gemini's delay, in prose, in the message.
_RETRY_HINT = re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)


def _retry_hint_seconds(detail: str, policy: RetryPolicy) -> float | None:
    """The delay a vendor states in words because it sent no header.

    Gemini's interactions endpoint returns 429 with no Retry-After and no
    structured RetryInfo — the only copy of the delay is English inside
    `error.message`. Parsing prose is unpleasant, and the alternative is worse:
    exponential backoff waits about four seconds in total against a limit that
    wants sixty, so every attempt fails and each one spends another request
    from the quota that is already exhausted.

    Still capped by `max_retry_after`, so a vendor asking for ten minutes
    cannot quietly consume the agent's wall-clock budget.
    """
    match = _RETRY_HINT.search(detail)
    if match is None:
        return None
    return min(float(match.group(1)), policy.max_retry_after)


def _retry_after_seconds(response: httpx.Response, policy: RetryPolicy) -> float | None:
    """Retry-After is either a delta in seconds or an HTTP-date. Both appear in the wild."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None

    try:
        return min(float(raw), policy.max_retry_after)
    except ValueError:
        pass

    try:
        when = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None

    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    delta = (when - datetime.now(UTC)).total_seconds()
    return min(max(delta, 0.0), policy.max_retry_after)


def _backoff(attempt: int, policy: RetryPolicy) -> float:
    """Exponential with proportional jitter. `attempt` is 1-based."""
    base = min(policy.initial_backoff * policy.multiplier ** (attempt - 1), policy.max_backoff)
    return base * random.uniform(1.0 - policy.jitter, 1.0 + policy.jitter)


def _attempt_timeout(base: httpx.Timeout, remaining: float | None) -> httpx.Timeout:
    """Clamp one attempt's timeouts to the retry budget that is left.

    The deadline has to reach inside the attempt, not merely decide whether to
    start another one. A provider that holds a request open for 78 seconds
    overruns a 90-second budget on its own, and four of them turned a
    120-second run into 313 seconds.
    """
    if remaining is None:
        return base

    def clamp(value: float | None) -> float | None:
        return remaining if value is None else min(value, remaining)

    return httpx.Timeout(
        connect=clamp(base.connect),
        read=clamp(base.read),
        write=clamp(base.write),
        pool=clamp(base.pool),
    )


async def post_json(
    client: httpx.AsyncClient,
    url: str,
    payload: Any,
    *,
    headers: Mapping[str, str] | None = None,
    policy: RetryPolicy = DEFAULT_RETRY,
) -> HTTPResult:
    """POST JSON, retrying transient failures, and return the decoded body with timings.

    Raises PermanentError for 4xx that won't improve, TransientError when retries
    run out, the transport keeps failing, or the retry budget is spent.

    Every failed attempt is reported twice on the way past: to `policy.on_attempt`
    for the trace, and to this module's logger for whoever is watching. A silent
    five-minute retry sequence is indistinguishable from a hung process.

    Note on cost: a request that timed out client-side may still have been billed
    server-side. Retries can therefore cost money that never shows up in a
    response you got to read. If your spend and your traces disagree, look here
    first — the per-attempt records are what make that visible.
    """
    started_total = time.perf_counter()
    deadline = (
        None if policy.max_total_seconds is None else started_total + policy.max_total_seconds
    )

    def elapsed() -> float:
        return time.perf_counter() - started_total

    def report(
        attempt: int, status: int | None, latency_ms: float, delay: float | None, detail: str
    ) -> None:
        if policy.on_attempt is not None:
            policy.on_attempt(
                Attempt(
                    number=attempt,
                    of=policy.max_attempts,
                    status=status,
                    latency_ms=round(latency_ms, 2),
                    delay_s=delay,
                    detail=detail,
                )
            )
        # Only the waits are announced. A failure with nothing left to try is
        # about to be raised, and the caller reports that in its own words.
        if delay is not None:
            logger.warning(
                "retry %d/%d · %s · waiting %.1fs · %s",
                attempt,
                policy.max_attempts,
                f"HTTP {status}" if status is not None else "transport failure",
                delay,
                # Shorter than the exception's detail on purpose. This line is
                # for recognizing the problem at a glance while the wait runs;
                # the full text is in the trace and in the final error, and a
                # vendor paragraph wrapped across a terminal reads as noise.
                _clamp(detail, NOTICE_DETAIL) if detail else url,
            )

    def out_of_budget(delay: float) -> bool:
        return deadline is not None and time.perf_counter() + delay >= deadline

    for attempt in range(1, policy.max_attempts + 1):
        remaining = None if deadline is None else max(deadline - time.perf_counter(), 0.0)
        started = time.perf_counter()
        try:
            response = await client.post(
                url,
                json=payload,
                headers=dict(headers or {}),
                timeout=_attempt_timeout(client.timeout, remaining),
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            delay = _backoff(attempt, policy)
            last = attempt == policy.max_attempts or out_of_budget(delay)
            report(attempt, None, latency_ms, None if last else delay, repr(exc))
            if last:
                raise TransientError(
                    f"transport failure after {attempt} attempts"
                    f" and {elapsed():.1f}s: {exc!r}",
                    status=None,
                    body=None,
                    url=url,
                ) from exc
            await asyncio.sleep(delay)
            continue

        latency_ms = (time.perf_counter() - started) * 1000

        if response.status_code < 400:
            return HTTPResult(
                status=response.status_code,
                body=_parse_body(response),
                headers=dict(response.headers),
                latency_ms=latency_ms,
                total_ms=(time.perf_counter() - started_total) * 1000,
                attempts=attempt,
            )

        body = _parse_body(response)
        # Folded into the message, not just left on `.body`: the message is what
        # reaches a user through the CLI's one-line error.
        detail = _detail(body)
        said = f": {detail}" if detail else ""

        if response.status_code not in policy.retry_on:
            report(attempt, response.status_code, latency_ms, None, detail)
            raise PermanentError(
                f"HTTP {response.status_code} from {url}{said}",
                status=response.status_code,
                body=body,
                url=url,
            )

        # Header first, the vendor's own prose second, blind backoff last.
        asked_for = _retry_after_seconds(response, policy)
        if asked_for is None:
            asked_for = _retry_hint_seconds(detail, policy)
        delay = asked_for if asked_for is not None else _backoff(attempt, policy)

        if attempt == policy.max_attempts:
            report(attempt, response.status_code, latency_ms, None, detail)
            raise TransientError(
                f"HTTP {response.status_code} from {url}"
                f" after {attempt} attempts and {elapsed():.1f}s{said}",
                status=response.status_code,
                body=body,
                url=url,
            )

        if out_of_budget(delay):
            # Stopping here rather than waiting is the point of the budget: the
            # provider is asking for more time than this call is allowed to
            # spend, and finding that out late costs the caller the difference.
            report(attempt, response.status_code, latency_ms, None, detail)
            raise TransientError(
                f"HTTP {response.status_code} from {url} after {attempt} attempts"
                f" and {elapsed():.1f}s; a {delay:.1f}s wait would exceed the"
                f" {policy.max_total_seconds:.0f}s retry budget{said}",
                status=response.status_code,
                body=body,
                url=url,
            )

        report(attempt, response.status_code, latency_ms, delay, detail)
        await asyncio.sleep(delay)

    raise AssertionError("unreachable: retry loop exited without a result")
