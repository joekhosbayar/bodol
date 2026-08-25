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
import random
import re
import textwrap
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

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
class RetryPolicy:
    max_attempts: int = 4
    initial_backoff: float = 0.5
    max_backoff: float = 30.0
    multiplier: float = 2.0
    # Fraction of the delay to randomize by, +/-. Prevents a batch of parallel
    # runs from re-colliding in lockstep after a shared 429.
    jitter: float = 0.3
    retry_on: frozenset[int] = RETRYABLE_STATUS
    # Cap on Retry-After. Providers occasionally hand back minutes; without this
    # a single 429 silently blows through the agent's max_seconds budget.
    max_retry_after: float = 60.0


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
    run out or the transport keeps failing.

    Note on cost: a request that timed out client-side may still have been billed
    server-side. Retries can therefore cost money that never shows up in a
    response you got to read. If your spend and your traces disagree, look here
    first — consider logging every attempt, not just the one that returned.
    """
    started_total = time.perf_counter()
    last_error: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        started = time.perf_counter()
        try:
            response = await client.post(url, json=payload, headers=dict(headers or {}))
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            last_error = exc
            if attempt == policy.max_attempts:
                raise TransientError(
                    f"transport failure after {attempt} attempts: {exc!r}",
                    status=None,
                    body=None,
                    url=url,
                ) from exc
            await asyncio.sleep(_backoff(attempt, policy))
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
            raise PermanentError(
                f"HTTP {response.status_code} from {url}{said}",
                status=response.status_code,
                body=body,
                url=url,
            )

        if attempt == policy.max_attempts:
            raise TransientError(
                f"HTTP {response.status_code} from {url} after {attempt} attempts{said}",
                status=response.status_code,
                body=body,
                url=url,
            )

        # Header first, the vendor's own prose second, blind backoff last.
        delay = _retry_after_seconds(response, policy)
        if delay is None:
            delay = _retry_hint_seconds(detail, policy)
        await asyncio.sleep(delay if delay is not None else _backoff(attempt, policy))

    raise AssertionError(f"unreachable: retry loop exited without result ({last_error!r})")
