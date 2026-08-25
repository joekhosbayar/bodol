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
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

# One clock for every deadline here. `perf_counter` measures short intervals
# better, but a deadline set by the agent loop has to be comparable to a reading
# taken inside this module, and two different monotonic clocks in one comparison
# is a bug waiting for a slow afternoon.
_clock = time.monotonic

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


class QuotaError(ProviderHTTPError):
    """The allowance is gone, and asking again spends more of it.

    Deliberately NOT a `TransientError`, which is the whole point of a separate
    type: the agent loop retries transient failures, and retrying this one is
    how a 90-second wait becomes a three-minute one. A 429 is temporary in the
    sense that it clears eventually; it is not temporary in the sense that
    waiting-and-asking makes it clear sooner. Those are different things and the
    old code could not tell them apart.

    Raised only once the provider's own requested delay has *grown* — see
    `_delay_grew`. A single 429 still earns one retry, because a per-minute rate
    limit really does clear on its own.
    """


# The run's deadline, on `_clock`, or None when nothing is bounding the run.
#
# Ambient rather than a parameter, for the reason `telemetry/events.py` sets out
# for `trace_id` and `step`: the alternative is threading a deadline through
# `Provider.generate`, three adapters and `TracedProvider`, none of which have
# any other business knowing what time it is. ContextVars are copied into tasks
# spawned by `asyncio.gather`, so parallel calls inside one run share it.
_run_deadline: ContextVar[float | None] = ContextVar("bodol_run_deadline", default=None)

# The delay a provider last asked for, in its own words, anywhere in this run.
#
# Run-scoped rather than call-scoped because the escalation that identifies a
# spent quota now spans calls: the agent loop retries a failed step, so a
# sequence that used to be four attempts inside one call can be one attempt in
# each of four calls. Kept per-run so one run cannot poison the next.
_last_asked: ContextVar[float | None] = ContextVar("bodol_last_asked", default=None)


@contextmanager
def run_budget(seconds: float | None) -> Iterator[None]:
    """Bound every call made inside this block by the run's own time budget.

    Without it, `max_total_seconds` is the only ceiling a call knows about, and a
    run asked to stop after 10 seconds keeps a call alive for 90 — the run's
    limit is checked between calls, so it cannot reach inside one.

    Restores the previous values on exit, so sequential runs in one process do
    not inherit each other's deadlines or each other's quota history.
    """
    deadline = _run_deadline.set(None if seconds is None else _clock() + seconds)
    asked = _last_asked.set(None)
    try:
        yield
    finally:
        _last_asked.reset(asked)
        _run_deadline.reset(deadline)


@dataclass(frozen=True, slots=True)
class Budget:
    """The deadline a call must respect, and which limit imposed it.

    The source is carried because an operator reading "the deadline expired"
    immediately needs to know whose deadline: theirs, from `--max-seconds`, or
    the library's own per-call ceiling. Those have different fixes.
    """

    deadline: float | None
    source: str

    def remaining(self) -> float | None:
        return None if self.deadline is None else max(self.deadline - _clock(), 0.0)


def _budget(policy: RetryPolicy, started: float) -> Budget:
    """Whichever of the two clocks runs out first.

    `min`, not either alone. The run's budget must be able to cut a call short,
    and the call's ceiling must still stop one call from eating a long run — a
    600-second run does not license a single 600-second retry sequence.
    """
    call = None if policy.max_total_seconds is None else started + policy.max_total_seconds
    run = _run_deadline.get()
    call_source = (
        "call budget"
        if policy.max_total_seconds is None
        else f"{policy.max_total_seconds:.0f}s retry budget"
    )

    if run is None:
        return Budget(call, call_source)
    if call is None or run < call:
        return Budget(run, "run's remaining time")
    return Budget(call, call_source)


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


# Google names the exhausted quota inside the 429 prose, after ~240 characters of
# boilerplate. The metric is a path; only its last segment is worth reading.
_QUOTA_METRIC = re.compile(r"Quota exceeded for metric:\s*(\S+)")
_QUOTA_LIMIT = re.compile(r"\blimit:\s*(\d+)")

# The provider's stated delay is precise to nine decimal places, so a whole
# second of tolerance is far more than jitter needs and safely under the smallest
# real escalation observed (16.6s to 27.3s).
DELAY_GROWTH_TOLERANCE = 1.0


def _quota_note(detail: str) -> str:
    """Name the exhausted quota, when the vendor named it.

    Used only to enrich a message, never to decide anything — deciding on a
    vendor's phrasing would make this layer's behavior hostage to a copy edit.
    """
    metric = _QUOTA_METRIC.search(detail)
    if metric is None:
        return ""
    name = metric.group(1).rstrip(",").rsplit("/", 1)[-1]
    limit = _QUOTA_LIMIT.search(detail)
    said_limit = f", limit {limit.group(1)}" if limit is not None else ""
    return f" (quota {name}{said_limit})"


def _delay_grew(asked_for: float | None, previous: float | None) -> bool:
    """Is the provider asking for longer than it asked for last time?

    That is the fingerprint of retrying an exhausted request quota: every attempt
    spends another request from the allowance it is waiting on, so the wait gets
    longer rather than shorter. Observed on three consecutive runs — 27.3s to
    59.6s, 2.4s to 59.7s, 18.1s to 59.7s — while the run made no progress at all.

    Both delays must be the provider's own words. Our exponential backoff grows
    by construction, and reading our own escalation as the provider's would abort
    every retry sequence on its second attempt.
    """
    if asked_for is None or previous is None:
        return False
    return asked_for > previous + DELAY_GROWTH_TOLERANCE


def _timeout_note(exc: BaseException, timeout: httpx.Timeout) -> str:
    """Describe a transport exception, including the timeout that caused it.

    `httpx` timeout exceptions carry no message at all, so `repr()` renders the
    useless and slightly insulting `ReadTimeout('')` — which is what a caller saw
    after this layer cut off its own request at a budget-clamped 27.4 seconds.
    """
    if not isinstance(exc, httpx.TimeoutException):
        return repr(exc)

    kind = type(exc).__name__
    seconds = {
        "ConnectTimeout": timeout.connect,
        "ReadTimeout": timeout.read,
        "WriteTimeout": timeout.write,
        "PoolTimeout": timeout.pool,
    }.get(kind)
    return kind if seconds is None else f"{kind} after {seconds:.1f}s"


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

    Raises PermanentError for 4xx that won't improve, QuotaError when the provider
    starts asking for longer waits than it asked for before, and TransientError
    when retries run out, the transport keeps failing, or a deadline is spent.

    Two deadlines apply, and the tighter one wins: this policy's
    `max_total_seconds`, and the run's own budget if `run_budget` set one. Errors
    name whichever it was, because "the deadline expired" is not actionable until
    you know whose.

    Every failed attempt is reported twice on the way past: to `policy.on_attempt`
    for the trace, and to this module's logger for whoever is watching. A silent
    five-minute retry sequence is indistinguishable from a hung process.

    Note on cost: a request that timed out client-side may still have been billed
    server-side. Retries can therefore cost money that never shows up in a
    response you got to read. If your spend and your traces disagree, look here
    first — the per-attempt records are what make that visible.
    """
    started_total = _clock()
    budget = _budget(policy, started_total)
    deadline = budget.deadline

    # What the provider last told us, kept so a run that ends on a transport
    # failure can still report the reason it was struggling. Losing this is how a
    # quota problem got reported as `ReadTimeout('')`.
    last_status: int | None = None
    last_detail = ""
    # The quota named in *any* response so far. Google names the metric in the
    # first 429 of a sequence and then stops, so reading only the last response
    # loses the one detail that says which allowance is gone.
    quota_note = ""

    def elapsed() -> float:
        return _clock() - started_total

    def vendor_tail() -> str:
        if last_status is None:
            return ""
        said = f": {last_detail}" if last_detail else ""
        return f"; last provider response was HTTP {last_status}{said}"

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
        return deadline is not None and _clock() + delay >= deadline

    for attempt in range(1, policy.max_attempts + 1):
        attempt_timeout = _attempt_timeout(client.timeout, budget.remaining())
        started = _clock()
        try:
            response = await client.post(
                url,
                json=payload,
                headers=dict(headers or {}),
                timeout=attempt_timeout,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            latency_ms = (_clock() - started) * 1000
            note = _timeout_note(exc, attempt_timeout)
            delay = _backoff(attempt, policy)
            # A deadline reached while the request was in flight is not a
            # transport failure, and calling it one sends the reader to their
            # network. We clamped the timeout; we get to own the outcome.
            expired = deadline is not None and _clock() >= deadline
            last = attempt == policy.max_attempts or out_of_budget(delay) or expired
            report(attempt, None, latency_ms, None if last else delay, note)
            if last:
                reason = (
                    f"the {budget.source} expired during attempt {attempt} ({note})"
                    if expired
                    else f"transport failure after {attempt} attempts"
                    f" and {elapsed():.1f}s: {note}"
                )
                raise TransientError(
                    f"{reason}{vendor_tail()}",
                    status=None,
                    body=None,
                    url=url,
                ) from exc
            await asyncio.sleep(delay)
            continue

        latency_ms = (_clock() - started) * 1000

        if response.status_code < 400:
            return HTTPResult(
                status=response.status_code,
                body=_parse_body(response),
                headers=dict(response.headers),
                latency_ms=latency_ms,
                total_ms=(_clock() - started_total) * 1000,
                attempts=attempt,
            )

        body = _parse_body(response)
        # Folded into the message, not just left on `.body`: the message is what
        # reaches a user through the CLI's one-line error.
        detail = _detail(body)
        said = f": {detail}" if detail else ""
        last_status, last_detail = response.status_code, detail
        quota_note = _quota_note(detail) or quota_note

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

        previous_asked = _last_asked.get()
        if _delay_grew(asked_for, previous_asked):
            # Waiting is not what fixes this. The allowance is spent, and every
            # attempt spends more of it, which is precisely why the number the
            # provider is asking for went up instead of down.
            assert asked_for is not None and previous_asked is not None  # _delay_grew
            report(attempt, response.status_code, latency_ms, None, detail)
            raise QuotaError(
                f"HTTP {response.status_code} from {url}: out of allowance rather"
                f" than busy{quota_note}. The wait asked for grew from"
                f" {previous_asked:.1f}s to {asked_for:.1f}s over this run, which"
                f" is what retrying an exhausted quota looks like: each attempt"
                f" spends another request from it{said}",
                status=response.status_code,
                body=body,
                url=url,
            )
        if asked_for is not None:
            _last_asked.set(asked_for)

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
                f" {budget.source}{said}",
                status=response.status_code,
                body=body,
                url=url,
            )

        report(attempt, response.status_code, latency_ms, delay, detail)
        await asyncio.sleep(delay)

    raise AssertionError("unreachable: retry loop exited without a result")
