"""Trace records: ambient trace/step context, pricing lookup, cost arithmetic.

Two decisions worth knowing before reading:

**trace_id and step are ambient, not parameters.** They live in ContextVars set
once at `agent.run()` entry, so nothing between the loop and the HTTP call has to
thread them through its signature. ContextVars are copied into tasks spawned by
`asyncio.gather`, so parallel tool dispatch reads the same trace and step;
mutations inside a gathered task do not leak back out, which is what we want.

**Every record carries the rates it was priced at.** Vendors change prices, and a
trace re-costed months later at current rates is a different number than what you
were actually billed. `cost_usd` is computed once, at emit time, from rates that
are written into the row beside it.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from typing import Any

import yaml

from bodol import config
from bodol.providers.base import ModelResponse, Usage

# ---------------------------------------------------------------- ambient context

_trace_id: ContextVar[str | None] = ContextVar("bodol_trace_id", default=None)
_step: ContextVar[int] = ContextVar("bodol_step", default=0)


@contextmanager
def start_trace(trace_id: str | None = None) -> Iterator[str]:
    """Scope a run. Generates an id when not given, and restores the previous
    values on exit so nested or sequential runs do not bleed into each other."""
    tid = trace_id or f"tr_{uuid.uuid4().hex[:16]}"
    trace_token = _trace_id.set(tid)
    step_token = _step.set(0)
    try:
        yield tid
    finally:
        _trace_id.reset(trace_token)
        _step.reset(step_token)


def advance_step() -> int:
    """Called by the agent loop at the top of each iteration."""
    nxt = _step.get() + 1
    _step.set(nxt)
    return nxt


def current_trace_id() -> str | None:
    return _trace_id.get()


def current_step() -> int:
    return _step.get()


# ---------------------------------------------------------------- pricing

# Anthropic returns a dated model id (claude-haiku-4-5-20251001) while Gemini and
# OpenAI return undated ones. Pricing is keyed on the undated form.
_DATE_SUFFIX = re.compile(r"-\d{8}$")


def pricing_key(model: str) -> str:
    return _DATE_SUFFIX.sub("", model)


@dataclass(frozen=True, slots=True)
class Rates:
    """USD per million tokens. None means "not published here" — see pricing.yaml."""

    input: float | None = None
    output: float | None = None
    cached_input: float | None = None

    @property
    def priceable(self) -> bool:
        return self.input is not None and self.output is not None


@cache
def _pricing_table() -> dict[str, Rates]:
    """Flattened `{provider}:{model}` -> Rates, read once per process."""
    if not config.PRICING_FILE.exists():
        return {}
    raw = yaml.safe_load(config.PRICING_FILE.read_text()) or {}
    table: dict[str, Rates] = {}
    for provider, models in raw.items():
        for model, rates in (models or {}).items():
            table[f"{provider}:{model}"] = Rates(
                input=(rates or {}).get("input"),
                output=(rates or {}).get("output"),
                cached_input=(rates or {}).get("cached_input"),
            )
    return table


def rates_for(provider: str, model: str) -> Rates:
    return _pricing_table().get(f"{provider}:{pricing_key(model)}", Rates())


def cost_usd(usage: Usage, rates: Rates) -> float | None:
    """None when the model has no published rates — an unpriced call is visible
    as a null in the trace rather than a silent zero.

    Known simplification: cache *writes* bill above the standard input rate
    (~1.25x on Anthropic), but Usage folds them into input_tokens without a
    separate count, so they are priced as ordinary input here. Under-reports on
    the turn that populates a cache; correct on every turn that reads one.
    """
    if not rates.priceable:
        return None
    assert rates.input is not None and rates.output is not None  # narrowed by priceable

    cached = min(usage.cached_tokens, usage.input_tokens)
    uncached = usage.input_tokens - cached
    cached_rate = rates.cached_input if rates.cached_input is not None else rates.input

    per_token = 1_000_000
    total = (
        uncached * rates.input + cached * cached_rate + usage.output_tokens * rates.output
    ) / per_token
    return round(total, 10)


# ---------------------------------------------------------------- records


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def call_record(response: ModelResponse) -> dict[str, Any]:
    """One JSONL row for a successful provider call."""
    rates = rates_for(response.provider, response.model)
    usage = response.usage
    return {
        "ts": _now(),
        "trace_id": current_trace_id(),
        "step": current_step(),
        "event": "call",
        "provider": response.provider,
        "model": response.model,
        "pricing_key": pricing_key(response.model),
        "response_id": response.id,
        "finish_reason": str(response.finish_reason),
        "latency_ms": round(response.latency_ms, 2),
        "tokens": {
            "input": usage.input_tokens,
            "output": usage.output_tokens,
            # Zero across a whole run means the prompt prefix is moving. That is
            # invisible unless it is written down, so it is written down.
            "cached": usage.cached_tokens,
            "reasoning": usage.reasoning_tokens,
            "total": usage.total_tokens,
        },
        "cost_usd": cost_usd(usage, rates),
        "rates_usd_per_mtok": {
            "input": rates.input,
            "output": rates.output,
            "cached_input": rates.cached_input,
        },
        "tool_calls": [c.name for c in response.tool_calls],
        "malformed_tool_calls": [c.name for c in response.tool_calls if c.args is None],
    }


def error_record(
    provider: str, model: str, exc: BaseException, *, latency_ms: float | None = None
) -> dict[str, Any]:
    """One JSONL row for a call that never produced a response.

    Cost is unknown rather than zero: a request that timed out client-side may
    still have been generated and billed server-side.
    """
    status = getattr(exc, "status", None)
    return {
        "ts": _now(),
        "trace_id": current_trace_id(),
        "step": current_step(),
        "event": "error",
        "provider": provider,
        "model": model,
        "error_type": type(exc).__name__,
        "error_status": status,
        "error_message": str(exc)[:500],
        "latency_ms": round(latency_ms, 2) if latency_ms is not None else None,
        "cost_usd": None,
    }
