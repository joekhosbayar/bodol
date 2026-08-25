"""Model context windows and the budget derived from them.

A `ContextPolicy` needs a token budget, and hard-coding one is wrong in both
directions: 120k wastes nine tenths of a Gemini window and overflows a smaller
model outright. The budget is instead a fraction of the window of the model
actually being run.

Why a fraction and not the whole window: `max_input_tokens` is compared against
the input of the call that just happened, so compaction always starts *after* an
over-budget call has been sent. Leaving headroom means the call that trips the
budget still fits, and the summarization call has room to work in. Sixty percent
leaves that slack without compacting so eagerly that a run summarizes history it
was still using.

An unknown window yields no policy at all rather than a default guess. Sizing a
budget from a number nobody verified is how a run gets compacted at 8k on a
model that could have held a million tokens.
"""

from __future__ import annotations

from functools import cache
from typing import Any

import yaml

from bodol import config
from bodol.context.manager import ContextPolicy
from bodol.providers.registry import parse_provider_spec
from bodol.telemetry.events import pricing_key

# Sixty percent of the window. See the module docstring for why not 100.
DEFAULT_BUDGET_RATIO = 0.6


@cache
def _window_table() -> dict[str, int]:
    """Flattened `{provider}:{model}` -> window, read once per process.

    Null and non-integer entries are dropped rather than stored, so a lookup
    miss and an explicitly-unknown window are the same answer: None.
    """
    if not config.CONTEXT_WINDOW_FILE.exists():
        return {}
    raw: dict[str, Any] = yaml.safe_load(config.CONTEXT_WINDOW_FILE.read_text()) or {}
    table: dict[str, int] = {}
    for provider, models in raw.items():
        for model, window in (models or {}).items():
            if isinstance(window, int) and window > 0:
                table[f"{provider}:{model}"] = window
    return table


def context_window(provider: str, model: str) -> int | None:
    """Input window for a model, or None when it is unknown."""
    return _window_table().get(f"{provider.strip().lower()}:{pricing_key(model)}")


def policy_for(
    spec: str,
    *,
    ratio: float = DEFAULT_BUDGET_RATIO,
    default_family: str | None = None,
    summary_max_tokens: int = 1024,
    min_recent_turns: int = 2,
) -> ContextPolicy | None:
    """Build a compaction policy for a provider spec, or None if unknown.

    Takes the same `family:model` string the CLI accepts, so a caller that has
    a `--provider` value has everything it needs.

    The window is resolved from the *requested* model id, before any call has
    been made. An alias that resolves server-side to a model with a different
    window will be budgeted by the alias — visible in the trace, where
    `ModelResponse.model` reports what actually served the request.
    """
    if not 0.0 < ratio <= 1.0:
        raise ValueError(f"ratio must be in (0, 1], got {ratio}")

    family, model = parse_provider_spec(spec, default_family=default_family)
    window = context_window(family, model)
    if window is None:
        return None

    budget = int(window * ratio)
    if budget < 1:
        raise ValueError(f"ratio {ratio} yields an unusable budget for a {window}-token window")
    return ContextPolicy(
        max_input_tokens=budget,
        summary_max_tokens=summary_max_tokens,
        min_recent_turns=min_recent_turns,
    )


__all__ = ["DEFAULT_BUDGET_RATIO", "context_window", "policy_for"]
