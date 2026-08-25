"""Context management: token budgets, truncation, compaction, externalized state."""

from bodol.context.manager import (
    CompactionError,
    ContextManager,
    ContextManagerError,
    ContextPolicy,
)
from bodol.context.windows import DEFAULT_BUDGET_RATIO, context_window, policy_for

__all__ = [
    "DEFAULT_BUDGET_RATIO",
    "CompactionError",
    "ContextManager",
    "ContextManagerError",
    "ContextPolicy",
    "context_window",
    "policy_for",
]
