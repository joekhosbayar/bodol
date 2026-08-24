"""Context management: token budgets, truncation, compaction, externalized state."""

from bodol.context.manager import (
    CompactionError,
    ContextManager,
    ContextManagerError,
    ContextPolicy,
)

__all__ = [
    "CompactionError",
    "ContextManager",
    "ContextManagerError",
    "ContextPolicy",
]
