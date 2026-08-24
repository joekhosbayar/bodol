"""The agent: loop, limits, stop rules, run results."""

from bodol.agent import prompts
from bodol.agent.loop import DEFAULT_LIMITS, Agent, Limits, RunResult, StopReason

__all__ = [
    "DEFAULT_LIMITS",
    "Agent",
    "Limits",
    "RunResult",
    "StopReason",
    "prompts",
]
