"""Telemetry: a provider-shaped wrapper that emits one JSONL record per call."""

from bodol.telemetry.events import (
    advance_step,
    call_record,
    cost_usd,
    current_step,
    current_trace_id,
    error_record,
    pricing_key,
    rates_for,
    start_trace,
)
from bodol.telemetry.middleware import TracedProvider
from bodol.telemetry.writer import JsonlSink, MemorySink, NullSink, Sink, read_trace

__all__ = [
    "JsonlSink",
    "MemorySink",
    "NullSink",
    "Sink",
    "TracedProvider",
    "advance_step",
    "call_record",
    "cost_usd",
    "current_step",
    "current_trace_id",
    "error_record",
    "pricing_key",
    "rates_for",
    "read_trace",
    "start_trace",
]
