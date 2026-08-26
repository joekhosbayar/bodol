"""Telemetry tests. Offline — records are built from captured fixtures."""

import asyncio
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from bodol.providers import anthropic
from bodol.providers.base import Message, ModelResponse, TextBlock, ToolSpec, Usage
from bodol.telemetry import events
from bodol.telemetry.middleware import TracedProvider
from bodol.telemetry.writer import JsonlSink, MemorySink, NullSink, Sink, read_trace
from tests.conftest import load

HAIKU = Usage(input_tokens=591, output_tokens=59)


@pytest.fixture
def unpriced_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """A pricing table whose entries are explicitly null.

    The null-rate path used to be tested against whichever shipped model
    happened to be unpriced, which broke the moment that model got a price.
    The behaviour under test belongs to the loader and cost_usd, not to the
    rate card, so the rate card here is a fixture.
    """
    from bodol import config

    table = tmp_path / "pricing.yaml"
    table.write_text(
        "gemini:\n"
        "  gemini-3.6-flash:\n"
        "    input: null\n"
        "    output: null\n"
        "    cached_input: null\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(config, "PRICING_FILE", table)
    events._pricing_table.cache_clear()
    yield
    events._pricing_table.cache_clear()


# ---------------------------------------------------------------- ambient context


def test_trace_id_and_step_are_ambient() -> None:
    assert events.current_trace_id() is None

    with events.start_trace() as tid:
        assert tid.startswith("tr_")
        assert events.current_trace_id() == tid
        assert events.current_step() == 0
        assert events.advance_step() == 1
        assert events.advance_step() == 2
        assert events.current_step() == 2

    assert events.current_trace_id() is None, "context restored on exit"


def test_traces_do_not_bleed_into_each_other() -> None:
    with events.start_trace("tr_a"):
        events.advance_step()
        with events.start_trace("tr_b"):
            assert events.current_step() == 0, "nested trace starts its own step counter"
        assert events.current_trace_id() == "tr_a"
        assert events.current_step() == 1


async def test_step_is_visible_inside_gathered_tasks() -> None:
    """Parallel tool dispatch must see the step that spawned it."""

    async def read_step() -> tuple[str | None, int]:
        return events.current_trace_id(), events.current_step()

    with events.start_trace("tr_par"):
        events.advance_step()
        seen = await asyncio.gather(read_step(), read_step(), read_step())

    assert list(seen) == [("tr_par", 1)] * 3


# ---------------------------------------------------------------- pricing


def test_dated_model_ids_are_normalized_for_lookup() -> None:
    assert events.pricing_key("claude-haiku-4-5-20251001") == "claude-haiku-4-5"
    assert events.pricing_key("gemini-3.6-flash") == "gemini-3.6-flash"


def test_rates_are_found_through_the_dated_id() -> None:
    rates = events.rates_for("anthropic", "claude-haiku-4-5-20251001")
    assert (rates.input, rates.output) == (1.00, 5.00)


def test_cost_of_a_real_captured_call() -> None:
    rates = events.rates_for("anthropic", "claude-haiku-4-5")
    # 591 input @ $1/Mtok + 59 output @ $5/Mtok
    assert events.cost_usd(HAIKU, rates) == pytest.approx(0.000591 + 0.000295)


def test_cached_tokens_are_priced_at_the_cache_rate() -> None:
    rates = events.rates_for("anthropic", "claude-haiku-4-5")
    usage = Usage(input_tokens=1000, output_tokens=0, cached_tokens=800)
    # 200 uncached @ $1.00 + 800 cached @ $0.10
    assert events.cost_usd(usage, rates) == pytest.approx((200 * 1.00 + 800 * 0.10) / 1e6)


def test_unpriced_models_cost_none_not_zero(unpriced_table: None) -> None:
    rates = events.rates_for("gemini", "gemini-3.6-flash")
    assert not rates.priceable
    assert events.cost_usd(HAIKU, rates) is None


def test_every_shipped_model_is_priced() -> None:
    """A null rate is a placeholder. Shipping one silently reports null costs."""
    import yaml

    from bodol import config

    raw = yaml.safe_load(config.PRICING_FILE.read_text())
    unpriced = [
        f"{family}:{model}"
        for family, models in raw.items()
        for model, rates in (models or {}).items()
        if (rates or {}).get("input") is None or (rates or {}).get("output") is None
    ]
    assert unpriced == [], f"models shipped without rates: {unpriced}"


def test_cache_write_rates_are_present_for_every_model() -> None:
    """Forward compatibility: the keys a future cost_usd needs must already be
    filled in or explicitly null, never absent, so nobody has to guess whether a
    missing key meant "no premium" or "nobody looked"."""
    import yaml

    from bodol import config

    raw = yaml.safe_load(config.PRICING_FILE.read_text())
    required = {"cached_input", "cache_write", "cache_write_1h", "cache_storage_per_hour"}
    missing = {
        f"{family}:{model}": sorted(required - set(rates or {}))
        for family, models in raw.items()
        for model, rates in (models or {}).items()
        if required - set(rates or {})
    }
    assert missing == {}, f"cache keys absent: {missing}"


def test_the_loader_reads_all_four_input_rates() -> None:
    """These were shipped as data a release ahead of the code that reads them."""
    rates = events.rates_for("anthropic", "claude-haiku-4-5")
    assert (rates.input, rates.output, rates.cached_input) == (1.00, 5.00, 0.10)
    assert (rates.cache_write, rates.cache_write_1h) == (1.25, 2.00)


def test_cache_writes_are_billed_at_their_premium() -> None:
    """A write costs MORE than fresh input. Folding it into plain input, as this
    used to, under-reports exactly the calls that populate a cache."""
    rates = events.rates_for("anthropic", "claude-haiku-4-5")
    usage = Usage(input_tokens=1000, output_tokens=0, cache_write_tokens=800)

    assert events.cost_usd(usage, rates) == pytest.approx((200 * 1.00 + 800 * 1.25) / 1e6)


def test_the_two_write_tiers_are_priced_apart() -> None:
    rates = events.rates_for("anthropic", "claude-haiku-4-5")
    usage = Usage(
        input_tokens=1000,
        output_tokens=0,
        cached_tokens=400,
        cache_write_tokens=300,
        cache_write_1h_tokens=200,
    )
    expected = (100 * 1.00 + 400 * 0.10 + 300 * 1.25 + 200 * 2.00) / 1e6

    assert events.cost_usd(usage, rates) == pytest.approx(expected)


def test_a_null_write_premium_falls_back_to_the_input_rate() -> None:
    """True for Gemini: its implicit cache charges nothing extra to write."""
    rates = events.rates_for("gemini", "gemini-3.7-flash")
    assert rates.cache_write is None
    usage = Usage(input_tokens=1000, output_tokens=0, cache_write_tokens=1000)

    assert events.cost_usd(usage, rates) == pytest.approx(1000 * 0.75 / 1e6)


def test_overlapping_subsets_never_produce_a_negative_bucket() -> None:
    """Three vendors' arithmetic feeds these fields. A sum that overshoots its
    parent must cost more, not less, and must never go negative."""
    rates = events.rates_for("anthropic", "claude-haiku-4-5")
    usage = Usage(
        input_tokens=100, output_tokens=0, cached_tokens=80, cache_write_1h_tokens=80
    )
    buckets = usage.input_buckets

    assert buckets.total == 100
    # The 2x tier is claimed first, so the 0.1x read bucket absorbs the overlap.
    assert (buckets.uncached, buckets.cached, buckets.write_1h) == (0, 20, 80)
    assert events.cost_usd(usage, rates) == pytest.approx((20 * 0.10 + 80 * 2.00) / 1e6)


def test_cache_savings_is_signed() -> None:
    """A run that has only filled a cache has not saved anything yet, and saying
    it did would be the lie in the flattering direction."""
    rates = events.rates_for("anthropic", "claude-haiku-4-5")

    read = Usage(input_tokens=1000, output_tokens=500, cached_tokens=1000)
    # 1000 tokens that would have cost $1/Mtok cost $0.10/Mtok instead.
    assert events.cache_savings(read, rates) == pytest.approx(1000 * 0.90 / 1e6)

    wrote = Usage(input_tokens=1000, output_tokens=500, cache_write_tokens=1000)
    assert events.cache_savings(wrote, rates) == pytest.approx(-1000 * 0.25 / 1e6)

    nothing = Usage(input_tokens=1000, output_tokens=500)
    assert events.cache_savings(nothing, rates) == 0.0


def test_cache_savings_ignores_output(unpriced_table: None) -> None:
    """Caching cannot touch output, and an unpriced model has no answer at all."""
    assert events.cache_savings(HAIKU, events.rates_for("gemini", "gemini-3.6-flash")) is None


def test_unknown_models_cost_none() -> None:
    assert events.cost_usd(HAIKU, events.rates_for("openai", "not-a-model")) is None


# ---------------------------------------------------------------- records


def _haiku_response() -> ModelResponse:
    return anthropic.normalize(load("toolcalls/claude_weather_0"), latency_ms=812.456)


def test_call_record_shape() -> None:
    with events.start_trace("tr_rec"):
        events.advance_step()
        rec = events.call_record(_haiku_response())

    assert rec["trace_id"] == "tr_rec"
    assert rec["step"] == 1
    assert rec["event"] == "call"
    assert rec["provider"] == "anthropic"
    assert rec["model"] == "claude-haiku-4-5-20251001"
    assert rec["pricing_key"] == "claude-haiku-4-5"
    assert rec["finish_reason"] == "tool_calls"
    assert rec["latency_ms"] == 812.46
    assert rec["tokens"] == {
        "input": 591,
        "output": 59,
        "cached": 0,
        "cache_write": 0,
        "cache_write_1h": 0,
        "reasoning": 0,
        "total": 650,
    }
    assert rec["tool_calls"] == ["get_weather"]
    assert rec["malformed_tool_calls"] == []
    assert rec["cost_usd"] == pytest.approx(0.000886)


def test_record_carries_the_rates_it_was_priced_at() -> None:
    """So a trace re-read after a price change still shows what was billed."""
    rec = events.call_record(_haiku_response())
    assert rec["rates_usd_per_mtok"] == {
        "input": 1.00,
        "output": 5.00,
        "cached_input": 0.10,
        "cache_write": 1.25,
        "cache_write_1h": 2.00,
    }


def test_unpriced_call_records_null_cost_and_null_rates(unpriced_table: None) -> None:
    from bodol.providers import gemini

    rec = events.call_record(gemini.normalize(load("toolcalls/gemini_weather_0")))
    assert rec["cost_usd"] is None
    assert rec["rates_usd_per_mtok"]["input"] is None
    assert rec["tokens"]["input"] == 89, "tokens are still recorded"


def test_error_record_does_not_claim_zero_cost() -> None:
    with events.start_trace("tr_err"):
        rec = events.error_record(
            "openai", "gpt-5.6-luna", TimeoutError("read timeout"), latency_ms=40.0
        )
    assert rec["event"] == "error"
    assert rec["error_type"] == "TimeoutError"
    assert rec["cost_usd"] is None, "a timed-out request may still have been billed"
    assert rec["latency_ms"] == 40.0


# ---------------------------------------------------------------- sinks


def test_jsonl_sink_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tr_x.jsonl"
    with JsonlSink(path) as sink:
        sink.emit({"event": "call", "n": 1})
        sink.emit({"event": "call", "n": 2})

    assert read_trace(path) == [{"event": "call", "n": 1}, {"event": "call", "n": 2}]


def test_jsonl_sink_flushes_each_record(tmp_path: Path) -> None:
    """A run that dies mid-way must leave the completed steps on disk."""
    path = tmp_path / "tr_y.jsonl"
    sink = JsonlSink(path)
    sink.emit({"n": 1})
    assert len(path.read_text().splitlines()) == 1, "readable before close()"
    sink.close()


def test_jsonl_sink_survives_unserializable_values(tmp_path: Path) -> None:
    path = tmp_path / "tr_z.jsonl"
    with JsonlSink(path) as sink:
        sink.emit({"weird": object()})
    assert "weird" in json.loads(path.read_text())


def test_sinks_satisfy_the_protocol() -> None:
    assert isinstance(NullSink(), Sink)
    assert isinstance(MemorySink(), Sink)


# ---------------------------------------------------------------- middleware


class FakeProvider:
    name = "anthropic"
    model = "claude-haiku-4-5"

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.fail = fail
        self.closed = False
        self.calls = 0

    async def generate(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        tools: Sequence[ToolSpec] = (),
        max_tokens: int = 4096,
    ) -> ModelResponse:
        self.calls += 1
        if self.fail:
            raise self.fail
        return _haiku_response()

    async def aclose(self) -> None:
        self.closed = True


def _msgs() -> list[Message]:
    return [Message("user", (TextBlock("weather?"),))]


async def test_middleware_is_transparent_and_emits_one_record() -> None:
    sink = MemorySink()
    inner = FakeProvider()
    traced = TracedProvider(inner, sink)

    with events.start_trace("tr_mw"):
        events.advance_step()
        response = await traced.generate(_msgs())

    assert response.finish_reason.value == "tool_calls", "response passes through untouched"
    assert inner.calls == 1
    (rec,) = sink.records
    assert (rec["trace_id"], rec["step"], rec["event"]) == ("tr_mw", 1, "call")
    assert traced.name == "anthropic"
    assert traced.model == "claude-haiku-4-5"


async def test_middleware_records_failures_then_reraises() -> None:
    sink = MemorySink()
    traced = TracedProvider(FakeProvider(fail=RuntimeError("boom")), sink)

    with events.start_trace("tr_fail"), pytest.raises(RuntimeError, match="boom"):
        await traced.generate(_msgs())

    (rec,) = sink.records
    assert rec["event"] == "error"
    assert rec["error_type"] == "RuntimeError"
    assert rec["latency_ms"] is not None


async def test_aclose_closes_both_layers() -> None:
    inner = FakeProvider()
    sink = MemorySink()
    traced = TracedProvider(inner, sink)
    await traced.aclose()
    assert inner.closed


async def test_a_whole_run_lands_in_one_trace_file(tmp_path: Path) -> None:
    """End to end: three steps, one file, costs summable."""
    with events.start_trace("tr_run") as tid, JsonlSink.for_trace(tid, directory=tmp_path) as sink:
        traced = TracedProvider(FakeProvider(), sink)
        for _ in range(3):
            events.advance_step()
            await traced.generate(_msgs())

    records: list[Mapping[str, Any]] = list(read_trace(tmp_path / "tr_run.jsonl"))
    assert [r["step"] for r in records] == [1, 2, 3]
    total = sum(r["cost_usd"] for r in records)
    assert total == pytest.approx(0.000886 * 3)
