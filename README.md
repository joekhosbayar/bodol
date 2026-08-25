# bodol

A model-agnostic agent harness, built layer by layer in Python: provider
adapters, a tool registry, context compaction, an agent loop with hard
limits, and per-call cost telemetry — driven from a single CLI command.

Works with Gemini, OpenAI, and Anthropic behind one interface. Adding a
model is a data change, not a code change.

## Quickstart

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone git@github.com:joekhosbayar/bodol.git
cd bodol
uv sync
cp .env.example .env   # add a key: GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY
```

```sh
uv run bodol run "how many python files are in this repo, and which is largest?"

uv run bodol run "..." --provider anthropic:claude-haiku-4-5
uv run bodol run "..." --json --max-cost 0.05 --max-steps 8
```

The default provider is `gemini:gemini-3.7-flash`; override per run with
`--provider` or globally with `BODOL_PROVIDER`.

## What a run does

`bodol run "task"` starts an agent loop rooted in the current directory:

1. Loads a versioned system prompt from `prompts/<version>.md`
   (`--system v2` loads `v2.md`; an unknown version exits 2 and lists what
   exists).
2. Registers three read-only tools — `calculator`, `file_read`, `grep` —
   sandboxed to the working tree. Paths are resolved after symlink
   expansion and containment-checked, so `../` and outward symlinks are
   refused. Dotfiles are readable, `.env` included: a coding agent that
   cannot see `.gitignore` is crippled, and the sandbox already assumes you
   trust the model with the repo you invoked it in.
3. Sizes the context budget at 60% of the model's known context window.
   The check runs against calls that already happened, so the remaining 40%
   pays for the turn that trips it plus the summarization call. A model
   with no verified window gets no budget and a note on stderr, rather than
   a budget invented from a guess.
4. Enforces step, cost, and time limits *before* every model call, so a run
   never overspends by making one more request.
5. Writes one JSONL trace record per call — tokens, cost, latency, and the
   rates it was priced at — flushed per record, so a run that dies at step
   9 still leaves nine rows behind.

## Output contract

- The answer goes to **stdout**; stop reason, cost, and trace path go to
  **stderr**. `bodol run ... > out.txt` leaves the answer in the file.
- Exit code **0** means the run happened, whatever the outcome (including
  hitting a limit or a mid-run provider failure — the stop reason says
  which). Exit code **2** means the run never started: bad provider spec,
  missing API key, unknown prompt.
- `--json` emits the whole `RunResult`, transcript included.

## Traces

Every run writes `traces/<trace_id>.jsonl`, one record per model call:

```json
{
  "event": "call",
  "step": 1,
  "model": "claude-haiku-4-5-20251001",
  "tokens": { "input": 591, "output": 59, "cached": 0, "reasoning": 0, "total": 650 },
  "cost_usd": 0.000886,
  "rates_usd_per_mtok": { "input": 1.0, "output": 5.0, "cached_input": 0.1 },
  "finish_reason": "tool_calls",
  "latency_ms": 812.4
}
```

Two rules the telemetry layer holds to:

- **`cost_usd` is `null`, never `0`, when a model has no published rates.**
  A silent zero is a lie; a null is visible.
- **Every record carries the rates it was priced at.** A trace re-costed
  months later at current prices is not what was actually billed.

Rate cards (`bodol/data/pricing.yaml`) and context windows
(`bodol/data/context_windows.yaml`) are data, verified against the vendors'
own pricing pages. Both files carry a VERIFY-BEFORE-TRUSTING header and the
URLs to check against — prices change, and Gemini's current Flash rates
double on 2027-01-01.

## How it's put together

| Layer | Module | Job |
| --- | --- | --- |
| Providers | `bodol/providers/` | A port (`base.py`) plus one adapter per vendor. Normalizes messages, tool calls, finish reasons, and usage. |
| Tools | `bodol/tools/` | Explicit `ToolSpec` registration; parallel dispatch with per-call error isolation. |
| Context | `bodol/context/` | Budget policy and summarization-based compaction; never splits a tool-use/tool-result pair. |
| Agent | `bodol/agent/` | The step loop: limits, unknown-tool handling, `RunResult` with a stop reason. |
| Telemetry | `bodol/telemetry/` | Ambient trace context, JSONL sink, per-call pricing. |
| CLI | `bodol/cli.py` | Parse, delegate, print. No logic lives here. |

The piece worth reading first is the docstring at the top of
`bodol/providers/base.py`: the token containment contract. The three
vendors disagree about what is nested inside what — Gemini excludes thought
tokens from its output total, Anthropic excludes cache buckets from its
input total — so `Usage` fixes one convention and each adapter does the
arithmetic to satisfy it.

## Development

```sh
uv run pytest           # 223 tests, fully offline — nothing reaches a provider
uv run ruff check .
uv run mypy bodol tests # strict
```

Provider tests run against recorded API payloads in `tests/fixtures/`, with
HTTP mocked by respx. CLI tests stub the agent; agent tests stub the
provider. Every layer is tested against the seam below it.

## Status

`run` works. `eval`, `compare`, `cost`, and `replay` are stubs that exit 2:
the eval suite format is undesigned, and the trace rollups are unbuilt.
Known gaps, stated plainly: no streaming, no `--no-trace` (the sink is
built before the adapter exists, so there is no seam for the flag), and the
built-in tools are read-only by design.
