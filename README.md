# aca — an autonomous coding agent

[![ci](https://github.com/yuvrajnode/Autonomous-AI-Coding-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/yuvrajnode/Autonomous-AI-Coding-Agent/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A coding agent that plans a task, works through it with tools, and checks its own
output before saying it is done. It keeps long-term memory and a retrieval index
in Postgres (pgvector), writes a trace of every decision, and ships with an eval
harness so "it works" is a number rather than an impression.

There is a CLI, an HTTP API, and a dashboard that streams a run as it happens.

Here is `aca demo` — the offline scripted task, so you can see the shape of a run
without spending a token:

```
$ aca demo

   plan the plan:
  1. Inspect the workspace to see what already exists
  2. Write the module the task asks for
  3. Run the tests and confirm they pass
   step 1. Inspect the workspace to see what already exists
   tool list_dir {"path": "."}
 result list_dir ok (0ms)
observe Tool call returned cleanly; the step looks satisfied.
   step 2. Write the module the task asks for
   tool write_file {"path": "solution.py", "content": "def fizzbuzz(n: int) -> str: ...
 result write_file ok (0ms)
observe Tool call returned cleanly; the step looks satisfied.
   step 3. Run the tests and confirm they pass
   tool run_python {"code": "from solution import fizzbuzz; print(fizzbuzz(15))"}
 result run_python ok (21ms)
observe Tool call returned cleanly; the step looks satisfied.
╭─ succeeded ──────────────────────────────────────────────────────────────────╮
│ Created solution.py with a fizzbuzz(n) helper covering the three             │
│ divisibility cases, then executed it with run_python: fizzbuzz(15) printed   │
│ FizzBuzz and the process exited 0.                                           │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## Why this exists

Most agent demos are a `while` loop around a chat completion. That works until
the model decides it is finished while the tests are red, or spends nine turns
calling `ls` on the same directory, or reports editing a file that was never
created.

This one is built around those three failure modes:

| failure | what stops it |
| --- | --- |
| declaring success on broken work | a **separate observer call** that only sees the step and the raw tool output, and grades it on evidence |
| looping on the same action | tool calls are **fingerprinted**; the third identical call is refused with an error the model can read |
| inventing files, flags, APIs | retrieval has a **relevance floor** and returns nothing rather than something weak; the eval harness fails any run whose summary names a file that does not exist |

Everything else — the memory, the tracing, the dashboard — exists to make those
three things observable.

---

## The loop

```
                    ┌──────────┐
                    │  ground  │  memory + retrieval, once, before planning
                    └────┬─────┘
                         ▼
                    ┌──────────┐
        ┌──────────▶│   plan   │  2-6 concrete steps, last one is verification
        │           └────┬─────┘
        │                ▼
        │           ┌──────────┐
        │      ┌───▶│   act    │  one tool call per turn, in a sandbox
        │      │    └────┬─────┘
        │      │         ▼
        │      │    ┌──────────┐
        │      └────│ observe  │  did that actually work? evidence only
        │           └────┬─────┘
        │   replan       │  finish / budget hit
        └────────────────┼──────────────┐
                         ▼              ▼
                    ┌──────────┐   ┌──────────┐
                    │  halt    │──▶│summarise │──▶ done
                    └──────────┘   └──────────┘
```

Built on [LangGraph](https://github.com/langchain-ai/langgraph). Every edge out of
`act` and `observe` runs through a router, and the routers are the only place that
decides to stop — so there is always exactly one answer to "why did this run end".

Budgets are hard: an iteration ceiling, a dollar ceiling, and a cap on how many
times the plan may be rewritten. See [`agent/graph/builder.py`](agent/graph/builder.py).

---

## Quick start

No API key needed for this part — a scripted client stands in for the model so a
fresh clone actually runs.

```bash
git clone https://github.com/yuvrajnode/Autonomous-AI-Coding-Agent.git
cd Autonomous-AI-Coding-Agent
make install
.venv/bin/aca demo
```

That plans a task, writes a file, executes it, and prints the summary. Then open
the dashboard:

```bash
make serve   # http://localhost:8000
```

### With a real model

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY, and VOYAGE_API_KEY if you want real embeddings
make db-up          # postgres + pgvector on :5433
make migrate
aca index ~/code/some-project        # optional: give it something to cite
aca run "Fix the failing test in tests/test_parser.py" --workspace ~/code/some-project
```

Without Postgres running, memory and the index fall back to in-memory
implementations and the agent tells you so on startup. Nothing breaks; nothing
survives the process either.

---

## The dashboard

`make serve` gives you a console at `http://localhost:8000`:

- **Runs** — start a task, watch the timeline fill in over a websocket. Plan
  progress, tool calls with their arguments and output, per-run token and cost
  counters, and the final summary.
- **Memory** — everything the agent decided was worth keeping, searchable, and
  deletable when it turns out to be wrong.
- **Knowledge** — index a directory, then query it and see the chunks with the
  `path:lines` they came from and their retrieval score.
- **Tools** — the exact schema list the model is offered. Nothing else is
  reachable from the loop.

Dark and light, keyboard operable, and it degrades to a single column on a phone.
No build step — it is one HTML file, one stylesheet, one ES module.

---

## Tools

Fourteen, all defined the same way: a python function plus a pydantic model that
describes its arguments. The model is the single source of truth — it generates
the JSON schema sent to the provider *and* validates whatever comes back, so a
hallucinated argument becomes an ordinary error message the agent can correct
instead of a stack trace.

| group | tools |
| --- | --- |
| files | `list_dir` `read_file` `write_file` `edit_file` `append_file` |
| search | `grep` `find_files` |
| execution | `run_command` `run_python` `run_tests` |
| knowledge | `search_knowledge` `recall` `remember` |
| control | `submit_result` |

### The sandbox

Every path the model produces goes through `Workspace.resolve`, which rejects
absolute paths, `..` traversal, and symlinks that point outside the workspace.
Commands never touch a shell — they are split with `shlex` and executed directly,
and the executable has to be on an allowlist. That removes `; rm -rf ~` as a
category rather than as a string match.

This is the one part of the codebase where a mistake turns a coding agent into a
remote shell, so it is small, separate, and has the most hostile tests in the
suite ([`tests/test_sandbox.py`](tests/test_sandbox.py)).

---

## Memory and retrieval

**Long-term memory** holds four kinds of record — `lesson`, `fact`, `preference`,
`failure` — and nothing else. Transcripts are not memory; they are traces, and
they live on disk. Recall ranks on cosine similarity with a small recency decay
and a bonus for memories that keep proving useful, so a note that gets used every
run floats above one written once and never touched again.

**Retrieval** is hybrid. Postgres runs cosine over pgvector *and* full-text over a
stored `tsvector`, and the two rankings are normalised and blended. Neither half
is enough alone: embeddings generalise over phrasing but miss exact identifiers,
full-text nails `ToolRegistry.invoke` but falls over on a paraphrase.

Two details that matter more than the ranking:

- **Chunks know their line range.** Splitting happens at real boundaries — a blank
  line, a top-level `def`, a markdown heading — so a chunk is readable, and it
  comes back labelled `path:12-48` for the agent to cite and for you to go check.
- **There is a floor.** Below it the retriever returns *nothing*, and the tool
  tells the agent to go read the file instead of guessing. Handing a model three
  weak chunks and letting it treat them as ground truth is precisely how an agent
  invents an API that does not exist.

Embeddings come from Voyage when a key is set. Without one it uses a hashed
bag-of-ngrams: deterministic, no network, no dependencies, and honestly lexical
rather than semantic — it will miss a paraphrase. It exists so CI and a fresh
clone work, and it is never chosen when a real provider is configured.

---

## Observability

Every run emits one stream of typed events. The trace writer, the metrics
collector, and the dashboard websocket are all just subscribers, so the loop
never knows who is watching.

- **Traces** — one newline-delimited JSON file per run under `traces/`, flushed on
  every write. If a run hangs, `tail -f` shows you the tool call it is stuck in.
  Replay one with `aca trace <run_id>`.
- **Metrics** — tool failure rate, retrieval hit rate, prompt-cache hit rate,
  replans, tokens, and cost per run.
- **Cost** — the system prompt and tool schemas are marked with `cache_control`.
  They are identical on every turn, so the cache read is nearly free and the input
  bill on a long run drops by roughly an order of magnitude. The dashboard shows
  the cache hit rate so you can tell when it stops working.

---

## Evals

```bash
make eval                                             # against a live model
python -m evals.runner --suite evals/tasks/core.yaml --scripted   # harness only
```

Tasks are YAML. Each seeds its own workspace and is graded by checks that a
machine can settle — not a vibe score:

```yaml
- id: fix-off-by-one
  prompt: chunk.py drops the final partial chunk. Fix it and prove it.
  files:
    chunk.py: |
      def chunks(items, size):
          ...
  checks:
    - type: tests_pass
    - type: grounding
    - type: within_iterations
      max: 10
```

The headline number is the **completion rate**: the share of tasks where every
check passed. Not partial credit — a task where the tests still fail is a failed
task.

The grader I get the most out of is `grounding`. It pulls every file-path-shaped
token out of the agent's own summary and checks the file exists. An agent that
reports editing `src/utils/retry.py` when no such file was ever created is
hallucinating, and unlike a quality score, that is cheap to detect and impossible
to argue with.

Reports land in `eval_reports/` as JSON and markdown. The markdown is shaped like
this — run the suite yourself for real numbers, they depend on the model, the
temperature, and the day:

```
| task             | result | steps | time | failing checks |
| ---------------- | ------ | ----- | ---- | -------------- |
| fizzbuzz         | ...    | ...   | ...  | ...            |
| fix-off-by-one   | ...    | ...   | ...  | ...            |
| add-a-test       | ...    | ...   | ...  | ...            |
| refuse-to-invent | ...    | ...   | ...  | ...            |
```

`--scripted` runs the harness against a canned client. It exercises the graders
and the report, not the model — a green scripted run says the plumbing works, not
that the agent is good. More detail in [docs/evals.md](docs/evals.md).

---

## Configuration

Everything is read from the environment or a local `.env` (see
[`.env.example`](.env.example)) exactly once, into a `Settings` object that gets
passed down explicitly. Tests build one directly and never touch `os.environ`.

| variable | default | what it does |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | – | unset means the scripted client, not a crash |
| `ACA_MODEL` | `claude-sonnet-5` | the acting and observing model |
| `ACA_PLANNER_MODEL` | `claude-opus-5` | planning and replanning |
| `ACA_DATABASE_URL` | `postgresql://aca:aca@localhost:5433/aca` | memory + index |
| `ACA_EMBEDDING_PROVIDER` | `voyage` | or `hashing` for offline |
| `ACA_MAX_ITERATIONS` | `24` | hard ceiling on act/observe turns |
| `ACA_USD_BUDGET` | `1.50` | hard ceiling on spend per run |
| `ACA_RETRIEVAL_MIN_SCORE` | `0.25` | the relevance floor |
| `ACA_WORKSPACE_ROOT` | `./workspaces` | sandbox root |

---

## Layout

```
agent/
  graph/          the plan-act-observe state machine (LangGraph)
  llm/            provider client, pricing, and a scripted stand-in
  memory/         embeddings, pgvector store, schema
  rag/            chunking, indexing, hybrid retrieval
  tools/          registry, sandbox, and the tools themselves
  observability/  event bus, traces, metrics
  cli.py          aca run / index / search / memory / trace / serve
  loop.py         Agent.run() - the one entry point everything else uses
server/           FastAPI app, run manager, and the dashboard
evals/            suites, graders, report writer
tests/            unit, API, and pgvector integration tests
```

`Agent.run(task)` is the only entry point the CLI, the API, and the eval harness
use, and every dependency is an optional constructor argument. That is what makes
the whole thing testable — a scripted client and an in-memory store run the entire
loop in milliseconds, with no network and no database.

---

## Development

```bash
make install   # venv + dev dependencies + editable install
make test      # pytest
make lint      # ruff + mypy
make db-up     # postgres with pgvector, on :5433
make eval      # run the core suite
```

The suite runs without an API key and without Postgres — the pgvector tests skip
themselves when nothing is listening on `:5433`, and CI brings the real image up
so they run for real there.

---

## Known limitations

- **One workspace, one process.** Runs are threads in a single process. Fine for
  a handful at a time, wrong for anything shared — there is no queue, no
  persistence of in-flight runs, and a restart loses them.
- **The sandbox is a sandbox, not a jail.** Path confinement plus an executable
  allowlist stops accidents and casual mistakes. It is not a container, and I
  would not point this at a repo I did not trust without one.
- **Run history is in memory.** The `runs` table exists in the schema; the API
  still serves history from a bounded in-process deque. Restarting the server
  loses the list, though the traces on disk survive.
- **The eval suite is small.** Four tasks is enough to catch a regression in the
  loop, nowhere near enough to claim a benchmark result.
- **Chunking is boundary-aware, not language-aware.** Real tree-sitter parsing
  would beat the regex heuristics, especially on languages other than Python.

## License

MIT — see [LICENSE](LICENSE).
