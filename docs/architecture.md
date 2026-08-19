# Architecture

Notes on why the pieces are shaped the way they are. The README covers what it
does; this covers the decisions I would otherwise have to re-explain.

## One entry point

`Agent.run(task)` is the only way in. The CLI calls it, the HTTP API calls it,
the eval harness calls it. Every dependency — the model client, the memory store,
the retriever, the tool registry — is an optional constructor argument that falls
back to a real implementation.

That single decision is what makes the loop testable. `tests/test_loop.py` drives
budget exhaustion, replanning, and the repeated-call guard with a queue of canned
responses, no network, and no database, in under two seconds.

## Why a separate observer call

The obvious design has one model call per turn: give it the tools, let it decide
when it is finished. It fails in a specific way — a context window that has just
spent six turns doing work is heavily primed to conclude the work is done, and it
will read a traceback as "nearly working".

So `observe` is its own call with its own prompt. It sees the current step and
the raw tool output, and nothing else — not the agent's commentary, not the
history of how hard the run has been. It grades on evidence: exit code, expected
output present, no traceback.

Two guards on top of the model's judgement, in `nodes.py`:

- `step_satisfied` is ANDed with `result.ok`. A failed tool can never satisfy a
  step, no matter what the observer says.
- If the observer returns unparseable output, the fallback is the tool's exit
  status, not the observer's prose.

It costs an extra call per turn. It is the cheapest reliability I have found here.

## Why LangGraph, and where it stops

The graph gives three things worth having: nodes as pure functions of state,
conditional edges as first-class objects, and a recursion limit that is enforced
rather than remembered.

What it does not give is a policy for stopping, so that lives in the routers in
`graph/builder.py`. Every edge out of `act` and `observe` goes through one, and
budget checks happen there and nowhere else. If a run ends, exactly one router
call decided it — which means the answer to "why did this stop?" is always in one
place.

## State

`AgentState` is a flat `TypedDict`. Nested models inside it would serialise more
prettily; flat survives being printed at 3am. Nodes return only the keys they
changed and LangGraph merges, so no node can accidentally clobber a sibling's
work by returning a stale copy of the whole state.

One sharp edge: `plan` is a mutable pydantic model held in the state, and `act`
and `observe` mutate step statuses in place. That is deliberate — reconstructing
the plan on every turn to change one enum was noise — but it does mean the plan
object is shared, so treat it as owned by those two nodes only.

## The message conversation

`act` maintains a real Anthropic tool-use conversation: assistant turn with
`tool_use` blocks, user turn with matching `tool_result` blocks. Every tool_use
must get a result block, including the ones the repeat-guard refuses — the
refusal is delivered *as* a tool result with `is_error: true`, so the model reads
it as feedback in the normal way instead of the request 400ing on a missing
block.

`plan`, `observe`, and `summarise` are one-shot calls with no history. They do not
need it, and keeping them stateless means a bad turn cannot poison them.

## Storage

Postgres holds three things: `memories`, `documents` + `chunks`, and `runs`.
Vectors go over the wire as literals cast with `::vector` rather than through an
adapter — one fewer dependency, and the SQL in the logs is the SQL that ran.

Every store has an in-memory twin implementing the same protocol. They are not
mocks; they are the fallback path that a fresh clone and CI actually take, which
is why they are exercised by the same tests.

The vector column width cannot be a bind parameter, so `schema.sql` is templated
with the embedding dimension at migration time. Change `ACA_EMBEDDING_DIM` after
you have indexed something and you will need to re-migrate and re-index. There is
no dimension-mismatch detection yet; that is a real gap.

## Events

One bus per run. Subscribers are called synchronously and their exceptions are
swallowed and logged, because a dashboard that disconnected mid-run must not take
the run down with it.

`EventBus.__bool__` returns `True` explicitly. Without it, `__len__` makes an
empty bus falsy, and `bus = bus or EventBus(run_id)` silently discards the bus
the caller passed in — which is exactly the bug that shipped once and cost an
afternoon of "why is the dashboard empty".

## Costing

`llm/pricing.py` holds per-model rates and falls back to the mid-tier rate for an
unknown model rather than reporting zero. A run that claims it cost $0.00 is
worse than one that admits to an approximation, because you will believe it.
