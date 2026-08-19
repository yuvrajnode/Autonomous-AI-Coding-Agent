"""Memory and retrieval tools.

These are the two tools that keep the agent honest. `search_knowledge` returns
chunks with a source label, and the system prompt requires the agent to cite
that label when it states a fact about the codebase or the docs. Anything it
cannot cite it has to verify with `read_file` or admit it does not know.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.errors import ToolError
from agent.tools.registry import ToolContext, registry


class SearchKnowledgeArgs(BaseModel):
    query: str = Field(description="What you want to know, phrased as a question or a topic")
    k: int = Field(default=5, ge=1, le=12, description="How many chunks to return")


@registry.register(SearchKnowledgeArgs)
def search_knowledge(ctx: ToolContext, args: SearchKnowledgeArgs) -> str:
    """Semantic search over the indexed documents and source files.

    Every result is prefixed with `[source]`. Cite that source when you rely on
    it. If nothing comes back above the relevance floor, say so instead of
    filling the gap from memory.
    """
    if ctx.retriever is None:
        raise ToolError("no knowledge index is attached to this run")
    chunks = ctx.retriever.search(args.query, k=args.k)
    if not chunks:
        return (
            f"nothing in the index passed the relevance floor for {args.query!r}. "
            "Do not guess - read the files directly or tell the user it is unknown."
        )
    blocks = [f"[{c.citation()}] (score {c.score:.2f})\n{c.text.strip()}" for c in chunks]
    return "\n\n---\n\n".join(blocks)


class RecallArgs(BaseModel):
    query: str = Field(description="What to look for in past runs")
    k: int = Field(default=5, ge=1, le=10, description="How many memories to return")


@registry.register(RecallArgs)
def recall(ctx: ToolContext, args: RecallArgs) -> str:
    """Search long-term memory for what happened on earlier runs.

    Worth a call at the start of anything that smells familiar - a build command
    that needed a flag, a test that is flaky, a directory layout you already
    mapped once.
    """
    if ctx.memory is None:
        raise ToolError("long-term memory is not available on this run")
    records = ctx.memory.search(args.query, k=args.k)
    if not records:
        return f"no memories matched {args.query!r}"
    return "\n\n".join(f"[{r.kind} - {r.score:.2f}] {r.text.strip()}" for r in records)


class RememberArgs(BaseModel):
    text: str = Field(description="The lesson, stated so it is useful without this run's context")
    kind: str = Field(default="lesson", description="One of: lesson, fact, preference, failure")


@registry.register(RememberArgs, mutating=True)
def remember(ctx: ToolContext, args: RememberArgs) -> str:
    """Store something worth carrying into future runs.

    Write the durable part only. "The test suite needs `-p no:randomly`" is
    useful; "I ran the tests" is not.
    """
    if ctx.memory is None:
        raise ToolError("long-term memory is not available on this run")
    if len(args.text.strip()) < 12:
        raise ToolError("that memory is too thin to be useful later; be specific")
    record_id = ctx.memory.write(
        text=args.text.strip(), kind=args.kind, metadata={"run_id": ctx.run_id}
    )
    return f"stored memory {record_id}"


class SubmitArgs(BaseModel):
    summary: str = Field(description="What you did and how you verified it")
    artifacts: list[str] = Field(
        default_factory=list, description="Workspace-relative paths you created or changed"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="How sure you are the task is actually done"
    )


@registry.register(SubmitArgs)
def submit_result(ctx: ToolContext, args: SubmitArgs) -> str:
    """Declare the task finished.

    Only call this after the verification step has actually passed. State what
    you ran and what it printed - "should work" is not a result.
    """
    missing = [p for p in args.artifacts if not ctx.workspace.exists(p)]
    if missing:
        raise ToolError(
            f"these artifacts do not exist in the workspace: {', '.join(missing)}. "
            "Fix the paths or create the files before submitting."
        )
    ctx.scratch["submitted"] = {
        "summary": args.summary,
        "artifacts": args.artifacts,
        "confidence": args.confidence,
    }
    return "result recorded"
