"""Command line interface.

`aca run "..."` is the one people use. Everything else exists to support it:
`index` fills the retrieval store, `memory` lets you look at what the agent has
decided to remember, `trace` replays a finished run, and `migrate` sets up
postgres.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

from agent import __version__
from agent.config import get_settings
from agent.observability.events import Event, EventBus, EventType
from agent.observability.tracing import read_trace
from agent.types import RunStatus

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="An autonomous coding agent with long-term memory and retrieval.",
)
memory_app = typer.Typer(no_args_is_help=True, help="Inspect long-term memory.")
app.add_typer(memory_app, name="memory")

console = Console()

STATUS_STYLE = {
    RunStatus.SUCCEEDED: "bold green",
    RunStatus.FAILED: "bold red",
    RunStatus.ABORTED: "bold yellow",
}

EVENT_STYLE = {
    EventType.PLAN_CREATED: ("plan", "cyan"),
    EventType.PLAN_REVISED: ("replan", "yellow"),
    EventType.STEP_STARTED: ("step", "magenta"),
    EventType.THOUGHT: ("think", "dim white"),
    EventType.TOOL_CALLED: ("tool", "blue"),
    EventType.TOOL_RESULT: ("result", "green"),
    EventType.OBSERVATION: ("observe", "cyan"),
    EventType.WARNING: ("warn", "yellow"),
    EventType.RETRIEVAL: ("recall", "dim cyan"),
}


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False, rich_tracebacks=True)],
    )


def _printer(verbose: bool):
    def handle(event: Event) -> None:
        style = EVENT_STYLE.get(event.type)
        if style is None:
            return
        label, colour = style
        data = event.data
        if event.type is EventType.PLAN_CREATED or event.type is EventType.PLAN_REVISED:
            steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(data.get("steps", []), 1))
            console.print(f"[{colour}]{label:>7}[/] the plan:\n{steps}")
        elif event.type is EventType.STEP_STARTED:
            console.print(
                f"[{colour}]{label:>7}[/] {data.get('iteration')}. {data.get('step')}"
            )
        elif event.type is EventType.TOOL_CALLED:
            args = json.dumps(data.get("arguments", {}))[:110]
            console.print(f"[{colour}]{label:>7}[/] {data.get('tool')} {args}")
        elif event.type is EventType.TOOL_RESULT:
            mark = "ok" if data.get("ok") else "failed"
            console.print(
                f"[{colour}]{label:>7}[/] {data.get('tool')} {mark} "
                f"({data.get('duration_ms')}ms)"
            )
        elif event.type is EventType.OBSERVATION:
            console.print(f"[{colour}]{label:>7}[/] {data.get('summary')}")
        elif event.type is EventType.WARNING:
            console.print(f"[{colour}]{label:>7}[/] {data.get('message')}")
        elif verbose:
            console.print(f"[{colour}]{label:>7}[/] {json.dumps(data)[:160]}")

    return handle


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print the version and exit"),
) -> None:
    if version:
        console.print(f"aca {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())
        raise typer.Exit()


@app.command()
def run(
    task: str = typer.Argument(..., help="What the agent should do"),
    workspace: str = typer.Option(None, "--workspace", "-w", help="Directory to work in"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Run without long-term memory"),
    no_retrieval: bool = typer.Option(False, "--no-retrieval", help="Run without the RAG index"),
    as_json: bool = typer.Option(False, "--json", help="Print the result as JSON"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run a task end to end."""
    _setup_logging(verbose)
    settings = get_settings()

    from agent.loop import Agent
    from agent.memory.store import build_memory_store
    from agent.rag.retriever import build_retriever

    memory = None if no_memory else build_memory_store(settings)
    retriever = None if no_retrieval else build_retriever(settings)

    bus = EventBus()
    if not as_json:
        bus.subscribe(_printer(verbose))
        console.print(Panel(task, title="task", border_style="blue"))

    agent = Agent(settings=settings, memory=memory, retriever=retriever)
    result = agent.run(task, workspace=workspace, bus=bus)

    if as_json:
        console.print_json(result.model_dump_json())
        raise typer.Exit(0 if result.status is RunStatus.SUCCEEDED else 1)

    style = STATUS_STYLE.get(result.status, "white")
    console.print(
        Panel(
            result.summary or "(no summary)",
            title=f"[{style}]{result.status.value}[/] in {result.duration_s}s "
            f"across {result.iterations} iteration(s) - ${result.usage.usd:.4f}",
            border_style="green" if result.status is RunStatus.SUCCEEDED else "red",
        )
    )
    if result.artifacts:
        console.print("artifacts: " + ", ".join(result.artifacts))
    raise typer.Exit(0 if result.status is RunStatus.SUCCEEDED else 1)


@app.command()
def demo(
    workspace: str = typer.Option(None, "--workspace", "-w"),
) -> None:
    """Run a scripted task with no API key, to check the plumbing works."""
    _setup_logging(False)
    from agent.llm.scripted import ScriptedClient
    from agent.loop import Agent
    from agent.memory.embeddings import HashingEmbedder
    from agent.memory.store import InMemoryStore

    bus = EventBus()
    bus.subscribe(_printer(False))
    agent = Agent(
        llm=ScriptedClient.demo(), memory=InMemoryStore(HashingEmbedder(256)), retriever=None
    )
    result = agent.run(
        "Write a fizzbuzz helper in solution.py and verify it runs", workspace=workspace, bus=bus
    )
    console.print(Panel(result.summary, title=result.status.value, border_style="green"))


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory to index"),
) -> None:
    """Index a codebase or a document set into the retrieval store."""
    _setup_logging(False)
    from agent.rag.indexer import build_indexer

    indexer = build_indexer()
    target = Path(path).expanduser()
    with console.status(f"indexing {target}..."):
        report = indexer.index_directory(target) if target.is_dir() else indexer.index_file(target)
    console.print(
        f"indexed [green]{report.indexed}[/] document(s) into "
        f"[green]{report.chunks}[/] chunk(s), skipped {report.skipped} unchanged"
    )
    for err in report.errors[:10]:
        console.print(f"[yellow]warn[/] {err}")


@app.command()
def search(
    query: str = typer.Argument(..., help="What to look for"),
    k: int = typer.Option(5, "-k", help="How many chunks to return"),
) -> None:
    """Query the retrieval store directly - useful for debugging bad answers."""
    _setup_logging(False)
    from agent.rag.retriever import build_retriever

    chunks = build_retriever().search(query, k=k)
    if not chunks:
        console.print("[yellow]nothing passed the relevance floor[/]")
        raise typer.Exit(1)
    for chunk in chunks:
        console.print(
            Panel(
                chunk.text[:800],
                title=f"{chunk.citation()}  score {chunk.score:.3f}",
                border_style="cyan",
            )
        )


@memory_app.command("list")
def memory_list(limit: int = typer.Option(20, "-n")) -> None:
    """Show what the agent has chosen to remember."""
    from agent.memory.store import build_memory_store

    table = Table("kind", "text", "written", box=None)
    for record in build_memory_store().recent(limit):
        table.add_row(record.kind, record.text[:100], f"{record.created_at:.0f}")
    console.print(table)


@memory_app.command("add")
def memory_add(
    text: str = typer.Argument(...),
    kind: str = typer.Option("fact", "--kind"),
) -> None:
    """Seed a memory by hand."""
    from agent.memory.store import build_memory_store

    console.print(build_memory_store().write(text, kind=kind))


@memory_app.command("stats")
def memory_stats() -> None:
    """Counts by kind, and which backend is actually in use."""
    from agent.memory.store import build_memory_store

    console.print_json(json.dumps(build_memory_store().stats()))


@app.command()
def trace(run_id: str = typer.Argument(..., help="Run id, or a path to a .jsonl trace")) -> None:
    """Replay a finished run from its trace file."""
    settings = get_settings()
    path = Path(run_id)
    if not path.exists():
        path = settings.trace_dir / f"{run_id}.jsonl"
    if not path.exists():
        console.print(f"[red]no trace at {path}[/]")
        raise typer.Exit(1)

    table = Table("t+", "event", "detail", box=None)
    events = read_trace(path)
    start = events[0]["ts"] if events else 0
    for event in events:
        detail = json.dumps(event.get("data", {}))
        table.add_row(f"{event['ts'] - start:6.2f}s", event["type"], detail[:120])
    console.print(table)


@app.command()
def migrate() -> None:
    """Create the postgres schema (safe to run repeatedly)."""
    from agent.memory import db

    settings = get_settings()
    applied = db.migrate(settings.database_url, settings.embedding_dim)
    console.print(f"applied [green]{len(applied)}[/] statement(s)")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Start the dashboard and HTTP API."""
    import uvicorn

    uvicorn.run("server.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
