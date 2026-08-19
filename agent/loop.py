"""The façade.

`Agent.run(task)` is the only entry point the CLI, the HTTP API and the eval
harness use. It builds a workspace, an event bus, a trace file and a metrics
collector for the run, executes the graph, and hands back a `RunResult`.

Dependencies are all optional constructor arguments. That is what makes the
whole thing testable: pass a scripted client and an in-memory store and the
entire loop runs in milliseconds with no network and no database.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, cast

from agent.config import Settings, get_settings
from agent.graph.builder import build_graph
from agent.graph.nodes import Deps
from agent.graph.state import AgentState, initial_state
from agent.llm.base import LLMClient, build_client
from agent.memory.store import MemoryStore
from agent.observability.events import EventBus, EventType
from agent.observability.metrics import RunMetrics
from agent.observability.metrics import registry as metrics_registry
from agent.observability.tracing import TraceWriter
from agent.rag.retriever import Retriever
from agent.tools import registry as tool_registry
from agent.tools.registry import ToolContext, ToolRegistry
from agent.tools.sandbox import Workspace
from agent.types import RunResult, RunStatus, Usage, new_id

log = logging.getLogger(__name__)


class Agent:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        memory: MemoryStore | None = None,
        retriever: Retriever | None = None,
        registry: ToolRegistry | None = None,
        tools: list[str] | None = None,
        trace: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm = llm or build_client(self.settings)
        self.memory = memory
        self.retriever = retriever
        self.registry = registry or tool_registry
        self.tools = tools
        self.trace = trace

    def run(
        self,
        task: str,
        *,
        run_id: str | None = None,
        workspace: str | Path | None = None,
        bus: EventBus | None = None,
    ) -> RunResult:
        run_id = run_id or new_id("run")
        started = time.time()

        ws = Workspace(
            workspace or (self.settings.workspace_root / run_id),
            max_file_bytes=self.settings.max_file_bytes,
        )
        if bus is None:
            bus = EventBus(run_id)
        bus.run_id = run_id

        metrics = RunMetrics(run_id)
        metrics.attach(bus)
        writer = TraceWriter(run_id, self.settings.trace_dir) if self.trace else None
        if writer:
            writer.attach(bus)

        ctx = ToolContext(
            workspace=ws,
            settings=self.settings,
            run_id=run_id,
            memory=self.memory,
            retriever=self.retriever,
            events=bus,
        )
        deps = Deps(
            settings=self.settings,
            llm=self.llm,
            registry=self.registry,
            ctx=ctx,
            bus=bus,
            memory=self.memory,
            retriever=self.retriever,
            tool_names=self._enabled_tools(),
        )

        bus.emit(
            EventType.RUN_STARTED,
            task=task,
            workspace=str(ws.root),
            model=self.settings.model,
            tools=deps.tool_names,
        )
        log.info("run %s started in %s", run_id, ws.root)

        try:
            final = build_graph(deps).invoke(
                initial_state(run_id, task),
                config={"recursion_limit": self.settings.max_iterations * 4 + 20},
            )
            result = self._to_result(run_id, task, final, started)
        except Exception as exc:  # noqa: BLE001 - one bad run must not kill the server
            log.exception("run %s crashed", run_id)
            bus.emit(EventType.WARNING, message=f"run crashed: {exc}")
            result = RunResult(
                run_id=run_id,
                task=task,
                status=RunStatus.ABORTED,
                summary=f"The run crashed: {exc}",
                error=f"{type(exc).__name__}: {exc}",
                started_at=started,
                finished_at=time.time(),
            )
            bus.emit(EventType.RUN_FINISHED, status=result.status.value, summary=result.summary)
        finally:
            if writer:
                writer.close()

        metrics_registry.record({**metrics.snapshot(), "status": result.status.value,
                                 "task": task})
        return result

    # -- helpers ------------------------------------------------------------

    def _enabled_tools(self) -> list[str]:
        names = self.tools or self.registry.names()
        # A tool whose backing service is missing would only ever return an
        # error, so it is cheaper to keep it out of the schema entirely.
        if self.memory is None:
            names = [n for n in names if n not in {"recall", "remember"}]
        if self.retriever is None:
            names = [n for n in names if n != "search_knowledge"]
        return names

    def _to_result(
        self, run_id: str, task: str, final: dict[str, Any], started: float
    ) -> RunResult:
        state = cast(AgentState, final)  # LangGraph returns the merged state dict
        return RunResult(
            run_id=run_id,
            task=task,
            status=state.get("status", RunStatus.FAILED),
            summary=state.get("summary", ""),
            iterations=state.get("iterations", 0),
            plan=state.get("plan"),
            usage=state.get("usage", Usage()),
            started_at=started,
            finished_at=time.time(),
            error=state.get("error"),
            artifacts=state.get("artifacts", []),
        )
