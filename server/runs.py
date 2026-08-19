"""In-process run manager.

Runs execute on a small thread pool. The graph is synchronous and spends almost
all of its wall clock blocked on the provider or on a subprocess, so threads are
the right shape here - and it keeps the agent code free of async plumbing it
does not otherwise need.

Each run keeps its event bus alive after it finishes so a dashboard that
connects late still gets the whole story replayed.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.loop import Agent
from agent.observability.events import EventBus
from agent.types import RunResult, RunStatus, new_id

log = logging.getLogger(__name__)

MAX_KEPT_RUNS = 100


@dataclass
class RunRecord:
    id: str
    task: str
    bus: EventBus
    status: RunStatus = RunStatus.QUEUED
    created_at: float = field(default_factory=time.time)
    workspace: str | None = None
    result: RunResult | None = None
    error: str | None = None
    future: Future[RunResult] | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "status": self.status.value,
            "created_at": self.created_at,
            "workspace": self.workspace,
            "iterations": self.result.iterations if self.result else 0,
            "duration_s": self.result.duration_s if self.result else None,
            "usd": self.result.usage.usd if self.result else 0.0,
            "summary": self.result.summary if self.result else "",
            "error": self.error,
        }

    def detail(self) -> dict[str, Any]:
        data = self.summary()
        data["events"] = [e.model_dump(mode="json") for e in self.bus.history()]
        if self.result:
            data["plan"] = self.result.plan.model_dump(mode="json") if self.result.plan else None
            data["artifacts"] = self.result.artifacts
            data["usage"] = self.result.usage.model_dump()
        return data


class RunManager:
    def __init__(self, agent_factory: Callable[[], Agent], max_workers: int = 4) -> None:
        self._agent_factory = agent_factory
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="aca-run")
        self._runs: OrderedDict[str, RunRecord] = OrderedDict()
        self._lock = threading.Lock()

    def start(self, task: str, workspace: str | None = None) -> RunRecord:
        run_id = new_id("run")
        record = RunRecord(id=run_id, task=task, bus=EventBus(run_id), workspace=workspace)
        with self._lock:
            self._runs[run_id] = record
            while len(self._runs) > MAX_KEPT_RUNS:
                self._runs.popitem(last=False)
        record.future = self._pool.submit(self._execute, record)
        return record

    def _execute(self, record: RunRecord) -> RunResult:
        record.status = RunStatus.PLANNING
        try:
            agent = self._agent_factory()
            result = agent.run(
                record.task, run_id=record.id, workspace=record.workspace, bus=record.bus
            )
            record.result = result
            record.status = result.status
            return result
        except Exception as exc:  # noqa: BLE001 - surface it to the client, keep serving
            log.exception("run %s failed in the manager", record.id)
            record.status = RunStatus.ABORTED
            record.error = f"{type(exc).__name__}: {exc}"
            raise

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list(self, limit: int = 50) -> list[RunRecord]:
        with self._lock:
            return list(self._runs.values())[-limit:][::-1]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
