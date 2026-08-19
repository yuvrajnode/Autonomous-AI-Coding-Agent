"""The event bus.

One stream of typed events per run. The trace writer, the metrics collector and
the websocket that feeds the dashboard are all just subscribers, which means the
loop itself never knows or cares who is watching.

Subscribers are called synchronously and their exceptions are swallowed and
logged. A broken dashboard connection must not take down a run.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    RUN_FINISHED = "run.finished"
    PLAN_CREATED = "plan.created"
    PLAN_REVISED = "plan.revised"
    STEP_STARTED = "step.started"
    STEP_FINISHED = "step.finished"
    THOUGHT = "agent.thought"
    TOOL_CALLED = "tool.called"
    TOOL_RESULT = "tool.result"
    OBSERVATION = "agent.observation"
    RETRIEVAL = "memory.retrieval"
    MEMORY_WRITE = "memory.write"
    LLM_CALL = "llm.call"
    BUDGET = "run.budget"
    WARNING = "run.warning"


class Event(BaseModel):
    type: EventType
    run_id: str
    ts: float = Field(default_factory=time.time)
    seq: int = 0
    data: dict[str, Any] = Field(default_factory=dict)

    def line(self) -> str:
        return self.model_dump_json()


Subscriber = Callable[[Event], None]


class EventBus:
    def __init__(self, run_id: str = "") -> None:
        self.run_id = run_id
        self._subscribers: list[Subscriber] = []
        self._history: list[Event] = []
        self._seq = 0
        self._lock = threading.Lock()

    def subscribe(self, fn: Subscriber, *, replay: bool = False) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(fn)
            backlog = list(self._history) if replay else []
        for event in backlog:
            _safe_call(fn, event)

        def unsubscribe() -> None:
            with self._lock:
                if fn in self._subscribers:
                    self._subscribers.remove(fn)

        return unsubscribe

    def emit(self, type: EventType, **data: Any) -> Event:
        with self._lock:
            self._seq += 1
            event = Event(type=type, run_id=self.run_id, seq=self._seq, data=data)
            self._history.append(event)
            listeners = list(self._subscribers)
        for fn in listeners:
            _safe_call(fn, event)
        return event

    def history(self, since_seq: int = 0) -> list[Event]:
        with self._lock:
            return [e for e in self._history if e.seq > since_seq]

    def __len__(self) -> int:
        return len(self._history)

    def __bool__(self) -> bool:
        # Without this, an empty bus is falsy and `bus or EventBus()` quietly
        # throws away the caller's bus. That cost an afternoon once.
        return True


def _safe_call(fn: Subscriber, event: Event) -> None:
    try:
        fn(event)
    except Exception:  # noqa: BLE001 - a subscriber must never break the run
        log.exception("event subscriber failed on %s", event.type)
