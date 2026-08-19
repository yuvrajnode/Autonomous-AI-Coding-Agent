"""Trace files.

Every run writes one newline-delimited JSON file under the trace directory. It
is append-only and flushed on every write, so if a run hangs you can tail the
file and see exactly which tool call it is stuck inside.

Spans are nested by a simple stack. This is not OpenTelemetry - it is the 5% of
it that is useful when the thing you are debugging is a model's decision.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from agent.observability.events import Event, EventBus

log = logging.getLogger(__name__)


class TraceWriter:
    """Persists every event on the bus to `<trace_dir>/<run_id>.jsonl`."""

    def __init__(self, run_id: str, trace_dir: str | Path) -> None:
        self.run_id = run_id
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{run_id}.jsonl"
        self._fh = self.path.open("a", encoding="utf-8")

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self.write)

    def write(self, event: Event) -> None:
        self._fh.write(event.line() + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                log.warning("skipping malformed trace line in %s", path)
    return out


class Tracer:
    """Nested spans with wall-clock timing."""

    def __init__(self, run_id: str, bus: EventBus | None = None) -> None:
        self.run_id = run_id
        self.bus = bus
        self.spans: list[dict[str, Any]] = []
        self._stack: list[str] = []

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        span_id = uuid.uuid4().hex[:8]
        record: dict[str, Any] = {
            "id": span_id,
            "name": name,
            "parent": self._stack[-1] if self._stack else None,
            "start": time.time(),
            "attrs": attrs,
            "status": "ok",
        }
        self.spans.append(record)
        self._stack.append(span_id)
        try:
            yield record
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._stack.pop()
            record["end"] = time.time()
            record["duration_ms"] = int((record["end"] - record["start"]) * 1000)

    def timeline(self) -> list[dict[str, Any]]:
        return sorted(self.spans, key=lambda s: s["start"])

    def slowest(self, n: int = 5) -> list[dict[str, Any]]:
        finished = [s for s in self.spans if "duration_ms" in s]
        return sorted(finished, key=lambda s: s["duration_ms"], reverse=True)[:n]
