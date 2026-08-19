"""Metrics.

`RunMetrics` subscribes to a single run's event bus and derives the numbers that
actually tell you whether the agent is behaving: how often tools fail, how often
retrieval comes back empty, how many times the plan had to be rewritten, and
what the whole thing cost.

`MetricsRegistry` keeps the last N run summaries in process so the dashboard has
something to draw without querying postgres on every poll.
"""

from __future__ import annotations

import statistics
import threading
import time
from collections import Counter, deque
from typing import Any

from agent.observability.events import Event, EventBus, EventType


class RunMetrics:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.started_at = time.time()
        self.finished_at: float | None = None

        self.iterations = 0
        self.replans = 0
        self.tool_calls: Counter[str] = Counter()
        self.tool_failures: Counter[str] = Counter()
        self.tool_latency_ms: dict[str, list[int]] = {}

        self.llm_requests = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.usd = 0.0

        self.retrievals = 0
        self.empty_retrievals = 0
        self.memories_written = 0
        self.warnings: list[str] = []

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self.handle)

    def handle(self, event: Event) -> None:
        data = event.data
        if event.type is EventType.STEP_STARTED:
            self.iterations += 1
        elif event.type is EventType.PLAN_REVISED:
            self.replans += 1
        elif event.type is EventType.TOOL_CALLED:
            self.tool_calls[data.get("tool", "?")] += 1
        elif event.type is EventType.TOOL_RESULT:
            tool = data.get("tool", "?")
            if not data.get("ok", True):
                self.tool_failures[tool] += 1
            self.tool_latency_ms.setdefault(tool, []).append(int(data.get("duration_ms", 0)))
        elif event.type is EventType.LLM_CALL:
            self.llm_requests += 1
            self.input_tokens += int(data.get("input_tokens", 0))
            self.output_tokens += int(data.get("output_tokens", 0))
            self.cache_read_tokens += int(data.get("cache_read_tokens", 0))
            self.usd = round(self.usd + float(data.get("usd", 0.0)), 6)
        elif event.type is EventType.RETRIEVAL:
            self.retrievals += 1
            if not data.get("hits"):
                self.empty_retrievals += 1
        elif event.type is EventType.MEMORY_WRITE:
            self.memories_written += 1
        elif event.type is EventType.WARNING:
            self.warnings.append(str(data.get("message", "")))
        elif event.type is EventType.RUN_FINISHED:
            self.finished_at = time.time()

    # -- derived ------------------------------------------------------------

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())

    @property
    def tool_failure_rate(self) -> float:
        total = self.total_tool_calls
        return round(sum(self.tool_failures.values()) / total, 4) if total else 0.0

    @property
    def retrieval_hit_rate(self) -> float:
        if not self.retrievals:
            return 0.0
        return round(1 - (self.empty_retrievals / self.retrievals), 4)

    @property
    def cache_hit_rate(self) -> float:
        total = self.input_tokens + self.cache_read_tokens
        return round(self.cache_read_tokens / total, 4) if total else 0.0

    @property
    def duration_s(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 3)

    def p50_latency(self, tool: str) -> int:
        samples = self.tool_latency_ms.get(tool, [])
        return int(statistics.median(samples)) if samples else 0

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "iterations": self.iterations,
            "replans": self.replans,
            "duration_s": self.duration_s,
            "tool_calls": dict(self.tool_calls),
            "tool_failures": dict(self.tool_failures),
            "tool_failure_rate": self.tool_failure_rate,
            "tool_p50_ms": {t: self.p50_latency(t) for t in self.tool_calls},
            "llm_requests": self.llm_requests,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": self.cache_hit_rate,
            "usd": round(self.usd, 4),
            "retrievals": self.retrievals,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "memories_written": self.memories_written,
            "warnings": self.warnings,
        }


class MetricsRegistry:
    """Rolling window of finished runs, for the dashboard header."""

    def __init__(self, capacity: int = 200) -> None:
        self._runs: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def record(self, summary: dict[str, Any]) -> None:
        with self._lock:
            self._runs.append(summary)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._runs)[-limit:][::-1]

    def rollup(self) -> dict[str, Any]:
        with self._lock:
            runs = list(self._runs)
        if not runs:
            return {
                "runs": 0,
                "success_rate": 0.0,
                "avg_iterations": 0.0,
                "avg_duration_s": 0.0,
                "total_usd": 0.0,
                "tool_failure_rate": 0.0,
            }
        succeeded = sum(1 for r in runs if r.get("status") == "succeeded")
        return {
            "runs": len(runs),
            "success_rate": round(succeeded / len(runs), 4),
            "avg_iterations": round(statistics.fmean(r.get("iterations", 0) for r in runs), 2),
            "avg_duration_s": round(statistics.fmean(r.get("duration_s", 0.0) for r in runs), 2),
            "total_usd": round(sum(r.get("usd", 0.0) for r in runs), 4),
            "tool_failure_rate": round(
                statistics.fmean(r.get("tool_failure_rate", 0.0) for r in runs), 4
            ),
        }


registry = MetricsRegistry()
