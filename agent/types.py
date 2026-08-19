"""Domain models shared by the loop, the tools, the API and the eval harness.

These are plain pydantic models on purpose: they get serialised straight into
trace files and pushed over the websocket to the dashboard.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class RunStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    ACTING = "acting"
    OBSERVING = "observing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABORTED = "aborted"

    @property
    def terminal(self) -> bool:
        return self in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.ABORTED}


class StepStatus(str, Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class Step(BaseModel):
    id: str = Field(default_factory=lambda: new_id("step"))
    description: str
    rationale: str = ""
    expected_tool: str | None = None
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    notes: str = ""


class Plan(BaseModel):
    goal: str
    steps: list[Step] = Field(default_factory=list)
    revision: int = 0
    assumptions: list[str] = Field(default_factory=list)

    @property
    def current(self) -> Step | None:
        for step in self.steps:
            if step.status in (StepStatus.PENDING, StepStatus.ACTIVE):
                return step
        return None

    @property
    def complete(self) -> bool:
        return all(s.status in (StepStatus.DONE, StepStatus.SKIPPED) for s in self.steps)

    def progress(self) -> tuple[int, int]:
        done = sum(1 for s in self.steps if s.status is StepStatus.DONE)
        return done, len(self.steps)


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: new_id("call"))
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    call_id: str
    tool: str
    ok: bool
    content: str = ""
    error: str | None = None
    duration_ms: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def truncated(self, limit: int = 4000) -> str:
        body = self.content if self.ok else (self.error or "unknown error")
        if len(body) <= limit:
            return body
        head = body[: limit - 400]
        tail = body[-360:]
        return f"{head}\n... [{len(body) - limit} chars elided] ...\n{tail}"


class Verdict(str, Enum):
    CONTINUE = "continue"
    REPLAN = "replan"
    FINISH = "finish"
    GIVE_UP = "give_up"


class Observation(BaseModel):
    """What the observer node concluded after a tool result came back."""

    step_id: str | None = None
    summary: str
    verdict: Verdict = Verdict.CONTINUE
    step_satisfied: bool = False
    evidence: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    id: str
    source: str
    text: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)

    def citation(self) -> str:
        loc = self.metadata.get("lines")
        return f"{self.source}:{loc}" if loc else self.source


class MemoryRecord(BaseModel):
    id: str
    kind: str
    text: str
    score: float = 0.0
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    requests: int = 0
    usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            requests=self.requests + other.requests,
            usd=round(self.usd + other.usd, 6),
        )


class RunResult(BaseModel):
    run_id: str
    task: str
    status: RunStatus
    summary: str = ""
    iterations: int = 0
    plan: Plan | None = None
    usage: Usage = Field(default_factory=Usage)
    started_at: float = 0.0
    finished_at: float = 0.0
    error: str | None = None
    artifacts: list[str] = Field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return round(max(0.0, self.finished_at - self.started_at), 3)
