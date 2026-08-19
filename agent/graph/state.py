"""The state that flows through the graph.

LangGraph hands the same dict to every node and merges what each one returns, so
this is the entire contract between the nodes. Keeping it flat and boring is
deliberate: it serialises straight into a trace, and a state you cannot print is
a state you cannot debug.
"""

from __future__ import annotations

from typing import Any, TypedDict

from agent.llm.base import Message
from agent.types import Observation, Plan, RunStatus, ToolResult, Usage


class AgentState(TypedDict, total=False):
    run_id: str
    task: str

    # grounding gathered once, before planning
    context: str
    memories: str
    citations: list[str]

    plan: Plan
    messages: list[Message]

    last_result: ToolResult | None
    last_observation: Observation | None
    history: list[str]

    iterations: int
    replans: int
    idle_turns: int
    seen_calls: dict[str, int]

    usage: Usage
    status: RunStatus
    summary: str
    artifacts: list[str]
    error: str | None
    finished: bool


def initial_state(run_id: str, task: str) -> AgentState:
    return AgentState(
        run_id=run_id,
        task=task,
        context="",
        memories="",
        citations=[],
        messages=[],
        last_result=None,
        last_observation=None,
        history=[],
        iterations=0,
        replans=0,
        idle_turns=0,
        seen_calls={},
        usage=Usage(),
        status=RunStatus.QUEUED,
        summary="",
        artifacts=[],
        error=None,
        finished=False,
    )


def note(state: AgentState, line: str) -> list[str]:
    """Append to the human-readable run log without mutating the input state."""
    return [*state.get("history", []), line]


def merge_usage(state: AgentState, usage: Usage) -> Usage:
    return state.get("usage", Usage()) + usage


def as_dict(state: AgentState) -> dict[str, Any]:
    plan = state.get("plan")
    return {
        "run_id": state.get("run_id"),
        "task": state.get("task"),
        "status": getattr(state.get("status"), "value", state.get("status")),
        "iterations": state.get("iterations", 0),
        "replans": state.get("replans", 0),
        "plan": plan.model_dump(mode="json") if plan else None,
        "history": state.get("history", []),
        "summary": state.get("summary", ""),
        "artifacts": state.get("artifacts", []),
        "usage": state.get("usage", Usage()).model_dump(),
        "error": state.get("error"),
    }
