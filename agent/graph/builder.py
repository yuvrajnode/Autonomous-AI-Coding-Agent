"""Wiring the nodes into a graph.

    ground -> plan -> act -> observe -> act
                       ^        |
                       |        +--> replan --> act
                       |        +--> halt ----> summarise -> END
                       +-----------------------------------+

Every edge out of `act` and `observe` goes through a router, and the routers are
the only place that decides to stop. Budget enforcement lives there rather than
inside the nodes so there is exactly one answer to "why did this run end".
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agent.graph.nodes import Deps, Nodes
from agent.graph.state import AgentState
from agent.types import RunStatus, Usage, Verdict

log = logging.getLogger(__name__)


def over_budget(state: AgentState, settings: Any) -> str | None:
    """Returns a human-readable reason, or None if the run may continue."""
    if state.get("iterations", 0) >= settings.max_iterations:
        return f"hit the iteration ceiling ({settings.max_iterations})"
    usage: Usage = state.get("usage", Usage())
    if settings.usd_budget and usage.usd >= settings.usd_budget:
        return f"hit the cost ceiling (${settings.usd_budget:.2f})"
    return None


def build_graph(deps: Deps) -> Any:
    nodes = Nodes(deps)
    settings = deps.settings

    def halt(state: AgentState) -> AgentState:
        reason = over_budget(state, settings) or "the run was stopped"
        log.warning("run %s halted: %s", state.get("run_id"), reason)
        return AgentState(
            error=reason,
            status=RunStatus.ABORTED,
            history=[*state.get("history", []), f"halted: {reason}"],
        )

    def route_after_act(state: AgentState) -> str:
        if state.get("finished"):
            return "summarise"
        if over_budget(state, settings):
            return "halt"
        return "observe" if state.get("last_result") is not None else "act"

    def route_after_observe(state: AgentState) -> str:
        if state.get("finished"):
            return "summarise"
        if over_budget(state, settings):
            return "halt"
        observation = state.get("last_observation")
        if observation is not None:
            if observation.verdict is Verdict.GIVE_UP:
                return "summarise"
            if (
                observation.verdict is Verdict.REPLAN
                and state.get("replans", 0) < settings.max_replans
            ):
                return "replan"
        return "act"

    graph = StateGraph(AgentState)
    graph.add_node("ground", nodes.ground)
    graph.add_node("plan", nodes.plan)
    graph.add_node("replan", nodes.replan)
    graph.add_node("act", nodes.act)
    graph.add_node("observe", nodes.observe)
    graph.add_node("summarise", nodes.summarise)
    graph.add_node("halt", halt)

    graph.set_entry_point("ground")
    graph.add_edge("ground", "plan")
    graph.add_edge("plan", "act")
    graph.add_edge("replan", "act")
    graph.add_edge("halt", "summarise")
    graph.add_edge("summarise", END)

    graph.add_conditional_edges(
        "act",
        route_after_act,
        {"observe": "observe", "act": "act", "summarise": "summarise", "halt": "halt"},
    )
    graph.add_conditional_edges(
        "observe",
        route_after_observe,
        {"act": "act", "replan": "replan", "summarise": "summarise", "halt": "halt"},
    )

    return graph.compile()
