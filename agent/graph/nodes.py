"""The nodes of the plan-act-observe loop.

Each node is a plain function of state that returns the keys it changed. The
interesting logic is all in three places:

* `act` guards against the loop's favourite failure mode - calling the same tool
  with the same arguments over and over - by fingerprinting calls and refusing
  the third repeat with an error the model can read.
* `observe` is a separate model call on purpose. Asking the same context window
  that just did the work whether the work is done gets you "yes" far too often;
  a fresh call that only sees the step and the tool output is markedly harder to
  talk into a false positive.
* `route_*` owns every budget check, so no node has to remember to stop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from agent import prompts
from agent.config import Settings
from agent.graph.state import AgentState, merge_usage, note
from agent.llm.base import LLMClient, LLMResponse, Message
from agent.memory.store import MemoryStore
from agent.observability.events import EventBus, EventType
from agent.rag.retriever import Retriever
from agent.tools.registry import ToolContext, ToolRegistry
from agent.types import (
    Observation,
    Plan,
    RunStatus,
    Step,
    StepStatus,
    ToolCall,
    ToolResult,
    Verdict,
)
from agent.utils import fingerprint, parse_json_object, truncate

log = logging.getLogger(__name__)

MAX_REPEATED_CALL = 2
MAX_IDLE_TURNS = 2


@dataclass
class Deps:
    """Everything the nodes need, injected once when the graph is built."""

    settings: Settings
    llm: LLMClient
    registry: ToolRegistry
    ctx: ToolContext
    bus: EventBus
    memory: MemoryStore | None = None
    retriever: Retriever | None = None
    tool_names: list[str] = field(default_factory=list)

    def schemas(self) -> list[dict[str, Any]]:
        return self.registry.schemas(self.tool_names or None)


class Nodes:
    def __init__(self, deps: Deps) -> None:
        self.d = deps

    # -- ground -------------------------------------------------------------

    def ground(self, state: AgentState) -> AgentState:
        """Pull memory and retrieval context before a single token is planned."""
        task = state["task"]
        memories_text = ""
        context_text = ""
        citations: list[str] = []

        if self.d.memory is not None:
            records = self.d.memory.search(task, k=4)
            memories_text = "\n".join(f"- [{r.kind}] {r.text}" for r in records)
            self.d.bus.emit(
                EventType.RETRIEVAL, kind="memory", query=task, hits=len(records)
            )

        if self.d.retriever is not None:
            chunks = self.d.retriever.search(task)
            context_text = self.d.retriever.as_context_block(chunks)
            citations = [c.citation() for c in chunks]
            self.d.bus.emit(
                EventType.RETRIEVAL,
                kind="knowledge",
                query=task,
                hits=len(chunks),
                sources=citations,
            )

        return AgentState(
            context=context_text,
            memories=memories_text,
            citations=citations,
            status=RunStatus.PLANNING,
            history=note(
                state,
                f"grounded on {len(citations)} retrieved chunk(s) "
                f"and {len(memories_text.splitlines())} memory hit(s)",
            ),
        )

    # -- plan ---------------------------------------------------------------

    def plan(self, state: AgentState) -> AgentState:
        response = self._call(
            system=prompts.PLANNER,
            user=prompts.planner_user(
                state["task"], state.get("context", ""), state.get("memories", "")
            ),
            model=self.d.settings.planner_model,
            label="plan",
        )
        plan = self._parse_plan(state["task"], response)
        self.d.bus.emit(
            EventType.PLAN_CREATED,
            steps=[s.description for s in plan.steps],
            assumptions=plan.assumptions,
        )
        return AgentState(
            plan=plan,
            messages=[Message.user(self._opening_brief(state, plan))],
            usage=merge_usage(state, response.usage),
            status=RunStatus.ACTING,
            history=note(state, f"planned {len(plan.steps)} step(s)"),
        )

    def replan(self, state: AgentState) -> AgentState:
        current = state["plan"]
        response = self._call(
            system=prompts.REPLANNER,
            user=prompts.replanner_user(state["task"], current, state.get("history", [])),
            model=self.d.settings.planner_model,
            label="replan",
        )
        revised = self._parse_plan(state["task"], response)
        revised.revision = current.revision + 1
        # Steps already finished stay finished; the revision covers what is left.
        done = [s for s in current.steps if s.status is StepStatus.DONE]
        revised.steps = done + revised.steps

        self.d.bus.emit(
            EventType.PLAN_REVISED,
            revision=revised.revision,
            steps=[s.description for s in revised.steps],
        )
        messages = [
            *state.get("messages", []),
            Message.user(
                "The plan has been revised. Remaining steps:\n"
                + prompts.render_plan(revised)
                + "\n\nContinue from the first unfinished step."
            ),
        ]
        return AgentState(
            plan=revised,
            messages=messages,
            replans=state.get("replans", 0) + 1,
            usage=merge_usage(state, response.usage),
            status=RunStatus.ACTING,
            history=note(state, f"revised the plan (revision {revised.revision})"),
        )

    # -- act ----------------------------------------------------------------

    def act(self, state: AgentState) -> AgentState:
        plan: Plan = state["plan"]
        step = plan.current
        if step is not None and step.status is StepStatus.PENDING:
            step.status = StepStatus.ACTIVE
            step.attempts += 1

        self.d.bus.emit(
            EventType.STEP_STARTED,
            step=step.description if step else "(no step)",
            step_id=step.id if step else None,
            iteration=state.get("iterations", 0) + 1,
        )

        response = self._complete(
            system=prompts.actor_system(str(self.d.ctx.workspace.root)),
            messages=state["messages"],
            tools=self.d.schemas(),
            label="act",
        )
        if response.text:
            self.d.bus.emit(EventType.THOUGHT, text=truncate(response.text, 1200))

        updates = AgentState(
            iterations=state.get("iterations", 0) + 1,
            usage=merge_usage(state, response.usage),
            status=RunStatus.ACTING,
            plan=plan,
        )

        if not response.wants_tools:
            return {**updates, **self._handle_no_tool_call(state, response)}

        messages = [*state["messages"], Message.assistant(_assistant_content(response))]
        seen = dict(state.get("seen_calls", {}))
        blocks: list[dict[str, Any]] = []
        results: list[ToolResult] = []

        for call in response.tool_calls:
            result = self._execute(call, seen)
            results.append(result)
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": result.truncated(),
                    "is_error": not result.ok,
                }
            )

        messages.append(Message.tool_results(blocks))
        submitted = self.d.ctx.scratch.get("submitted")

        return {
            **updates,
            "messages": messages,
            "seen_calls": seen,
            "last_result": results[0] if results else None,
            "idle_turns": 0,
            "artifacts": list(submitted.get("artifacts", [])) if submitted else
                         state.get("artifacts", []),
            "finished": bool(submitted),
            "history": note(
                state,
                "; ".join(
                    f"{r.tool} -> {'ok' if r.ok else 'error: ' + truncate(r.error or '', 120)}"
                    for r in results
                ),
            ),
        }

    def _handle_no_tool_call(self, state: AgentState, response: LLMResponse) -> dict[str, Any]:
        """The model answered in prose instead of calling a tool."""
        idle = state.get("idle_turns", 0) + 1
        if idle >= MAX_IDLE_TURNS:
            return {
                "idle_turns": idle,
                "finished": True,
                "summary": response.text,
                "history": note(state, "stopped calling tools; treating the reply as final"),
            }
        nudge = (
            "You did not call a tool. If the task is finished, call submit_result "
            "with what you did and how you verified it. Otherwise call the next tool."
        )
        return {
            "idle_turns": idle,
            "messages": [
                *state["messages"],
                Message.assistant(_assistant_content(response)),
                Message.user(nudge),
            ],
            "last_result": None,
            "history": note(state, "no tool call - nudged the agent back to the plan"),
        }

    def _execute(self, call: ToolCall, seen: dict[str, int]) -> ToolResult:
        key = fingerprint(call.name, call.arguments)
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > MAX_REPEATED_CALL:
            self.d.bus.emit(
                EventType.WARNING,
                message=f"blocked a {seen[key]}th identical call to {call.name}",
            )
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                ok=False,
                error=(
                    f"You have already called {call.name} with these exact arguments "
                    f"{seen[key] - 1} times and got the same result. Repeating it will not "
                    "help. Change the approach, or call submit_result and explain what is "
                    "blocking you."
                ),
            )

        self.d.bus.emit(EventType.TOOL_CALLED, tool=call.name, arguments=call.arguments)
        result = self.d.registry.invoke(call, self.d.ctx)
        self.d.bus.emit(
            EventType.TOOL_RESULT,
            tool=call.name,
            ok=result.ok,
            duration_ms=result.duration_ms,
            preview=truncate(result.content if result.ok else (result.error or ""), 600),
        )
        if call.name == "remember" and result.ok:
            self.d.bus.emit(EventType.MEMORY_WRITE, text=call.arguments.get("text", ""))
        return result

    # -- observe ------------------------------------------------------------

    def observe(self, state: AgentState) -> AgentState:
        result = state.get("last_result")
        plan: Plan = state["plan"]
        step = plan.current

        if result is None:
            return AgentState(status=RunStatus.ACTING)

        response = self._call(
            system=prompts.OBSERVER,
            user=prompts.observer_user(
                step.description if step else state["task"],
                result.tool,
                result.ok,
                result.truncated(6000),
            ),
            label="observe",
        )
        observation = self._parse_observation(response, step.id if step else None, result)

        if step is not None:
            if observation.step_satisfied:
                step.status = StepStatus.DONE
                step.notes = truncate(observation.summary, 160)
            elif not result.ok and step.attempts >= 3:
                step.status = StepStatus.FAILED
                step.notes = "failed three times"

        self.d.bus.emit(
            EventType.OBSERVATION,
            summary=observation.summary,
            verdict=observation.verdict.value,
            step_satisfied=observation.step_satisfied,
            concerns=observation.concerns,
        )

        finished = observation.verdict is Verdict.FINISH or plan.complete
        return AgentState(
            plan=plan,
            last_observation=observation,
            last_result=None,
            usage=merge_usage(state, response.usage),
            status=RunStatus.OBSERVING,
            finished=bool(state.get("finished")) or finished,
            history=note(state, f"observed: {truncate(observation.summary, 160)}"),
        )

    # -- summarise ----------------------------------------------------------

    def summarise(self, state: AgentState) -> AgentState:
        status = self._final_status(state)
        outcome = state.get("error") or (
            "the agent submitted a result" if status is RunStatus.SUCCEEDED else "the run stopped"
        )
        submitted = self.d.ctx.scratch.get("submitted") or {}
        summary = state.get("summary", "")

        if not summary:
            response = self._call(
                system=prompts.SUMMARISER,
                user=prompts.summariser_user(
                    state["task"], state.get("plan"), state.get("history", []), outcome
                ),
                label="summarise",
            )
            summary = response.text.strip() or submitted.get("summary", "")
            state = {**state, "usage": merge_usage(state, response.usage)}

        self.d.bus.emit(
            EventType.RUN_FINISHED,
            status=status.value,
            summary=summary,
            iterations=state.get("iterations", 0),
            usd=state.get("usage").usd if state.get("usage") else 0.0,
        )
        return AgentState(
            summary=summary,
            status=status,
            finished=True,
            usage=state.get("usage"),
            artifacts=state.get("artifacts", []) or list(submitted.get("artifacts", [])),
        )

    def _final_status(self, state: AgentState) -> RunStatus:
        if state.get("error"):
            return RunStatus.ABORTED
        observation = state.get("last_observation")
        if observation is not None and observation.verdict is Verdict.GIVE_UP:
            return RunStatus.FAILED
        if self.d.ctx.scratch.get("submitted"):
            return RunStatus.SUCCEEDED
        plan = state.get("plan")
        if plan is not None and plan.complete:
            return RunStatus.SUCCEEDED
        return RunStatus.FAILED

    # -- llm plumbing -------------------------------------------------------

    def _call(self, *, system: str, user: str, label: str,
              model: str | None = None) -> LLMResponse:
        return self._complete(
            system=system, messages=[Message.user(user)], tools=None, label=label, model=model
        )

    def _complete(self, *, system: str, messages: list[Message],
                  tools: list[dict[str, Any]] | None, label: str,
                  model: str | None = None) -> LLMResponse:
        response = self.d.llm.complete(
            system=system, messages=messages, tools=tools, model=model
        )
        self.d.bus.emit(
            EventType.LLM_CALL,
            phase=label,
            model=model or self.d.settings.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            usd=response.usage.usd,
            stop_reason=response.stop_reason,
        )
        return response

    # -- parsing ------------------------------------------------------------

    def _parse_plan(self, task: str, response: LLMResponse) -> Plan:
        payload = parse_json_object(response.text)
        if not payload or not isinstance(payload.get("steps"), list):
            self.d.bus.emit(
                EventType.WARNING,
                message="planner did not return usable JSON; falling back to a single step",
            )
            return Plan(
                goal=task,
                steps=[Step(description=task, rationale="planner output was unparseable")],
            )

        steps: list[Step] = []
        for raw in payload["steps"][:8]:
            if isinstance(raw, str):
                steps.append(Step(description=raw))
            elif isinstance(raw, dict) and raw.get("description"):
                steps.append(
                    Step(
                        description=str(raw["description"]),
                        rationale=str(raw.get("rationale", "")),
                        expected_tool=raw.get("expected_tool"),
                    )
                )
        if not steps:
            steps = [Step(description=task)]

        assumptions = [str(a) for a in payload.get("assumptions", []) if a]
        return Plan(goal=task, steps=steps, assumptions=assumptions)

    def _parse_observation(self, response: LLMResponse, step_id: str | None,
                           result: ToolResult) -> Observation:
        payload = parse_json_object(response.text)
        if not payload:
            # No usable verdict: trust the tool's own exit status rather than prose.
            return Observation(
                step_id=step_id,
                summary=truncate(response.text or "observer returned nothing", 300),
                step_satisfied=result.ok,
                verdict=Verdict.CONTINUE,
            )
        try:
            verdict = Verdict(str(payload.get("verdict", "continue")).lower())
        except ValueError:
            verdict = Verdict.CONTINUE
        return Observation(
            step_id=step_id,
            summary=str(payload.get("summary", "")).strip() or "(no summary)",
            step_satisfied=bool(payload.get("step_satisfied", False)) and result.ok,
            verdict=verdict,
            evidence=[str(e) for e in payload.get("evidence", [])][:6],
            concerns=[str(c) for c in payload.get("concerns", [])][:6],
        )

    # -- brief --------------------------------------------------------------

    def _opening_brief(self, state: AgentState, plan: Plan) -> str:
        parts = [f"TASK\n{state['task']}", f"PLAN\n{prompts.render_plan(plan)}"]
        if state.get("memories"):
            parts.append(f"FROM EARLIER RUNS\n{state['memories']}")
        if state.get("context"):
            parts.append(
                "RETRIEVED CONTEXT (cite these paths if you use them)\n"
                + truncate(state["context"], 8000)
            )
        parts.append("Start with the first step. One tool call per turn.")
        return "\n\n".join(parts)


def _assistant_content(response: LLMResponse) -> Any:
    """Rebuild the assistant turn, including tool_use blocks, for the next request."""
    if response.raw_content:
        return response.raw_content
    blocks: list[dict[str, Any]] = []
    if response.text:
        blocks.append({"type": "text", "text": response.text})
    for call in response.tool_calls:
        blocks.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    return blocks or response.text or "(no content)"
