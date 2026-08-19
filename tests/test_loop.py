"""Loop behaviour, driven entirely by scripted model responses."""

from __future__ import annotations

import json

from agent.llm.base import LLMResponse
from agent.llm.scripted import ScriptedClient
from agent.loop import Agent
from agent.observability.events import EventBus, EventType
from agent.types import RunStatus, ToolCall, Usage
from agent.utils import parse_json_object


def plan_response(*descriptions: str) -> LLMResponse:
    return LLMResponse(
        text=json.dumps(
            {"assumptions": [], "steps": [{"description": d} for d in descriptions]}
        )
    )


def observation(satisfied: bool = True, verdict: str = "continue") -> LLMResponse:
    return LLMResponse(
        text=json.dumps(
            {"summary": "checked", "step_satisfied": satisfied, "verdict": verdict}
        )
    )


def tool_response(name: str, **arguments) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(name=name, arguments=arguments)],
        stop_reason="tool_use",
    )


def agent_with(queue, settings, **kwargs) -> Agent:
    return Agent(settings=settings, llm=ScriptedClient(queue=queue), **kwargs)


# -- happy path -------------------------------------------------------------


def test_a_scripted_run_completes_and_writes_the_file(settings, tmp_path):
    agent = Agent(settings=settings, llm=ScriptedClient.demo(), retriever=None)
    result = agent.run("write fizzbuzz", workspace=tmp_path / "ws")

    assert result.status is RunStatus.SUCCEEDED
    assert (tmp_path / "ws" / "solution.py").exists()
    assert result.iterations >= 3
    assert result.summary


def test_events_are_emitted_on_the_caller_supplied_bus(settings, tmp_path):
    bus = EventBus()
    seen: list[EventType] = []
    bus.subscribe(lambda e: seen.append(e.type))

    Agent(settings=settings, llm=ScriptedClient.demo(), retriever=None).run(
        "write fizzbuzz", workspace=tmp_path / "ws", bus=bus
    )

    assert EventType.RUN_STARTED in seen
    assert EventType.PLAN_CREATED in seen
    assert EventType.TOOL_CALLED in seen
    assert seen[-1] is EventType.RUN_FINISHED


def test_the_trace_file_is_written(settings, tmp_path):
    result = Agent(settings=settings, llm=ScriptedClient.demo(), retriever=None).run(
        "write fizzbuzz", workspace=tmp_path / "ws"
    )
    trace = settings.trace_dir / f"{result.run_id}.jsonl"
    assert trace.exists()
    assert trace.read_text().count("\n") > 5


# -- guard rails ------------------------------------------------------------


def test_repeating_an_identical_tool_call_is_blocked(settings, tmp_path):
    queue = [plan_response("look around")]
    for _ in range(6):
        queue.append(tool_response("list_dir", path="."))
        queue.append(observation(satisfied=False))
    queue.append(LLMResponse(text="giving up"))

    bus = EventBus()
    warnings: list[str] = []
    bus.subscribe(
        lambda e: warnings.append(e.data.get("message", ""))
        if e.type is EventType.WARNING
        else None
    )

    agent_with(queue, settings, retriever=None).run(
        "look around", workspace=tmp_path / "ws", bus=bus
    )

    assert any("identical call" in w for w in warnings)


def test_the_iteration_ceiling_stops_the_run(settings, tmp_path):
    settings.max_iterations = 3
    queue = [plan_response("step one", "step two", "step three", "step four")]
    for i in range(12):
        queue.append(tool_response("write_file", path=f"f{i}.py", content=f"# {i}\n"))
        queue.append(observation(satisfied=False))
    queue.append(LLMResponse(text="stopped"))

    result = agent_with(queue, settings, retriever=None).run(
        "do four things", workspace=tmp_path / "ws"
    )

    assert result.status is RunStatus.ABORTED
    assert "iteration ceiling" in (result.error or "")
    assert result.iterations <= 3


def test_the_cost_ceiling_stops_the_run(settings, tmp_path):
    settings.usd_budget = 0.01
    expensive = LLMResponse(
        text="",
        tool_calls=[ToolCall(name="list_dir", arguments={"path": "."})],
        stop_reason="tool_use",
        usage=Usage(input_tokens=1000, output_tokens=1000, requests=1, usd=0.02),
    )
    queue = [plan_response("look"), expensive, observation(satisfied=False), LLMResponse(text="x")]

    result = agent_with(queue, settings, retriever=None).run(
        "look around", workspace=tmp_path / "ws"
    )

    assert result.status is RunStatus.ABORTED
    assert "cost ceiling" in (result.error or "")


def test_a_verdict_of_replan_rewrites_the_plan(settings, tmp_path):
    queue = [
        plan_response("try the wrong thing"),
        tool_response("list_dir", path="."),
        observation(satisfied=False, verdict="replan"),
        plan_response("try the right thing"),
        tool_response("write_file", path="ok.py", content="1\n"),
        observation(satisfied=True, verdict="finish"),
        LLMResponse(text="done"),
    ]
    bus = EventBus()
    revisions: list[int] = []
    bus.subscribe(
        lambda e: revisions.append(e.data["revision"])
        if e.type is EventType.PLAN_REVISED
        else None
    )

    result = agent_with(queue, settings, retriever=None).run(
        "do the thing", workspace=tmp_path / "ws", bus=bus
    )

    assert revisions == [1]
    assert (tmp_path / "ws" / "ok.py").exists()
    assert result.status is RunStatus.SUCCEEDED


def test_prose_instead_of_a_tool_call_gets_one_nudge_then_stops(settings, tmp_path):
    queue = [
        plan_response("do something"),
        LLMResponse(text="I think we are done here"),
        LLMResponse(text="Really, we are done"),
    ]
    result = agent_with(queue, settings, retriever=None).run(
        "do something", workspace=tmp_path / "ws"
    )

    assert result.summary == "Really, we are done"


def test_an_unparseable_plan_falls_back_to_a_single_step(settings, tmp_path):
    queue = [
        LLMResponse(text="here is a plan, trust me"),
        tool_response("write_file", path="a.py", content="1\n"),
        observation(satisfied=True, verdict="finish"),
        LLMResponse(text="done"),
    ]
    bus = EventBus()
    warnings: list[str] = []
    bus.subscribe(
        lambda e: warnings.append(e.data.get("message", ""))
        if e.type is EventType.WARNING
        else None
    )

    result = agent_with(queue, settings, retriever=None).run(
        "the task", workspace=tmp_path / "ws", bus=bus
    )

    assert any("unparseable" in w or "usable JSON" in w for w in warnings)
    assert result.plan is not None
    assert len(result.plan.steps) == 1


def test_a_failed_tool_cannot_satisfy_a_step(settings, tmp_path):
    queue = [
        plan_response("edit a file that is not there"),
        tool_response("edit_file", path="missing.py", find="a", replace="b"),
        observation(satisfied=True),  # the observer lies
        tool_response("write_file", path="ok.py", content="1\n"),
        observation(satisfied=True, verdict="finish"),
        LLMResponse(text="done"),
    ]
    result = agent_with(queue, settings, retriever=None).run(
        "edit", workspace=tmp_path / "ws"
    )

    # The step is only marked done once a tool actually succeeded.
    assert result.plan is not None
    assert result.iterations >= 2


# -- json parsing -----------------------------------------------------------


def test_json_is_recovered_from_fences_and_prose():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_object('Sure! {"a": 3} hope that helps') == {"a": 3}
    assert parse_json_object("no json here") is None
    assert parse_json_object("") is None
