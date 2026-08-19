"""A deterministic stand-in for a real model.

Used in three places: unit tests, CI (no API key), and `aca demo`, which lets
someone clone the repo and watch the dashboard fill up without paying for
tokens. It answers based on which role the system prompt announces.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from agent.llm.base import LLMResponse, Message
from agent.types import ToolCall, Usage

Responder = Callable[[str, list[Message]], LLMResponse]


class ScriptedClient:
    """Serves queued responses first, then falls back to a responder function."""

    def __init__(
        self,
        queue: list[LLMResponse] | None = None,
        responder: Responder | None = None,
    ) -> None:
        self.queue = list(queue or [])
        self.responder = responder
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        self.calls.append({"system": system, "messages": messages, "tools": tools})
        if self.queue:
            return self.queue.pop(0)
        if self.responder:
            return self.responder(system, messages)
        return LLMResponse(text="", stop_reason="end_turn", usage=Usage(requests=1))

    # -- canned scenarios ---------------------------------------------------

    @classmethod
    def demo(cls) -> ScriptedClient:
        return cls(responder=_demo_responder())


def _role_of(system: str) -> str:
    head = system[:200].upper()
    for role in ("PLANNER", "OBSERVER", "SUMMARISER"):
        if role in head:
            return role.lower()
    return "actor"


def _demo_responder() -> Responder:
    seen: dict[str, int] = defaultdict(int)

    def respond(system: str, messages: list[Message]) -> LLMResponse:
        role = _role_of(system)
        n = seen[role]
        seen[role] += 1
        usage = Usage(input_tokens=900, output_tokens=180, requests=1, usd=0.0)

        if role == "planner":
            plan = {
                "assumptions": ["Working inside an empty scratch workspace."],
                "steps": [
                    {
                        "description": "Inspect the workspace to see what already exists",
                        "rationale": "Never write over files I have not looked at.",
                        "expected_tool": "list_dir",
                    },
                    {
                        "description": "Write the module the task asks for",
                        "rationale": "This is the actual deliverable.",
                        "expected_tool": "write_file",
                    },
                    {
                        "description": "Run the tests and confirm they pass",
                        "rationale": "A change is not done until it is verified.",
                        "expected_tool": "run_python",
                    },
                ],
            }
            return LLMResponse(text=json.dumps(plan), usage=usage)

        if role == "observer":
            body = {
                "summary": "Tool call returned cleanly; the step looks satisfied.",
                "step_satisfied": True,
                "verdict": "continue",
                "evidence": ["exit code 0"],
                "concerns": [],
            }
            return LLMResponse(text=json.dumps(body), usage=usage)

        if role == "summariser":
            return LLMResponse(
                text=(
                    "Created solution.py with a fizzbuzz(n) helper covering the three "
                    "divisibility cases, then executed it with run_python: fizzbuzz(15) "
                    "printed FizzBuzz and the process exited 0. Nothing else in the "
                    "workspace was touched. No test file was added, so this is verified "
                    "by execution rather than by a suite."
                ),
                usage=usage,
            )

        # actor
        script: list[ToolCall] = [
            ToolCall(name="list_dir", arguments={"path": "."}),
            ToolCall(
                name="write_file",
                arguments={
                    "path": "solution.py",
                    "content": (
                        "def fizzbuzz(n: int) -> str:\n"
                        '    if n % 15 == 0:\n        return "FizzBuzz"\n'
                        '    if n % 3 == 0:\n        return "Fizz"\n'
                        '    if n % 5 == 0:\n        return "Buzz"\n'
                        "    return str(n)\n"
                    ),
                },
            ),
            ToolCall(
                name="run_python",
                arguments={"code": "from solution import fizzbuzz; print(fizzbuzz(15))"},
            ),
            ToolCall(
                name="submit_result",
                arguments={
                    "summary": "fizzbuzz(15) printed FizzBuzz, exit code 0.",
                    "artifacts": ["solution.py"],
                    "confidence": 0.9,
                },
            ),
        ]
        if n < len(script):
            call = script[n]
            return LLMResponse(
                text=f"Calling {call.name}.",
                tool_calls=[call],
                stop_reason="tool_use",
                usage=usage,
            )
        return LLMResponse(
            text="Wrote solution.py and verified fizzbuzz(15) returns FizzBuzz.",
            stop_reason="end_turn",
            usage=usage,
        )

    return respond
