"""Tool registration and dispatch.

A tool is a python function plus a pydantic model describing its arguments. The
model is the single source of truth: it generates the JSON schema sent to the
provider *and* validates whatever the model sends back, so a hallucinated
argument turns into an ordinary error message the agent can read and correct
instead of a stack trace.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from agent.errors import ToolError
from agent.types import ToolCall, ToolResult

if TYPE_CHECKING:  # pragma: no cover
    from agent.config import Settings
    from agent.memory.store import MemoryStore
    from agent.observability.events import EventBus
    from agent.rag.retriever import Retriever
    from agent.tools.sandbox import Workspace

A = TypeVar("A", bound=BaseModel)


@dataclass
class ToolContext:
    """Everything a tool is allowed to touch."""

    workspace: Workspace
    settings: Settings
    run_id: str = ""
    memory: MemoryStore | None = None
    retriever: Retriever | None = None
    events: EventBus | None = None
    scratch: dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool(Generic[A]):
    name: str
    description: str
    args_model: type[A]
    fn: Callable[[ToolContext, A], str]
    mutating: bool = False

    def schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        schema.setdefault("type", "object")
        return {
            "name": self.name,
            "description": self.description.strip(),
            "input_schema": schema,
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any]] = {}

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def register(
        self, args_model: type[A], *, name: str | None = None, mutating: bool = False
    ) -> Callable[[Callable[[ToolContext, A], str]], Callable[[ToolContext, A], str]]:
        def decorator(fn: Callable[[ToolContext, A], str]) -> Callable[[ToolContext, A], str]:
            tool_name = name or fn.__name__
            if tool_name in self._tools:
                raise ValueError(f"tool {tool_name!r} is already registered")
            doc = (fn.__doc__ or "").strip()
            if not doc:
                raise ValueError(f"tool {tool_name!r} needs a docstring - it is the prompt")
            self._tools[tool_name] = Tool(
                name=tool_name,
                description=doc,
                args_model=args_model,
                fn=fn,
                mutating=mutating,
            )
            return fn

        return decorator

    def get(self, name: str) -> Tool[Any]:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(f"unknown tool {name!r}") from None

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, only: list[str] | None = None) -> list[dict[str, Any]]:
        names = only or self.names()
        return [self._tools[n].schema() for n in names if n in self._tools]

    def invoke(self, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            tool = self.get(call.name)
        except ToolError as exc:
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                ok=False,
                error=f"{exc}. Available tools: {', '.join(self.names())}",
                duration_ms=elapsed(),
            )

        try:
            args = tool.args_model(**call.arguments)
        except ValidationError as exc:
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                ok=False,
                error=f"invalid arguments for {call.name}: {_explain(exc)}",
                duration_ms=elapsed(),
            )

        try:
            output = tool.fn(ctx, args)
        except ToolError as exc:
            return ToolResult(
                call_id=call.id, tool=call.name, ok=False, error=str(exc), duration_ms=elapsed()
            )
        except Exception as exc:  # noqa: BLE001 - tools must never kill the run
            return ToolResult(
                call_id=call.id,
                tool=call.name,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed(),
            )

        return ToolResult(
            call_id=call.id,
            tool=call.name,
            ok=True,
            content=str(output),
            duration_ms=elapsed(),
            metadata={"mutating": tool.mutating},
        )


def _explain(exc: ValidationError) -> str:
    bits = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        bits.append(f"{loc}: {err['msg']}")
    return "; ".join(bits)


registry = ToolRegistry()
