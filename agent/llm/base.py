"""Provider-agnostic chat interface.

The rest of the codebase only knows about `LLMClient`, `Message` and
`LLMResponse`. Swapping providers, or dropping in a scripted client for tests,
never touches the loop.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from agent.types import ToolCall, Usage


class Message(BaseModel):
    role: str
    content: Any

    @classmethod
    def user(cls, text: str) -> Message:
        return cls(role="user", content=text)

    @classmethod
    def assistant(cls, content: Any) -> Message:
        return cls(role="assistant", content=content)

    @classmethod
    def tool_results(cls, blocks: list[dict[str, Any]]) -> Message:
        return cls(role="user", content=blocks)


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: str = "end_turn"
    usage: Usage = Field(default_factory=Usage)
    raw_content: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLMClient(Protocol):
    """Minimal surface the agent needs from a chat model."""

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...


def build_client(settings: Any | None = None) -> LLMClient:
    """Pick a client from configuration.

    Falls back to the scripted client when no API key is present so that the
    dashboard, the tests and `aca demo` all work on a fresh checkout.
    """
    from agent.config import get_settings

    settings = settings or get_settings()
    if settings.has_llm_credentials:
        from agent.llm.anthropic_client import AnthropicClient

        return AnthropicClient(settings)

    from agent.llm.scripted import ScriptedClient

    return ScriptedClient.demo()
