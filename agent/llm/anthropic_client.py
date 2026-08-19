"""Anthropic Messages API client.

Two things worth knowing:

* The system prompt and the tool schemas are marked with `cache_control`. They
  are identical on every turn of a run, so the cache read is close to free and
  cuts the input bill on long loops by roughly an order of magnitude.
* Retries only cover the transient failures (429, 5xx, overloaded). A 400 means
  we built a bad request and retrying just burns time.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any

from agent.config import Settings
from agent.errors import LLMError
from agent.llm.base import LLMResponse, Message
from agent.llm.pricing import estimate_cost
from agent.types import ToolCall, Usage

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504}


class AnthropicClient:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

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
        model = model or self.settings.model
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "temperature": (
                self.settings.temperature if temperature is None else temperature
            ),
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            cached = [dict(t) for t in tools]
            cached[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = cached

        raw = self._send(payload)
        return self._parse(raw, model)

    # -- transport ----------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> Any:
        last: Exception | None = None
        for attempt in range(self.settings.llm_max_retries):
            try:
                return self._client.messages.create(**payload)
            except Exception as exc:  # noqa: BLE001 - provider SDK error surface varies
                last = exc
                if not _is_retryable(exc):
                    raise LLMError(f"{type(exc).__name__}: {exc}") from exc
                backoff = min(2**attempt, 16) + random.uniform(0, 0.5)
                log.warning(
                    "llm call failed (%s), retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    self.settings.llm_max_retries,
                    backoff,
                )
                time.sleep(backoff)
        raise LLMError(f"exhausted retries: {last}")

    # -- parsing ------------------------------------------------------------

    def _parse(self, raw: Any, model: str) -> LLMResponse:
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_content: list[dict[str, Any]] = []

        for block in raw.content:
            kind = getattr(block, "type", None)
            if kind == "text":
                text_parts.append(block.text)
                raw_content.append({"type": "text", "text": block.text})
            elif kind == "tool_use":
                tool_calls.append(
                    ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {}))
                )
                raw_content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": dict(block.input or {}),
                    }
                )

        usage = _usage_from(raw, model)
        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=tool_calls,
            stop_reason=getattr(raw, "stop_reason", "end_turn") or "end_turn",
            usage=usage,
            raw_content=raw_content,
        )


def _usage_from(raw: Any, model: str) -> Usage:
    u = getattr(raw, "usage", None)
    if u is None:
        return Usage(requests=1)
    inp = getattr(u, "input_tokens", 0) or 0
    out = getattr(u, "output_tokens", 0) or 0
    cached = getattr(u, "cache_read_input_tokens", 0) or 0
    return Usage(
        input_tokens=inp,
        output_tokens=out,
        cache_read_tokens=cached,
        requests=1,
        usd=estimate_cost(model, inp, out, cached),
    )


def _is_retryable(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if isinstance(status, int):
        return status in RETRYABLE_STATUS
    # Connection-level failures carry no status but are always worth one more go.
    name = type(exc).__name__.lower()
    return any(k in name for k in ("connection", "timeout", "overloaded", "apiconnection"))
