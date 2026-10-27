"""Anthropic Messages API client.

Two things worth knowing:

* The system prompt and the tool schemas are marked with `cache_control`. They
  are identical on every turn of a run, so the cache read is close to free and
  cuts the input bill on long loops by roughly an order of magnitude.
* Retries only cover the transient failures (429, 5xx, overloaded). A 400 means
  we built a bad request and retrying just burns time. The SDK's own retries are
  switched off so the two loops do not multiply.
* No sampling parameters. Current models and SDKs reject `temperature` outright.
* Thinking is always on; `effort` is the only dial. Thinking blocks are kept in
  `raw_content` and replayed verbatim, which only works because the actor's
  system prompt, tools and history never change mid-run - keep it append-only.
* Safety classifiers can decline a request (HTTP 200, `stop_reason: "refusal"`).
  Models that support it get server-side fallbacks, so a false positive is
  retried on another model instead of ending the run; a refusal that survives
  that raises.
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

# 529 is the API's `overloaded_error`.
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

# Models that accept `fallbacks: "default"`; anything else would 400 on it.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicClient:
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        if client is not None:
            self._client = client
        else:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=settings.anthropic_api_key, max_retries=0
            )

    def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        model = model or self.settings.model
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "output_config": {"effort": self.settings.effort},
            "system": [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ],
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if tools:
            cached = [dict(t) for t in tools]
            cached[-1]["cache_control"] = {"type": "ephemeral"}
            payload["tools"] = cached
        if model in FALLBACK_MODELS:
            payload["betas"] = [FALLBACK_BETA]
            payload["fallbacks"] = "default"

        raw = self._send(payload)
        return self._parse(raw, model)

    # -- transport ----------------------------------------------------------

    def _send(self, payload: dict[str, Any]) -> Any:
        last: Exception | None = None
        for attempt in range(self.settings.llm_max_retries):
            try:
                return self._client.beta.messages.create(**payload)
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
        if getattr(raw, "stop_reason", None) == "refusal":
            category = getattr(getattr(raw, "stop_details", None), "category", None)
            raise LLMError(f"{model} declined the request (refusal, category: {category})")

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
            elif kind == "thinking":
                raw_content.append(
                    {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
                )
            elif kind == "redacted_thinking":
                raw_content.append({"type": "redacted_thinking", "data": block.data})

        # After a fallback the reply came from another model; bill at its rate.
        usage = _usage_from(raw, getattr(raw, "model", None) or model)
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
