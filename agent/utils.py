"""Small helpers that are needed in more than one place."""

from __future__ import annotations

import json
import re
from typing import Any

FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Best-effort extraction of a JSON object from a model response.

    Models wrap JSON in prose or fences more often than anyone would like. Try
    the whole string, then any fenced block, then the outermost pair of braces.
    Returns None rather than raising - the caller decides whether a miss is
    worth a retry or a fallback.
    """
    for candidate in _candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _candidates(text: str) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    out = [text]
    out.extend(m.strip() for m in FENCE_RE.findall(text))
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        out.append(text[start : end + 1])
    return out


def truncate(text: str, limit: int, marker: str = "...") -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    return text[: limit - len(marker)] + marker


def bullet_list(items: list[str], limit: int = 8) -> str:
    if not items:
        return "(none)"
    shown = items[:limit]
    body = "\n".join(f"- {item}" for item in shown)
    if len(items) > limit:
        body += f"\n- ... and {len(items) - limit} more"
    return body


def fingerprint(name: str, arguments: dict[str, Any]) -> str:
    """Stable key for 'has the agent already tried exactly this?'."""
    try:
        payload = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        payload = repr(sorted(arguments.items()))
    return f"{name}:{payload}"
