"""Chunking.

Splitting on a fixed character count cuts functions in half and produces chunks
that retrieve well and read badly. This splitter walks the file line by line and
only closes a chunk at a boundary that means something - a blank line, a
top-level `def`/`class`, a markdown heading - unless the chunk has grown past
its ceiling, in which case it closes anyway.

Every chunk carries its line range so retrieved context can be cited as
`path:12-48` and a human can go look.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CODE_BOUNDARY = re.compile(r"^(def |class |async def |@|export |function |const |type |impl )")
HEADING = re.compile(r"^#{1,6}\s")

# Rough enough for budgeting; we are choosing split points, not billing anyone.
CHARS_PER_TOKEN = 4


@dataclass
class Chunk:
    text: str
    ordinal: int
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def lines(self) -> str:
        return f"{self.start_line}-{self.end_line}"


def chunk_text(
    text: str,
    *,
    target_tokens: int = 400,
    overlap_tokens: int = 60,
    hard_ceiling_multiplier: float = 1.6,
) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []

    target = target_tokens * CHARS_PER_TOKEN
    ceiling = int(target * hard_ceiling_multiplier)
    overlap = overlap_tokens * CHARS_PER_TOKEN

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_chars = 0
    start_line = 1
    ordinal = 0

    def flush(end_line: int) -> None:
        nonlocal buf, buf_chars, start_line, ordinal
        body = "\n".join(buf).strip("\n")
        if body.strip():
            chunks.append(
                Chunk(text=body, ordinal=ordinal, start_line=start_line, end_line=end_line)
            )
            ordinal += 1
        tail = _tail(buf, overlap)
        buf = list(tail)
        buf_chars = sum(len(x) + 1 for x in buf)
        start_line = max(1, end_line - len(tail) + 1)

    for i, line in enumerate(lines, start=1):
        buf.append(line)
        buf_chars += len(line) + 1

        past_target = buf_chars >= target
        at_boundary = (
            not line.strip() or bool(CODE_BOUNDARY.match(line)) or bool(HEADING.match(line))
        )
        if (past_target and at_boundary) or buf_chars >= ceiling:
            flush(i)

    if buf and "\n".join(buf).strip():
        flush(len(lines))
    return chunks


def _tail(buf: list[str], overlap_chars: int) -> list[str]:
    """Last few lines of the buffer, used as the next chunk's lead-in."""
    if overlap_chars <= 0:
        return []
    out: list[str] = []
    total = 0
    for line in reversed(buf):
        total += len(line) + 1
        out.append(line)
        if total >= overlap_chars:
            break
    return list(reversed(out))
