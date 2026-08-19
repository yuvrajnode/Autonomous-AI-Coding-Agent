"""Code search: glob by name, grep by regex. Both are capped so a broad pattern
cannot flood the context window."""

from __future__ import annotations

import fnmatch
import re

from pydantic import BaseModel, Field

from agent.errors import ToolError
from agent.tools.registry import ToolContext, registry

MAX_HITS = 60


class FindFilesArgs(BaseModel):
    pattern: str = Field(description="Glob pattern, e.g. '*.py' or 'src/**/test_*.py'")


@registry.register(FindFilesArgs)
def find_files(ctx: ToolContext, args: FindFilesArgs) -> str:
    """Find files in the workspace whose path matches a glob pattern."""
    paths = ctx.workspace.walk()
    matches = [
        p
        for p in paths
        if fnmatch.fnmatch(p, args.pattern) or fnmatch.fnmatch(p.rsplit("/", 1)[-1], args.pattern)
    ]
    if not matches:
        return f"no files match {args.pattern!r} (workspace holds {len(paths)} files)"
    shown = matches[:MAX_HITS]
    out = "\n".join(shown)
    if len(matches) > len(shown):
        out += f"\n... {len(matches) - len(shown)} more"
    return out


class GrepArgs(BaseModel):
    pattern: str = Field(description="Python regular expression")
    glob: str = Field(default="*", description="Restrict the search to files matching this glob")
    ignore_case: bool = Field(default=False, description="Case-insensitive match")


@registry.register(GrepArgs)
def grep(ctx: ToolContext, args: GrepArgs) -> str:
    """Search file contents with a regular expression.

    Returns `path:line: text` for each hit. This is the cheapest way to orient
    yourself in an unfamiliar codebase - reach for it before reading whole files.
    """
    try:
        rx = re.compile(args.pattern, re.IGNORECASE if args.ignore_case else 0)
    except re.error as exc:
        raise ToolError(f"bad regex: {exc}") from exc

    hits: list[str] = []
    scanned = 0
    for rel in ctx.workspace.walk():
        if not fnmatch.fnmatch(rel, args.glob) and not fnmatch.fnmatch(
            rel.rsplit("/", 1)[-1], args.glob
        ):
            continue
        try:
            text = ctx.workspace.read(rel)
        except Exception:  # noqa: BLE001 - unreadable file just gets skipped
            continue
        scanned += 1
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                if len(hits) >= MAX_HITS:
                    hits.append(f"... stopped at {MAX_HITS} hits, narrow the pattern")
                    return "\n".join(hits)
    if not hits:
        return f"no matches for {args.pattern!r} across {scanned} files"
    return "\n".join(hits)
