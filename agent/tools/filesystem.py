"""File tools. All paths are workspace-relative; the sandbox enforces that."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agent.errors import ToolError
from agent.tools.registry import ToolContext, registry


class ListDirArgs(BaseModel):
    path: str = Field(default=".", description="Directory to list, relative to the workspace")


@registry.register(ListDirArgs)
def list_dir(ctx: ToolContext, args: ListDirArgs) -> str:
    """List the immediate contents of a directory in the workspace.

    Use this before writing anything, so you never clobber a file you have not seen.
    """
    entries = ctx.workspace.tree(args.path)
    if not entries:
        return f"{args.path} is empty"
    return "\n".join(entries)


class ReadFileArgs(BaseModel):
    path: str = Field(description="File to read, relative to the workspace")
    start_line: int | None = Field(default=None, description="1-indexed first line to return")
    end_line: int | None = Field(default=None, description="1-indexed last line to return")


@registry.register(ReadFileArgs)
def read_file(ctx: ToolContext, args: ReadFileArgs) -> str:
    """Read a text file. Returns the content with 1-indexed line numbers prefixed.

    Pass start_line/end_line to page through a long file instead of pulling it all
    into context.
    """
    text = ctx.workspace.read(args.path)
    lines = text.splitlines()
    start = max(1, args.start_line or 1)
    end = min(len(lines), args.end_line or len(lines))
    if start > len(lines):
        return f"{args.path} has {len(lines)} lines; start_line {start} is past the end"
    width = len(str(end))
    body = "\n".join(f"{i:>{width}}| {lines[i - 1]}" for i in range(start, end + 1))
    header = f"{args.path} (lines {start}-{end} of {len(lines)})"
    return f"{header}\n{body}"


class WriteFileArgs(BaseModel):
    path: str = Field(description="File to create or overwrite, relative to the workspace")
    content: str = Field(description="Full file content. This replaces the file entirely.")


@registry.register(WriteFileArgs, mutating=True)
def write_file(ctx: ToolContext, args: WriteFileArgs) -> str:
    """Create a file, or replace one wholesale.

    Prefer edit_file when you are changing part of an existing file - a full
    rewrite of a file you only half-read is how work gets destroyed.
    """
    existed = ctx.workspace.exists(args.path)
    ctx.workspace.write(args.path, args.content)
    lines = args.content.count("\n") + 1
    verb = "overwrote" if existed else "created"
    return f"{verb} {args.path} ({lines} lines, {len(args.content)} bytes)"


class EditFileArgs(BaseModel):
    path: str = Field(description="File to edit, relative to the workspace")
    find: str = Field(description="Exact text to replace. Must appear exactly once.")
    replace: str = Field(description="Replacement text")


@registry.register(EditFileArgs, mutating=True)
def edit_file(ctx: ToolContext, args: EditFileArgs) -> str:
    """Replace one exact snippet in a file.

    The `find` string must match exactly once, including indentation. If it
    matches zero or many times the edit is rejected and nothing is written -
    read the file again and widen the snippet until it is unique.
    """
    original = ctx.workspace.read(args.path)
    hits = original.count(args.find)
    if hits == 0:
        raise ToolError(f"`find` text not present in {args.path}; re-read the file")
    if hits > 1:
        raise ToolError(
            f"`find` text matches {hits} times in {args.path}; include more surrounding "
            "context so it is unique"
        )
    updated = original.replace(args.find, args.replace, 1)
    ctx.workspace.write(args.path, updated)
    delta = updated.count("\n") - original.count("\n")
    return f"edited {args.path} ({delta:+d} lines)"


class AppendFileArgs(BaseModel):
    path: str = Field(description="File to append to, relative to the workspace")
    content: str = Field(description="Text appended verbatim to the end of the file")


@registry.register(AppendFileArgs, mutating=True)
def append_file(ctx: ToolContext, args: AppendFileArgs) -> str:
    """Append text to the end of a file, creating it if it does not exist."""
    existing = ctx.workspace.read(args.path) if ctx.workspace.exists(args.path) else ""
    if existing and not existing.endswith("\n"):
        existing += "\n"
    ctx.workspace.write(args.path, existing + args.content)
    return f"appended {len(args.content)} bytes to {args.path}"
