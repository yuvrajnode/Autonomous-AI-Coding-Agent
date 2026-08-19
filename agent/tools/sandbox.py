"""Workspace confinement.

Every file path the model produces goes through `Workspace.resolve`. It rejects
absolute paths, symlinks that escape, and `..` traversal. This is the one place
in the codebase where a mistake turns a coding agent into a shell exploit, so it
is deliberately paranoid and separately unit tested.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from agent.errors import SandboxViolation

IGNORED_DIRS = {
    ".git",
    ".hg",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
}

BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar",
    ".so", ".dylib", ".dll", ".pyc", ".woff", ".woff2", ".mp4", ".wasm",
}


class Workspace:
    """A single run's sandbox directory."""

    def __init__(self, root: str | os.PathLike[str], max_file_bytes: int = 512_000) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_file_bytes = max_file_bytes

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Workspace({self.root})"

    # -- path handling ------------------------------------------------------

    def resolve(self, relative: str) -> Path:
        if relative is None:
            raise SandboxViolation("no path given")
        candidate = Path(str(relative).strip())
        if candidate.is_absolute():
            raise SandboxViolation(f"absolute paths are not allowed: {relative}")
        if any(part == ".." for part in candidate.parts):
            raise SandboxViolation(f"path escapes the workspace: {relative}")

        target = (self.root / candidate).resolve()
        # `resolve()` follows symlinks, so this also catches a symlink planted
        # inside the workspace that points somewhere else on disk.
        if target != self.root and self.root not in target.parents:
            raise SandboxViolation(f"path escapes the workspace: {relative}")
        return target

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root)) or "."
        except ValueError:  # pragma: no cover - only reachable on a bug above
            return str(path)

    # -- io -----------------------------------------------------------------

    def read(self, relative: str) -> str:
        path = self.resolve(relative)
        if not path.exists():
            raise SandboxViolation(f"no such file: {relative}")
        if path.is_dir():
            raise SandboxViolation(f"{relative} is a directory")
        if path.suffix.lower() in BINARY_SUFFIXES:
            raise SandboxViolation(f"{relative} looks binary; refusing to read it as text")
        size = path.stat().st_size
        if size > self.max_file_bytes:
            raise SandboxViolation(
                f"{relative} is {size} bytes, over the {self.max_file_bytes} byte read limit"
            )
        return path.read_text(encoding="utf-8", errors="replace")

    def write(self, relative: str, content: str) -> Path:
        if len(content.encode("utf-8")) > self.max_file_bytes:
            raise SandboxViolation("refusing to write a file over the size limit")
        path = self.resolve(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def exists(self, relative: str) -> bool:
        try:
            return self.resolve(relative).exists()
        except SandboxViolation:
            return False

    def walk(self, relative: str = ".", max_entries: int = 400) -> list[str]:
        base = self.resolve(relative)
        if not base.exists():
            raise SandboxViolation(f"no such directory: {relative}")
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(d for d in dirnames if d not in IGNORED_DIRS)
            for name in sorted(filenames):
                out.append(self.relative(Path(dirpath) / name))
                if len(out) >= max_entries:
                    return out
        return out

    def tree(self, relative: str = ".") -> list[str]:
        """One level, directories first, marked with a trailing slash."""
        base = self.resolve(relative)
        if not base.is_dir():
            raise SandboxViolation(f"{relative} is not a directory")
        entries = [e for e in base.iterdir() if e.name not in IGNORED_DIRS]
        dirs = sorted(f"{e.name}/" for e in entries if e.is_dir())
        files = sorted(
            f"{e.name}  ({e.stat().st_size}b)" for e in entries if e.is_file()
        )
        return dirs + files

    def snapshot(self) -> dict[str, str]:
        """Cheap content map used by the eval harness to diff before/after."""
        return {p: self.read(p) for p in self.walk() if _is_text(p)}

    def destroy(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _is_text(relative: str) -> bool:
    return Path(relative).suffix.lower() not in BINARY_SUFFIXES
