"""The sandbox is the security boundary, so it gets the most hostile tests."""

from __future__ import annotations

import pytest

from agent.errors import SandboxViolation
from agent.tools.sandbox import Workspace


def test_writes_and_reads_relative_paths(workspace: Workspace):
    workspace.write("pkg/mod.py", "x = 1\n")
    assert workspace.read("pkg/mod.py") == "x = 1\n"
    assert "pkg/mod.py" in workspace.walk()


@pytest.mark.parametrize(
    "path",
    ["/etc/passwd", "../outside.txt", "pkg/../../outside.txt", "a/../../b"],
)
def test_rejects_escapes(workspace: Workspace, path: str):
    with pytest.raises(SandboxViolation):
        workspace.resolve(path)


def test_rejects_symlink_pointing_outside(workspace: Workspace, tmp_path):
    secret = tmp_path / "secret.txt"
    secret.write_text("password")
    (workspace.root / "link.txt").symlink_to(secret)

    with pytest.raises(SandboxViolation):
        workspace.read("link.txt")


def test_refuses_oversized_reads(tmp_path):
    ws = Workspace(tmp_path / "ws", max_file_bytes=50)
    (ws.root / "big.txt").write_text("y" * 200)
    with pytest.raises(SandboxViolation, match="read limit"):
        ws.read("big.txt")


def test_refuses_oversized_writes(tmp_path):
    ws = Workspace(tmp_path / "ws", max_file_bytes=50)
    with pytest.raises(SandboxViolation):
        ws.write("big.txt", "y" * 200)


def test_refuses_binary_files(workspace: Workspace):
    (workspace.root / "logo.png").write_bytes(b"\x89PNG\r\n")
    with pytest.raises(SandboxViolation, match="binary"):
        workspace.read("logo.png")


def test_walk_skips_noise_directories(workspace: Workspace):
    workspace.write("src/app.py", "1")
    (workspace.root / "node_modules").mkdir()
    (workspace.root / "node_modules" / "junk.js").write_text("1")

    assert workspace.walk() == ["src/app.py"]


def test_tree_marks_directories(workspace: Workspace):
    workspace.write("src/app.py", "1")
    workspace.write("README.md", "hi")

    entries = workspace.tree(".")
    assert entries[0] == "src/"
    assert any(e.startswith("README.md") for e in entries)


def test_exists_is_false_for_escapes(workspace: Workspace):
    assert workspace.exists("../etc") is False
