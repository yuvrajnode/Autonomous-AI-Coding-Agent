from __future__ import annotations

import pytest

from agent.config import Settings
from agent.memory.embeddings import HashingEmbedder
from agent.memory.store import InMemoryStore
from agent.tools.registry import ToolContext
from agent.tools.sandbox import Workspace


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        anthropic_api_key="",
        workspace_root=tmp_path / "workspaces",
        trace_dir=tmp_path / "traces",
        database_url="postgresql://nobody@127.0.0.1:1/none",
        embedding_provider="hashing",
        embedding_dim=256,
        max_iterations=8,
        shell_timeout=20,
    )


@pytest.fixture
def workspace(tmp_path) -> Workspace:
    return Workspace(tmp_path / "ws")


@pytest.fixture
def embedder() -> HashingEmbedder:
    return HashingEmbedder(dim=256)


@pytest.fixture
def memory(embedder) -> InMemoryStore:
    return InMemoryStore(embedder)


@pytest.fixture
def ctx(workspace, settings, memory) -> ToolContext:
    return ToolContext(workspace=workspace, settings=settings, run_id="run_test", memory=memory)
