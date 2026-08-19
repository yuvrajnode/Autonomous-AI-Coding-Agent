"""API tests.

These run against the real app with its real lifespan; postgres is not up in CI
so the stores fall back to their in-memory implementations, which is exactly the
path a first-time contributor hits too.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("ACA_WORKSPACE_ROOT", str(tmp_path / "workspaces"))
    monkeypatch.setenv("ACA_TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("ACA_DATABASE_URL", "postgresql://nobody@127.0.0.1:1/none")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    from agent.config import get_settings

    get_settings.cache_clear()
    with TestClient(app) as test_client:
        yield test_client
    get_settings.cache_clear()


def test_health_reports_the_active_backends(client):
    body = client.get("/api/health").json()
    assert body["memory_backend"] == "in-memory"
    assert body["tools"] > 5
    assert body["llm_configured"] is False


def test_the_dashboard_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "aca" in response.text


def test_tools_are_listed_with_their_parameters(client):
    tools = client.get("/api/tools").json()["tools"]
    names = {t["name"] for t in tools}
    assert {"read_file", "write_file", "run_tests", "submit_result"} <= names
    read_file = next(t for t in tools if t["name"] == "read_file")
    assert "path" in read_file["parameters"]


def test_a_short_task_is_rejected(client):
    assert client.post("/api/runs", json={"task": "hi"}).status_code == 422


def test_a_run_starts_and_finishes(client):
    started = client.post(
        "/api/runs", json={"task": "Write a fizzbuzz helper and verify it"}
    )
    assert started.status_code == 202
    run_id = started.json()["id"]

    deadline = time.time() + 15
    while time.time() < deadline:
        detail = client.get(f"/api/runs/{run_id}").json()
        if detail["status"] in {"succeeded", "failed", "aborted"}:
            break
        time.sleep(0.1)

    assert detail["status"] == "succeeded"
    assert detail["events"]
    assert detail["summary"]
    assert any(e["type"] == "tool.called" for e in detail["events"])


def test_a_missing_run_is_a_404(client):
    assert client.get("/api/runs/run_nope").status_code == 404


def test_memory_can_be_written_read_and_forgotten(client):
    created = client.post(
        "/api/memory", json={"text": "the release script must run from the repo root"}
    )
    assert created.status_code == 201
    memory_id = created.json()["id"]

    found = client.get("/api/memory", params={"q": "release script"}).json()
    assert any(m["id"] == memory_id for m in found["memories"])

    assert client.delete(f"/api/memory/{memory_id}").status_code == 200
    assert client.delete(f"/api/memory/{memory_id}").status_code == 404


def test_indexing_a_directory_makes_it_searchable(client, tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "client.py").write_text(
        "def retry_with_backoff(attempt):\n    return 2 ** attempt\n"
    )

    report = client.post("/api/knowledge/index", json={"path": str(source)}).json()
    assert report["indexed"] == 1
    assert report["chunks"] >= 1

    hits = client.get("/api/knowledge", params={"q": "retry backoff"}).json()["chunks"]
    assert hits
    assert hits[0]["source"] == "client.py"


def test_indexing_a_missing_path_is_rejected(client):
    response = client.post("/api/knowledge/index", json={"path": "/definitely/not/here"})
    assert response.status_code == 400


def test_the_websocket_replays_the_whole_run(client):
    run_id = client.post(
        "/api/runs", json={"task": "Write a fizzbuzz helper and verify it"}
    ).json()["id"]

    types: list[str] = []
    with client.websocket_connect(f"/ws/runs/{run_id}") as socket:
        for _ in range(200):
            message = socket.receive_json()
            if message.get("type") == "run.closed":
                break
            types.append(message["type"])

    assert "run.started" in types
    assert "run.finished" in types
