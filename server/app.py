"""HTTP API and dashboard.

The API is deliberately thin - it starts runs, streams their events, and exposes
what memory and the index currently hold. All of the interesting behaviour lives
in `agent/`, which is what makes the same code path testable from pytest, usable
from the CLI, and drivable from the browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import __version__
from agent.config import get_settings
from agent.loop import Agent
from agent.observability.metrics import registry as metrics_registry
from agent.tools import registry as tool_registry
from server.runs import RunManager

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


class StartRunRequest(BaseModel):
    task: str = Field(min_length=4, max_length=4000)
    workspace: str | None = None


class MemoryRequest(BaseModel):
    text: str = Field(min_length=8, max_length=2000)
    kind: str = "fact"


class IndexRequest(BaseModel):
    path: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    from agent.memory.embeddings import build_embedder
    from agent.memory.store import build_memory_store
    from agent.rag.indexer import Indexer
    from agent.rag.retriever import Retriever
    from agent.rag.store import build_chunk_store

    # One embedder and one chunk store, shared by the indexer and the retriever.
    # Building them separately works against postgres and silently does not
    # against the in-memory fallback, where each would get its own copy.
    embedder = build_embedder(settings)
    chunk_store = build_chunk_store(settings)

    app.state.settings = settings
    app.state.memory = build_memory_store(settings, embedder)
    app.state.retriever = Retriever(
        chunk_store,
        embedder,
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
    )
    app.state.indexer = Indexer(
        chunk_store,
        embedder,
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
    )
    app.state.runs = RunManager(
        lambda: Agent(
            settings=settings, memory=app.state.memory, retriever=app.state.retriever
        )
    )
    log.info("aca %s ready (memory backend: %s)", __version__,
             getattr(app.state.memory, "backend", "?"))
    try:
        yield
    finally:
        app.state.runs.shutdown()


app = FastAPI(title="aca", version=__version__, lifespan=lifespan)


# -- pages ------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


# -- runs -------------------------------------------------------------------

@app.post("/api/runs", status_code=202)
async def start_run(payload: StartRunRequest) -> dict[str, Any]:
    record = app.state.runs.start(payload.task, payload.workspace)
    return {"id": record.id, "status": record.status.value}


@app.get("/api/runs")
async def list_runs(limit: int = 30) -> dict[str, Any]:
    return {"runs": [r.summary() for r in app.state.runs.list(limit)]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    record = app.state.runs.get(run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="no such run")
    return record.detail()


@app.websocket("/ws/runs/{run_id}")
async def stream_run(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    record = app.state.runs.get(run_id)
    if record is None:
        await websocket.send_json({"type": "error", "detail": "no such run"})
        await websocket.close()
        return

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def forward(event: Any) -> None:
        # Called from the run's worker thread, so hop back onto the loop.
        loop.call_soon_threadsafe(queue.put_nowait, event.model_dump(mode="json"))

    unsubscribe = record.bus.subscribe(forward, replay=True)
    try:
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=1.0)
            except TimeoutError:
                if record.status.terminal and queue.empty():
                    await websocket.send_json(
                        {"type": "run.closed", "data": record.summary()}
                    )
                    break
                continue
            await websocket.send_json(payload)
    except WebSocketDisconnect:
        pass
    finally:
        unsubscribe()
        await _close_quietly(websocket)


async def _close_quietly(websocket: WebSocket) -> None:
    try:
        await websocket.close()
    except RuntimeError:
        pass


# -- memory -----------------------------------------------------------------

@app.get("/api/memory")
async def list_memory(limit: int = 25, q: str | None = None) -> dict[str, Any]:
    store = app.state.memory
    records = store.search(q, k=limit) if q else store.recent(limit)
    return {
        "stats": store.stats(),
        "memories": [r.model_dump(mode="json") for r in records],
    }


@app.post("/api/memory", status_code=201)
async def add_memory(payload: MemoryRequest) -> dict[str, Any]:
    memory_id = app.state.memory.write(payload.text, kind=payload.kind)
    return {"id": memory_id}


@app.delete("/api/memory/{memory_id}")
async def forget_memory(memory_id: str) -> dict[str, Any]:
    if not app.state.memory.forget(memory_id):
        raise HTTPException(status_code=404, detail="no such memory")
    return {"deleted": memory_id}


# -- knowledge --------------------------------------------------------------

@app.get("/api/knowledge")
async def knowledge(q: str | None = None, k: int = 6) -> dict[str, Any]:
    retriever = app.state.retriever
    if q:
        chunks = retriever.search(q, k=k)
        return {"query": q, "chunks": [c.model_dump(mode="json") for c in chunks]}
    return {
        "stats": retriever.store.stats(),
        "documents": retriever.store.documents()[:200],
    }


@app.post("/api/knowledge/index")
async def index_path(payload: IndexRequest) -> dict[str, Any]:
    target = Path(payload.path).expanduser()
    if not target.exists():
        raise HTTPException(status_code=400, detail=f"{target} does not exist")
    indexer = app.state.indexer
    report = await asyncio.to_thread(
        indexer.index_directory if target.is_dir() else indexer.index_file, target
    )
    return report.as_dict()


# -- introspection ----------------------------------------------------------

@app.get("/api/tools")
async def tools() -> dict[str, Any]:
    return {
        "tools": [
            {
                "name": schema["name"],
                "description": schema["description"].split("\n")[0],
                "parameters": sorted(schema["input_schema"].get("properties", {})),
            }
            for schema in tool_registry.schemas()
        ]
    }


@app.get("/api/metrics")
async def metrics() -> dict[str, Any]:
    return {"rollup": metrics_registry.rollup(), "recent": metrics_registry.recent(20)}


@app.get("/api/evals")
async def evals() -> dict[str, Any]:
    reports = sorted(Path("eval_reports").glob("*.json"), reverse=True)
    if not reports:
        return {"report": None}
    return {"report": json.loads(reports[0].read_text(encoding="utf-8")),
            "file": reports[0].name}


@app.get("/api/health")
async def health() -> JSONResponse:
    settings = app.state.settings
    return JSONResponse(
        {
            "version": __version__,
            "model": settings.model,
            "memory_backend": getattr(app.state.memory, "backend", "unknown"),
            "index_backend": getattr(app.state.retriever.store, "backend", "unknown"),
            "llm_configured": settings.has_llm_credentials,
            "tools": len(tool_registry),
        }
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
