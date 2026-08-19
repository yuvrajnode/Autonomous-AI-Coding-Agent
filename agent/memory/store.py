"""Long-term memory.

What the agent chooses to remember is deliberately narrow: lessons, durable
facts about a codebase, and failures worth not repeating. Transcripts are not
memory - they are traces, and they live on disk.

Recall is scored on similarity with a small recency and usefulness bonus, so a
memory that keeps proving useful floats up over one that was written once and
never touched again.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from typing import Any, Protocol, runtime_checkable

from agent.memory import db
from agent.memory.embeddings import Embedder, cosine
from agent.types import MemoryRecord

log = logging.getLogger(__name__)

VALID_KINDS = {"lesson", "fact", "preference", "failure"}
HALF_LIFE_DAYS = 45.0


def _recency_weight(age_seconds: float) -> float:
    days = max(age_seconds, 0.0) / 86_400.0
    return 0.5 ** (days / HALF_LIFE_DAYS)


def _rank(similarity: float, age_seconds: float, use_count: int) -> float:
    recency = _recency_weight(age_seconds)
    usefulness = math.log1p(use_count) / 4.0
    return round(0.80 * similarity + 0.12 * recency + 0.08 * min(usefulness, 1.0), 6)


@runtime_checkable
class MemoryStore(Protocol):
    def write(self, text: str, kind: str = "lesson",
              metadata: dict[str, Any] | None = None) -> str: ...

    def search(self, query: str, k: int = 5, kinds: list[str] | None = None
               ) -> list[MemoryRecord]: ...

    def recent(self, limit: int = 20) -> list[MemoryRecord]: ...

    def stats(self) -> dict[str, Any]: ...

    def forget(self, memory_id: str) -> bool: ...


class InMemoryStore:
    """Process-local store. Used by the tests and whenever postgres is down."""

    backend = "in-memory"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._rows: dict[str, dict[str, Any]] = {}

    def write(self, text: str, kind: str = "lesson",
              metadata: dict[str, Any] | None = None) -> str:
        kind = kind if kind in VALID_KINDS else "lesson"
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        self._rows[memory_id] = {
            "id": memory_id,
            "kind": kind,
            "text": text,
            "embedding": self.embedder.embed([text])[0],
            "metadata": metadata or {},
            "created_at": time.time(),
            "use_count": 0,
        }
        return memory_id

    def search(self, query: str, k: int = 5,
               kinds: list[str] | None = None) -> list[MemoryRecord]:
        if not self._rows:
            return []
        qvec = self.embedder.embed([query])[0]
        now = time.time()
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in self._rows.values():
            if kinds and row["kind"] not in kinds:
                continue
            sim = cosine(qvec, row["embedding"])
            scored.append((_rank(sim, now - row["created_at"], row["use_count"]), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:k]
        for _, row in top:
            row["use_count"] += 1
        return [
            MemoryRecord(
                id=row["id"], kind=row["kind"], text=row["text"], score=score,
                created_at=row["created_at"], metadata=row["metadata"],
            )
            for score, row in top
        ]

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        rows = sorted(self._rows.values(), key=lambda r: r["created_at"], reverse=True)
        return [
            MemoryRecord(id=r["id"], kind=r["kind"], text=r["text"],
                         created_at=r["created_at"], metadata=r["metadata"])
            for r in rows[:limit]
        ]

    def stats(self) -> dict[str, Any]:
        by_kind: dict[str, int] = {}
        for row in self._rows.values():
            by_kind[row["kind"]] = by_kind.get(row["kind"], 0) + 1
        return {"backend": self.backend, "count": len(self._rows), "by_kind": by_kind}

    def forget(self, memory_id: str) -> bool:
        return self._rows.pop(memory_id, None) is not None


class PgVectorStore:
    """pgvector-backed store. Cosine distance in SQL, re-ranked in python."""

    backend = "pgvector"

    def __init__(self, url: str, embedder: Embedder) -> None:
        self.url = url
        self.embedder = embedder

    def write(self, text: str, kind: str = "lesson",
              metadata: dict[str, Any] | None = None) -> str:
        kind = kind if kind in VALID_KINDS else "lesson"
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        vec = db.vector_literal(self.embedder.embed([text])[0])
        meta = metadata or {}
        with db.connect(self.url) as conn:
            conn.execute(
                """
                INSERT INTO memories (id, kind, text, embedding, metadata, run_id)
                VALUES (%s, %s, %s, %s::vector, %s::jsonb, %s)
                """,
                (memory_id, kind, text, vec, json.dumps(meta), meta.get("run_id")),
            )
        return memory_id

    def search(self, query: str, k: int = 5,
               kinds: list[str] | None = None) -> list[MemoryRecord]:
        vec = db.vector_literal(self.embedder.embed([query])[0])
        sql = """
            SELECT id, kind, text, metadata,
                   EXTRACT(EPOCH FROM created_at) AS created_epoch,
                   use_count,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM memories
        """
        params: list[Any] = [vec]
        if kinds:
            sql += " WHERE kind = ANY(%s)"
            params.append(list(kinds))
        sql += " ORDER BY embedding <=> %s::vector LIMIT %s"
        params.extend([vec, k * 3])

        with db.connect(self.url) as conn:
            rows = conn.execute(sql, params).fetchall()

        now = time.time()
        records = [
            MemoryRecord(
                id=r[0], kind=r[1], text=r[2], metadata=r[3] or {},
                created_at=float(r[4]),
                score=_rank(float(r[6]), now - float(r[4]), int(r[5])),
            )
            for r in rows
        ]
        records.sort(key=lambda m: m.score, reverse=True)
        top = records[:k]
        if top:
            self._touch([m.id for m in top])
        return top

    def _touch(self, ids: list[str]) -> None:
        with db.connect(self.url) as conn:
            conn.execute(
                "UPDATE memories SET use_count = use_count + 1, last_used_at = now() "
                "WHERE id = ANY(%s)",
                (ids,),
            )

    def recent(self, limit: int = 20) -> list[MemoryRecord]:
        with db.connect(self.url) as conn:
            rows = conn.execute(
                "SELECT id, kind, text, metadata, EXTRACT(EPOCH FROM created_at) "
                "FROM memories ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [
            MemoryRecord(id=r[0], kind=r[1], text=r[2], metadata=r[3] or {},
                         created_at=float(r[4]))
            for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        with db.connect(self.url) as conn:
            rows = conn.execute("SELECT kind, count(*) FROM memories GROUP BY kind").fetchall()
        by_kind = {r[0]: int(r[1]) for r in rows}
        return {"backend": self.backend, "count": sum(by_kind.values()), "by_kind": by_kind}

    def forget(self, memory_id: str) -> bool:
        with db.connect(self.url) as conn:
            result = conn.execute("DELETE FROM memories WHERE id = %s", (memory_id,))
        return bool(getattr(result, "rowcount", 0))


def build_memory_store(settings: Any | None = None,
                       embedder: Embedder | None = None) -> MemoryStore:
    """Prefer pgvector, degrade to in-memory rather than failing the run."""
    from agent.config import get_settings
    from agent.memory.embeddings import build_embedder

    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)

    if db.is_available(settings.database_url):
        return PgVectorStore(settings.database_url, embedder)

    log.warning(
        "postgres is unreachable - falling back to in-memory memory. "
        "Nothing written this run will survive it. Start it with `make db-up`."
    )
    return InMemoryStore(embedder)
