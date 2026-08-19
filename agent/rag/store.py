"""Storage for the retrieval index.

Postgres does two searches: cosine over pgvector, and full-text over a stored
tsvector. Keeping both matters more than it sounds - embeddings are bad at exact
identifiers, and a developer asking about `ToolRegistry.invoke` wants the line
that literally contains it. The retriever fuses the two rankings.
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Protocol, runtime_checkable

from agent.memory import db
from agent.memory.embeddings import cosine
from agent.rag.chunker import Chunk
from agent.types import RetrievedChunk

log = logging.getLogger(__name__)


def checksum(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


@runtime_checkable
class ChunkStore(Protocol):
    def upsert_document(self, source: str, title: str, kind: str, digest: str,
                        metadata: dict[str, Any] | None = None) -> tuple[str, bool]: ...

    def replace_chunks(self, document_id: str, source: str,
                       chunks: list[Chunk], vectors: list[list[float]]) -> int: ...

    def vector_search(self, vector: list[float], k: int) -> list[RetrievedChunk]: ...

    def text_search(self, query: str, k: int) -> list[RetrievedChunk]: ...

    def documents(self) -> list[dict[str, Any]]: ...

    def stats(self) -> dict[str, Any]: ...

    def delete(self, source: str) -> bool: ...


class InMemoryChunkStore:
    backend = "in-memory"

    def __init__(self) -> None:
        self._docs: dict[str, dict[str, Any]] = {}
        self._chunks: dict[str, dict[str, Any]] = {}

    def upsert_document(self, source: str, title: str, kind: str, digest: str,
                        metadata: dict[str, Any] | None = None) -> tuple[str, bool]:
        for doc_id, doc in self._docs.items():
            if doc["source"] == source:
                if doc["checksum"] == digest:
                    return doc_id, False
                doc.update(title=title, kind=kind, checksum=digest, metadata=metadata or {})
                return doc_id, True
        doc_id = f"doc_{uuid.uuid4().hex[:12]}"
        self._docs[doc_id] = {
            "id": doc_id, "source": source, "title": title, "kind": kind,
            "checksum": digest, "metadata": metadata or {},
        }
        return doc_id, True

    def replace_chunks(self, document_id: str, source: str,
                       chunks: list[Chunk], vectors: list[list[float]]) -> int:
        for cid in [c for c, v in self._chunks.items() if v["document_id"] == document_id]:
            self._chunks.pop(cid)
        for chunk, vector in zip(chunks, vectors, strict=True):
            cid = f"chk_{uuid.uuid4().hex[:12]}"
            self._chunks[cid] = {
                "id": cid, "document_id": document_id, "source": source,
                "text": chunk.text, "ordinal": chunk.ordinal, "embedding": vector,
                "metadata": {"lines": chunk.lines, **chunk.metadata},
            }
        return len(chunks)

    def vector_search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        scored = [
            (cosine(vector, row["embedding"]), row) for row in self._chunks.values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_to_chunk(row, score) for score, row in scored[:k]]

    def text_search(self, query: str, k: int) -> list[RetrievedChunk]:
        terms = {t for t in query.lower().split() if len(t) > 2}
        if not terms:
            return []
        scored = []
        for row in self._chunks.values():
            body = row["text"].lower()
            hits = sum(1 for t in terms if t in body)
            if hits:
                scored.append((hits / len(terms), row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [_to_chunk(row, score) for score, row in scored[:k]]

    def documents(self) -> list[dict[str, Any]]:
        counts: dict[str, int] = {}
        for row in self._chunks.values():
            counts[row["document_id"]] = counts.get(row["document_id"], 0) + 1
        return [
            {"source": d["source"], "title": d["title"], "kind": d["kind"],
             "chunks": counts.get(d["id"], 0)}
            for d in self._docs.values()
        ]

    def stats(self) -> dict[str, Any]:
        return {"backend": self.backend, "documents": len(self._docs),
                "chunks": len(self._chunks)}

    def delete(self, source: str) -> bool:
        doc_id = next((i for i, d in self._docs.items() if d["source"] == source), None)
        if doc_id is None:
            return False
        self._docs.pop(doc_id)
        for cid in [c for c, v in self._chunks.items() if v["document_id"] == doc_id]:
            self._chunks.pop(cid)
        return True


class PgChunkStore:
    backend = "pgvector"

    def __init__(self, url: str) -> None:
        self.url = url

    def upsert_document(self, source: str, title: str, kind: str, digest: str,
                        metadata: dict[str, Any] | None = None) -> tuple[str, bool]:
        with db.connect(self.url) as conn:
            row = conn.execute(
                "SELECT id, checksum FROM documents WHERE source = %s", (source,)
            ).fetchone()
            if row and row[1] == digest:
                return row[0], False
            if row:
                conn.execute(
                    "UPDATE documents SET title=%s, kind=%s, checksum=%s, metadata=%s::jsonb, "
                    "indexed_at=now() WHERE id=%s",
                    (title, kind, digest, json.dumps(metadata or {}), row[0]),
                )
                return row[0], True
            doc_id = f"doc_{uuid.uuid4().hex[:12]}"
            conn.execute(
                "INSERT INTO documents (id, source, title, kind, checksum, metadata) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb)",
                (doc_id, source, title, kind, digest, json.dumps(metadata or {})),
            )
            return doc_id, True

    def replace_chunks(self, document_id: str, source: str,
                       chunks: list[Chunk], vectors: list[list[float]]) -> int:
        with db.connect(self.url) as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
            for chunk, vector in zip(chunks, vectors, strict=True):
                conn.execute(
                    "INSERT INTO chunks (id, document_id, ordinal, text, embedding, metadata) "
                    "VALUES (%s, %s, %s, %s, %s::vector, %s::jsonb)",
                    (
                        f"chk_{uuid.uuid4().hex[:12]}",
                        document_id,
                        chunk.ordinal,
                        chunk.text,
                        db.vector_literal(vector),
                        json.dumps({"lines": chunk.lines, "source": source, **chunk.metadata}),
                    ),
                )
        return len(chunks)

    def vector_search(self, vector: list[float], k: int) -> list[RetrievedChunk]:
        literal = db.vector_literal(vector)
        with db.connect(self.url) as conn:
            rows = conn.execute(
                """
                SELECT c.id, d.source, c.text, c.metadata,
                       1 - (c.embedding <=> %s::vector) AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id
                ORDER BY c.embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, k),
            ).fetchall()
        return [
            RetrievedChunk(id=r[0], source=r[1], text=r[2], metadata=r[3] or {},
                           score=float(r[4]))
            for r in rows
        ]

    def text_search(self, query: str, k: int) -> list[RetrievedChunk]:
        with db.connect(self.url) as conn:
            rows = conn.execute(
                """
                SELECT c.id, d.source, c.text, c.metadata,
                       ts_rank(c.tsv, plainto_tsquery('english', %s)) AS score
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE c.tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                (query, query, k),
            ).fetchall()
        return [
            RetrievedChunk(id=r[0], source=r[1], text=r[2], metadata=r[3] or {},
                           score=float(r[4]))
            for r in rows
        ]

    def documents(self) -> list[dict[str, Any]]:
        with db.connect(self.url) as conn:
            rows = conn.execute(
                "SELECT d.source, d.title, d.kind, count(c.id) "
                "FROM documents d LEFT JOIN chunks c ON c.document_id = d.id "
                "GROUP BY d.id ORDER BY d.source"
            ).fetchall()
        return [
            {"source": r[0], "title": r[1], "kind": r[2], "chunks": int(r[3])} for r in rows
        ]

    def stats(self) -> dict[str, Any]:
        with db.connect(self.url) as conn:
            docs = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
            chunks = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return {"backend": self.backend, "documents": int(docs), "chunks": int(chunks)}

    def delete(self, source: str) -> bool:
        with db.connect(self.url) as conn:
            result = conn.execute("DELETE FROM documents WHERE source = %s", (source,))
        return bool(getattr(result, "rowcount", 0))


def _to_chunk(row: dict[str, Any], score: float) -> RetrievedChunk:
    return RetrievedChunk(
        id=row["id"], source=row["source"], text=row["text"],
        score=float(score), metadata=row["metadata"],
    )


def build_chunk_store(settings: Any | None = None) -> ChunkStore:
    from agent.config import get_settings

    settings = settings or get_settings()
    if db.is_available(settings.database_url):
        return PgChunkStore(settings.database_url)
    log.warning("postgres unreachable - the knowledge index will live in memory only")
    return InMemoryChunkStore()
