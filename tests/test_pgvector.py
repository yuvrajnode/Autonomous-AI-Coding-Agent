"""pgvector round trip.

Skipped unless a database is actually reachable, so a normal `pytest` on a
laptop with nothing running stays green. CI brings up the pgvector image and
these run for real - the in-memory fallbacks are convenient, but they are not
the thing that ships.

    docker compose up -d postgres && pytest tests/test_pgvector.py
"""

from __future__ import annotations

import pytest

from agent.memory import db
from agent.memory.embeddings import HashingEmbedder
from agent.memory.store import PgVectorStore
from agent.rag.chunker import chunk_text
from agent.rag.indexer import Indexer
from agent.rag.retriever import Retriever
from agent.rag.store import PgChunkStore, checksum

DIM = 256
URL = "postgresql://aca:aca@localhost:5433/aca"

pytestmark = pytest.mark.skipif(
    not db.is_available(URL), reason="postgres is not running (make db-up)"
)


@pytest.fixture(scope="module")
def migrated() -> str:
    db.migrate(URL, DIM)
    return URL


@pytest.fixture
def clean(migrated: str) -> str:
    with db.connect(migrated) as conn:
        conn.execute("TRUNCATE memories, chunks, documents CASCADE")
    return migrated


def test_migration_is_idempotent(migrated: str):
    first = db.migrate(migrated, DIM)
    second = db.migrate(migrated, DIM)
    assert first and len(first) == len(second)


def test_memories_round_trip_through_pgvector(clean: str):
    store = PgVectorStore(clean, HashingEmbedder(DIM))
    store.write("the deploy job needs AWS_PROFILE set to staging", kind="lesson")
    store.write("frontend tests live under web/tests", kind="fact")

    hits = store.search("which profile does the deploy need", k=2)
    assert hits
    assert "AWS_PROFILE" in hits[0].text
    assert store.stats()["backend"] == "pgvector"


def test_forgetting_a_memory_deletes_the_row(clean: str):
    store = PgVectorStore(clean, HashingEmbedder(DIM))
    memory_id = store.write("something that turned out to be wrong", kind="failure")

    assert store.forget(memory_id) is True
    assert store.forget(memory_id) is False


def test_hybrid_retrieval_finds_an_exact_identifier(clean: str):
    embedder = HashingEmbedder(DIM)
    store = PgChunkStore(clean)
    indexer = Indexer(store, embedder, chunk_tokens=80, chunk_overlap=10)
    indexer.index_text(
        "http/client.py",
        "class RetryPolicy:\n"
        "    def next_delay(self, attempt: int) -> float:\n"
        "        return min(2 ** attempt, 30.0)\n",
    )
    indexer.index_text("docs/intro.md", "This project sends requests over HTTP.\n")

    hits = Retriever(store, embedder, top_k=3, min_score=0.0).search("RetryPolicy next_delay")
    assert hits
    assert hits[0].source == "http/client.py"
    assert hits[0].metadata.get("lines")


def test_reindexing_replaces_chunks_rather_than_duplicating(clean: str):
    embedder = HashingEmbedder(DIM)
    store = PgChunkStore(clean)
    indexer = Indexer(store, embedder, chunk_tokens=80, chunk_overlap=10)

    indexer.index_text("app.py", "def one():\n    return 1\n")
    indexer.index_text("app.py", "def two():\n    return 2\n")

    stats = store.stats()
    assert stats["documents"] == 1
    assert stats["chunks"] == len(chunk_text("def two():\n    return 2\n", target_tokens=80))


def test_checksums_are_stable():
    assert checksum("abc") == checksum("abc")
    assert checksum("abc") != checksum("abd")
