from __future__ import annotations

import math

from agent.memory.embeddings import cosine
from agent.memory.store import InMemoryStore
from agent.rag.chunker import chunk_text
from agent.rag.indexer import Indexer
from agent.rag.retriever import Retriever
from agent.rag.store import InMemoryChunkStore

# -- embeddings -------------------------------------------------------------


def test_embeddings_are_deterministic_and_normalised(embedder):
    a, b = embedder.embed(["retry with backoff", "retry with backoff"])
    assert a == b
    assert math.isclose(math.sqrt(sum(v * v for v in a)), 1.0, rel_tol=1e-6)


def test_similar_text_scores_above_unrelated_text(embedder):
    near, far = embedder.embed(["run the pytest suite", "the cat sat on the mat"])
    query = embedder.embed(["run pytest"])[0]
    assert cosine(query, near) > cosine(query, far)


def test_empty_text_does_not_blow_up(embedder):
    assert embedder.embed([""])[0] == [0.0] * embedder.dim


# -- memory -----------------------------------------------------------------


def test_memory_search_ranks_the_relevant_record_first(memory: InMemoryStore):
    memory.write("the build entrypoint is scripts/build.sh", kind="fact")
    memory.write("pytest needs -p no:randomly in this repo", kind="lesson")

    top = memory.search("how do I run pytest here")[0]
    assert "pytest" in top.text


def test_memory_filters_by_kind(memory: InMemoryStore):
    memory.write("prefers tabs over spaces in this repo", kind="preference")
    memory.write("the deploy script is flaky on fridays", kind="failure")

    results = memory.search("repo", kinds=["preference"])
    assert all(r.kind == "preference" for r in results)


def test_forgetting_removes_the_record(memory: InMemoryStore):
    memory_id = memory.write("something worth forgetting later")
    assert memory.forget(memory_id) is True
    assert memory.forget(memory_id) is False
    assert memory.stats()["count"] == 0


def test_use_count_is_incremented_on_recall(memory: InMemoryStore):
    memory.write("the migration command is make migrate")
    memory.search("migration command")
    memory.search("migration command")
    assert memory.stats()["count"] == 1


# -- chunking ---------------------------------------------------------------


def test_chunks_carry_their_line_range():
    text = "\n".join(f"line {i}" for i in range(1, 200))
    chunks = chunk_text(text, target_tokens=40, overlap_tokens=8)

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert all(c.end_line >= c.start_line for c in chunks)
    assert chunks[0].lines == f"{chunks[0].start_line}-{chunks[0].end_line}"


def test_chunks_overlap_so_context_is_not_cut_mid_thought():
    text = "\n".join(f"statement {i}" for i in range(1, 120))
    chunks = chunk_text(text, target_tokens=30, overlap_tokens=12)
    assert chunks[1].start_line <= chunks[0].end_line


def test_empty_input_produces_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("\n\n\n") == []


# -- retrieval --------------------------------------------------------------


def build(embedder) -> tuple[Indexer, Retriever, InMemoryChunkStore]:
    store = InMemoryChunkStore()
    indexer = Indexer(store, embedder, chunk_tokens=60, chunk_overlap=10)
    retriever = Retriever(store, embedder, top_k=4, min_score=0.2)
    return indexer, retriever, store


def test_retrieval_returns_cited_chunks(embedder):
    indexer, retriever, _ = build(embedder)
    indexer.index_text(
        "http/client.py", "def retry_with_backoff(attempt):\n    return 2 ** attempt\n"
    )

    hits = retriever.search("retry backoff")
    assert hits
    assert hits[0].citation().startswith("http/client.py")


def test_reindexing_unchanged_content_is_skipped(embedder):
    indexer, _, _ = build(embedder)
    body = "def handler():\n    return 1\n"

    first = indexer.index_text("app.py", body)
    second = indexer.index_text("app.py", body)

    assert first.indexed == 1
    assert second.indexed == 0
    assert second.skipped == 1


def test_changed_content_replaces_the_old_chunks(embedder):
    indexer, _, store = build(embedder)
    indexer.index_text("app.py", "def old():\n    return 1\n")
    indexer.index_text("app.py", "def new():\n    return 2\n")

    assert store.stats()["documents"] == 1
    assert "new" in store.vector_search(embedder.embed(["new"])[0], 1)[0].text


def test_the_relevance_floor_returns_nothing_rather_than_noise(embedder):
    store = InMemoryChunkStore()
    indexer = Indexer(store, embedder, chunk_tokens=60, chunk_overlap=10)
    indexer.index_text("recipes.md", "Fold the egg whites gently into the batter.\n")

    strict = Retriever(store, embedder, top_k=4, min_score=0.99)
    assert strict.search("kubernetes ingress controller") == []


def test_one_file_cannot_monopolise_the_results(embedder):
    store = InMemoryChunkStore()
    indexer = Indexer(store, embedder, chunk_tokens=20, chunk_overlap=4)
    indexer.index_text("big.py", "\n".join(f"def handler_{i}(): return {i}" for i in range(40)))
    indexer.index_text("small.py", "def handler_other(): return 0\n")

    retriever = Retriever(store, embedder, top_k=6, min_score=0.0)
    sources = [c.source for c in retriever.search("handler")]
    assert sources.count("big.py") <= 3
