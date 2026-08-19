"""Retrieval.

Hybrid, because neither half is enough on its own: embeddings generalise over
phrasing but miss exact identifiers, full-text nails identifiers but falls over
on a paraphrase. Both rankings are min-max normalised and blended, which keeps
the final score on a 0-1 scale that the relevance floor can be reasoned about.

The floor matters more than the ranking. Returning three weak chunks and letting
the model treat them as ground truth is exactly how a coding agent invents an
API that does not exist, so below the floor the retriever returns nothing and
the tool tells the agent to go read the file instead.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.memory.embeddings import Embedder
from agent.rag.store import ChunkStore
from agent.types import RetrievedChunk

log = logging.getLogger(__name__)

VECTOR_WEIGHT = 0.7
TEXT_WEIGHT = 0.3
MAX_PER_SOURCE = 3


def _normalise(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 if hi > 0 else 0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        *,
        top_k: int = 6,
        min_score: float = 0.25,
        hybrid: bool = True,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.top_k = top_k
        self.min_score = min_score
        self.hybrid = hybrid

    def search(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        k = k or self.top_k
        pool = k * 4

        qvec = self.embedder.embed([query])[0]
        dense = self.store.vector_search(qvec, pool)
        sparse = self.store.text_search(query, pool) if self.hybrid else []

        blended: dict[str, RetrievedChunk] = {}
        for chunk, norm in zip(dense, _normalise([c.score for c in dense]), strict=True):
            chunk.metadata = {**chunk.metadata, "dense": round(chunk.score, 4)}
            chunk.score = VECTOR_WEIGHT * norm
            blended[chunk.id] = chunk

        for chunk, norm in zip(sparse, _normalise([c.score for c in sparse]), strict=True):
            existing = blended.get(chunk.id)
            if existing is not None:
                existing.score += TEXT_WEIGHT * norm
                existing.metadata["sparse"] = round(norm, 4)
            else:
                chunk.metadata = {**chunk.metadata, "sparse": round(norm, 4)}
                chunk.score = TEXT_WEIGHT * norm
                blended[chunk.id] = chunk

        ranked = sorted(blended.values(), key=lambda c: c.score, reverse=True)
        kept = self._diversify([c for c in ranked if c.score >= self.min_score], k)
        log.debug("retrieved %d/%d chunks above floor for %r", len(kept), len(ranked), query)
        return kept

    def _diversify(self, chunks: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
        """Cap how much of the result set a single file may occupy."""
        per_source: dict[str, int] = {}
        out: list[RetrievedChunk] = []
        for chunk in chunks:
            count = per_source.get(chunk.source, 0)
            if count >= MAX_PER_SOURCE:
                continue
            per_source[chunk.source] = count + 1
            out.append(chunk)
            if len(out) >= k:
                break
        return out

    def as_context_block(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""
        parts = [f"[{c.citation()}]\n{c.text.strip()}" for c in chunks]
        return "\n\n".join(parts)


def build_retriever(settings: Any | None = None) -> Retriever:
    from agent.config import get_settings
    from agent.memory.embeddings import build_embedder
    from agent.rag.store import build_chunk_store

    settings = settings or get_settings()
    return Retriever(
        build_chunk_store(settings),
        build_embedder(settings),
        top_k=settings.retrieval_top_k,
        min_score=settings.retrieval_min_score,
    )
