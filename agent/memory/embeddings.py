"""Embedding providers.

Two implementations:

* `VoyageEmbedder` - what you actually want in production.
* `HashingEmbedder` - a deterministic hashed bag-of-ngrams with no network call
  and no dependencies. It is lexical, not semantic, so it will happily miss a
  paraphrase. It exists so that CI, the test suite and a fresh clone all work
  without an API key, and it is never the default when a key is present.

Both return L2-normalised vectors, which lets every caller treat the dot
product as cosine similarity.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any, Protocol, runtime_checkable

TOKEN_RE = re.compile(r"[a-z0-9_]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int
    name: str

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _normalise(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return vec
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    """Both sides are expected to be normalised already."""
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))


class HashingEmbedder:
    """Hashed unigram + bigram features with sublinear term weighting."""

    name = "hashing"

    def __init__(self, dim: int = 512) -> None:
        if dim < 32:
            raise ValueError("dim must be at least 32")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return [0.0] * self.dim
        grams = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        counts = Counter(grams)
        vec = [0.0] * self.dim
        for gram, count in counts.items():
            digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[index] += sign * (1.0 + math.log(count))
        return _normalise(vec)


class VoyageEmbedder:
    """Voyage AI embeddings. Batched, because per-text calls are slow and costly."""

    name = "voyage"
    BATCH = 96

    def __init__(self, api_key: str, model: str = "voyage-3", dim: int = 1024,
                 client: Any | None = None) -> None:
        self.model = model
        self.dim = dim
        if client is not None:
            self._client = client
        else:
            import voyageai

            self._client = voyageai.Client(api_key=api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), self.BATCH):
            batch = texts[start : start + self.BATCH]
            result = self._client.embed(batch, model=self.model, input_type="document")
            out.extend(_normalise(list(v)) for v in result.embeddings)
        return out


def build_embedder(settings: Any | None = None) -> Embedder:
    from agent.config import get_settings

    settings = settings or get_settings()
    if settings.embedding_provider == "voyage" and settings.voyage_api_key:
        return VoyageEmbedder(
            api_key=settings.voyage_api_key,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
    return HashingEmbedder(dim=min(settings.embedding_dim, 512))
