"""Indexing pipeline: read -> chunk -> embed -> upsert.

Documents are keyed by source and fingerprinted with a checksum, so re-running
`aca index` over a repo only re-embeds the files that actually changed. On a
2,000 file checkout that is the difference between a coffee break and a second.
"""

from __future__ import annotations

import fnmatch
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent.memory.embeddings import Embedder
from agent.rag.chunker import chunk_text
from agent.rag.store import ChunkStore, checksum

log = logging.getLogger(__name__)

DEFAULT_INCLUDE = (
    "*.py", "*.md", "*.rst", "*.txt", "*.toml", "*.yaml", "*.yml",
    "*.js", "*.ts", "*.tsx", "*.jsx", "*.go", "*.rs", "*.java", "*.sql", "*.sh",
)

DEFAULT_EXCLUDE = (
    "*/.git/*", "*/node_modules/*", "*/.venv/*", "*/venv/*", "*/__pycache__/*",
    "*/dist/*", "*/build/*", "*/.mypy_cache/*", "*/.pytest_cache/*", "*.lock",
)

MAX_FILE_BYTES = 400_000


@dataclass
class IndexReport:
    indexed: int = 0
    skipped: int = 0
    chunks: int = 0
    sources: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def merge(self, other: IndexReport) -> IndexReport:
        self.indexed += other.indexed
        self.skipped += other.skipped
        self.chunks += other.chunks
        self.sources.extend(other.sources)
        self.errors.extend(other.errors)
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "indexed": self.indexed,
            "skipped": self.skipped,
            "chunks": self.chunks,
            "errors": self.errors,
        }


class Indexer:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder,
        *,
        chunk_tokens: int = 400,
        chunk_overlap: int = 60,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.chunk_tokens = chunk_tokens
        self.chunk_overlap = chunk_overlap

    def index_text(
        self,
        source: str,
        text: str,
        *,
        title: str = "",
        kind: str = "doc",
        metadata: dict[str, Any] | None = None,
    ) -> IndexReport:
        report = IndexReport()
        digest = checksum(text)
        doc_id, changed = self.store.upsert_document(
            source=source, title=title or source, kind=kind, digest=digest, metadata=metadata
        )
        if not changed:
            report.skipped += 1
            return report

        chunks = chunk_text(
            text, target_tokens=self.chunk_tokens, overlap_tokens=self.chunk_overlap
        )
        if not chunks:
            report.skipped += 1
            return report

        vectors = self.embedder.embed([c.text for c in chunks])
        written = self.store.replace_chunks(doc_id, source, chunks, vectors)
        report.indexed += 1
        report.chunks += written
        report.sources.append(source)
        return report

    def index_file(self, path: str | Path, *, root: Path | None = None) -> IndexReport:
        path = Path(path)
        report = IndexReport()
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                report.skipped += 1
                return report
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            report.errors.append(f"{path}: {exc}")
            return report

        source = str(path.relative_to(root)) if root else str(path)
        return self.index_text(source, text, title=path.name, kind="file")

    def index_directory(
        self,
        directory: str | Path,
        *,
        include: tuple[str, ...] = DEFAULT_INCLUDE,
        exclude: tuple[str, ...] = DEFAULT_EXCLUDE,
    ) -> IndexReport:
        root = Path(directory).expanduser().resolve()
        report = IndexReport()
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            as_posix = path.as_posix()
            if any(fnmatch.fnmatch(as_posix, pattern) for pattern in exclude):
                continue
            if not any(fnmatch.fnmatch(path.name, pattern) for pattern in include):
                continue
            report.merge(self.index_file(path, root=root))
        log.info(
            "indexed %d files (%d chunks), skipped %d",
            report.indexed, report.chunks, report.skipped,
        )
        return report


def build_indexer(settings: Any | None = None) -> Indexer:
    from agent.config import get_settings
    from agent.memory.embeddings import build_embedder
    from agent.rag.store import build_chunk_store

    settings = settings or get_settings()
    return Indexer(
        build_chunk_store(settings),
        build_embedder(settings),
        chunk_tokens=settings.chunk_tokens,
        chunk_overlap=settings.chunk_overlap,
    )
