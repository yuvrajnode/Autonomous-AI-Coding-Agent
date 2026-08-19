-- Schema for long-term memory and the RAG index.
-- The vector dimension is templated in by agent.memory.db.migrate(), because a
-- vector column's width cannot be supplied as a bind parameter.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- memories --
CREATE TABLE IF NOT EXISTS memories (
    id           TEXT PRIMARY KEY,
    kind         TEXT NOT NULL DEFAULT 'lesson',
    text         TEXT NOT NULL,
    embedding    vector({dim}) NOT NULL,
    metadata     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    run_id       TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    use_count    INTEGER NOT NULL DEFAULT 0
);

-- HNSW rather than ivfflat: an ivfflat index built on an empty table has no
-- trained centroids, and index scans against it come back empty until it is
-- rebuilt. These tables start empty every time, so ivfflat is the wrong tool.
CREATE INDEX IF NOT EXISTS memories_embedding_idx
    ON memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS memories_kind_idx ON memories (kind);
CREATE INDEX IF NOT EXISTS memories_run_idx  ON memories (run_id);

-- --------------------------------------------------------------- documents --
CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    source     TEXT NOT NULL UNIQUE,
    title      TEXT NOT NULL DEFAULT '',
    kind       TEXT NOT NULL DEFAULT 'file',
    checksum   TEXT NOT NULL,
    metadata   JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);

-- -------------------------------------------------------------------- runs --
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    task         TEXT NOT NULL,
    status       TEXT NOT NULL,
    summary      TEXT NOT NULL DEFAULT '',
    iterations   INTEGER NOT NULL DEFAULT 0,
    usd          NUMERIC(10, 6) NOT NULL DEFAULT 0,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    duration_s   NUMERIC(10, 3) NOT NULL DEFAULT 0,
    payload      JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS runs_created_idx ON runs (created_at DESC);
