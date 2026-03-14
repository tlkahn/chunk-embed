from __future__ import annotations

import numpy as np
import psycopg

from chunk_embed.models import ChunkData

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_path   TEXT NOT NULL UNIQUE,
    mode          TEXT NOT NULL CHECK (mode IN ('document', 'per_page')),
    total_chunks  INTEGER NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id                BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id       BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index       INTEGER NOT NULL,
    text              TEXT NOT NULL,
    chunk_type        TEXT NOT NULL,
    heading_context   TEXT[] NOT NULL DEFAULT '{}',
    heading_level     SMALLINT,
    page_number       INTEGER,
    source_line_start INTEGER NOT NULL,
    source_line_end   INTEGER NOT NULL,
    embedding         vector(1024) NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks (document_id);
"""


def ensure_schema(conn: psycopg.Connection) -> None:
    conn.execute(_SCHEMA_SQL)


def upsert_document(
    conn: psycopg.Connection,
    source_path: str,
    mode: str,
    total_chunks: int,
) -> int:
    conn.execute("DELETE FROM documents WHERE source_path = %s", (source_path,))
    row = conn.execute(
        "INSERT INTO documents (source_path, mode, total_chunks) "
        "VALUES (%s, %s, %s) RETURNING id",
        (source_path, mode, total_chunks),
    ).fetchone()
    return row[0]


def insert_chunks(
    conn: psycopg.Connection,
    document_id: int,
    chunks: list[ChunkData],
    embeddings: list[np.ndarray],
) -> None:
    with conn.cursor() as cur:
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                "INSERT INTO chunks "
                "(document_id, chunk_index, text, chunk_type, heading_context, "
                "heading_level, page_number, source_line_start, source_line_end, embedding) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    document_id,
                    i,
                    chunk.text,
                    chunk.chunk_type,
                    chunk.heading_context,
                    chunk.heading_level,
                    chunk.page_number,
                    chunk.source_line_start,
                    chunk.source_line_end,
                    emb.tolist(),
                ),
            )
