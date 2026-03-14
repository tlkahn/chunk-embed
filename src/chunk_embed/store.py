from __future__ import annotations

import numpy as np
import psycopg
from tqdm import tqdm

from chunk_embed.models import ChunkData, SearchResult

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
        for i, (chunk, emb) in tqdm(enumerate(zip(chunks, embeddings)), total=len(chunks), desc="Storing", unit="chunk"):
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


def search_chunks(
    conn: psycopg.Connection,
    query_embedding: np.ndarray,
    top_k: int = 10,
    source_path: str | None = None,
    chunk_type: str | None = None,
    threshold: float = 0.0,
) -> list[SearchResult]:
    query_vec = query_embedding.tolist()

    sql = (
        "SELECT c.text, c.chunk_type, c.heading_context, c.heading_level, "
        "c.page_number, c.source_line_start, c.source_line_end, "
        "d.source_path, 1 - (c.embedding <=> %s::vector) AS similarity "
        "FROM chunks c "
        "JOIN documents d ON c.document_id = d.id "
        "WHERE 1=1"
    )
    params: list = [query_vec]

    if source_path is not None:
        sql += " AND d.source_path = %s"
        params.append(source_path)

    if chunk_type is not None:
        sql += " AND c.chunk_type = %s"
        params.append(chunk_type)

    sql += " ORDER BY c.embedding <=> %s::vector LIMIT %s"
    params.extend([query_vec, top_k])

    rows = conn.execute(sql, params).fetchall()

    results = []
    for row in rows:
        sim = float(row[8])
        if sim < threshold:
            continue
        results.append(SearchResult(
            text=row[0],
            chunk_type=row[1],
            heading_context=list(row[2]) if row[2] else [],
            heading_level=row[3],
            page_number=row[4],
            source_line_start=row[5],
            source_line_end=row[6],
            source_path=row[7],
            similarity=sim,
        ))

    return results
