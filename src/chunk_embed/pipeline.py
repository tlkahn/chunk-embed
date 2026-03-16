from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import psycopg
from pgvector.psycopg import register_vector

from chunk_embed.embed import Embedder, embed_chunks
from chunk_embed.parse import parse_chunks
from chunk_embed.split import split_chunks
from chunk_embed.store import ensure_schema, upsert_document, insert_chunks

MARKDOWN_SUFFIXES = frozenset({".md", ".markdown", ".mdown", ".mkd"})
ELIGIBLE_SUFFIXES = MARKDOWN_SUFFIXES | frozenset({".json"})


@dataclass(frozen=True)
class IngestResult:
    source_path: str
    doc_id: int | None
    num_chunks: int
    num_embeddings: int
    dry_run: bool


def resolve_paths(
    input_paths: Sequence[str | Path],
    glob_pattern: str | None = None,
    recursive: bool = True,
) -> list[Path]:
    """Expand files and directories into a sorted list of eligible file paths."""
    result: list[Path] = []
    for p in input_paths:
        path = Path(p)
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            if glob_pattern:
                pattern = f"**/{glob_pattern}" if recursive else glob_pattern
                result.extend(sorted(f for f in path.glob(pattern) if f.is_file()))
            else:
                iterator = path.rglob("*") if recursive else path.iterdir()
                result.extend(
                    sorted(f for f in iterator if f.is_file() and f.suffix.lower() in ELIGIBLE_SUFFIXES)
                )
        else:
            raise FileNotFoundError(f"Path not found: {path}")
    return result


def read_or_chunk_file(file_path: Path) -> str:
    """Read a file, running text-chunker first if it is markdown.

    Returns the raw JSON string ready for parse_chunks().
    """
    if file_path.suffix.lower() in MARKDOWN_SUFFIXES:
        try:
            result = subprocess.run(
                ["text-chunker", "--json", "chunks", str(file_path)],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"text-chunker failed (exit {e.returncode}): {e.stderr.strip()}"
            ) from e
        return result.stdout
    return file_path.read_text()


def ingest_one_file(
    file_path: Path,
    source: str | None,
    embedder: Embedder,
    split: bool = True,
    batch_size: int = 32,
    database_url: str | None = None,
    dry_run: bool = False,
    on_log: Callable[[str], None] | None = None,
) -> IngestResult:
    """Ingest a single file through the full pipeline.

    The caller provides an already-loaded Embedder so it can be reused
    across multiple files in a batch.
    """
    def log(msg: str) -> None:
        if on_log:
            on_log(msg)

    effective_source = source if source is not None else str(file_path)

    raw = read_or_chunk_file(file_path)
    chunks_input = parse_chunks(raw)
    log(f"Parsed {chunks_input.total_chunks} chunks ({chunks_input.mode} mode)")

    chunks = chunks_input.chunks
    if split:
        chunks = split_chunks(chunks)
        log(f"Split into {len(chunks)} sentence chunks")

    embeddings = embed_chunks(chunks, embedder, batch_size=batch_size)

    doc_id: int | None = None
    if not dry_run:
        if database_url is None:
            raise ValueError("database_url is required when dry_run is False")
        with psycopg.connect(database_url) as conn:
            register_vector(conn)
            ensure_schema(conn)
            doc_id = upsert_document(conn, effective_source, chunks_input.mode, chunks_input.total_chunks)
            insert_chunks(conn, doc_id, chunks, embeddings)
            conn.commit()

    return IngestResult(
        source_path=effective_source,
        doc_id=doc_id,
        num_chunks=len(chunks),
        num_embeddings=len(embeddings),
        dry_run=dry_run,
    )
