from __future__ import annotations

import gc
import json
import logging
import shutil
import sys
from pathlib import Path

import click
import psycopg
from pgvector.psycopg import register_vector

from chunk_embed.embed import BgeM3Embedder, embed_chunks
from chunk_embed.format import format_results_human, format_results_json
from chunk_embed.parse import parse_chunks, ParseError
from chunk_embed.pipeline import (
    MARKDOWN_SUFFIXES,
    ingest_one_file,
    resolve_paths,
)
from chunk_embed.split import split_chunks
from chunk_embed.store import (
    delete_documents,
    ensure_schema,
    get_chunk_summary,
    get_document,
    insert_chunks,
    list_documents,
    search_chunks,
    upsert_document,
)


@click.group("chunk-embed")
def main() -> None:
    """Embed text-chunker JSON output into pgvector for semantic search."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from chunk_embed._paths import prepend_bundled_bin_to_path
    prepend_bundled_bin_to_path()


@main.command()
@click.argument("input_paths", nargs=-1, type=click.Path())
@click.option("--source", default=None, help="Source file path for metadata (required for stdin)")
@click.option("--batch-size", default=32, type=int, help="Embedding batch size")
@click.option(
    "--database-url",
    default="postgresql://localhost/chunk_embed",
    envvar="DATABASE_URL",
    help="PostgreSQL connection string",
)
@click.option("--dry-run", is_flag=True, help="Parse and embed without writing to DB")
@click.option("--no-split", is_flag=True, help="Disable sentence splitting")
@click.option("--glob", "file_glob", default=None, help="Glob pattern for directory filtering")
@click.option("--fail-fast", is_flag=True, help="Stop on first file error")
@click.option("--no-recursive", is_flag=True, help="Don't recurse into subdirectories")
def ingest(
    input_paths: tuple[str, ...],
    source: str | None,
    batch_size: int,
    database_url: str,
    dry_run: bool,
    no_split: bool,
    file_glob: str | None,
    fail_fast: bool,
    no_recursive: bool,
) -> None:
    """Ingest text-chunker JSON (or markdown) into pgvector.

    Accepts one or more files and/or directories. With no arguments, reads
    JSON from stdin (requires --source).
    """
    # --- stdin path (backward compat) ---
    if not input_paths or input_paths == ("-",):
        if source is None:
            click.echo("Error: --source is required when reading from stdin", err=True)
            sys.exit(1)
        raw = click.get_text_stream("stdin").read()

        try:
            chunks_input = parse_chunks(raw)
        except ParseError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        click.echo(f"Parsed {chunks_input.total_chunks} chunks ({chunks_input.mode} mode)")

        chunks = chunks_input.chunks
        if not no_split:
            chunks = split_chunks(chunks)
            click.echo(f"Split into {len(chunks)} sentence chunks")

        embedder = BgeM3Embedder()
        embeddings = embed_chunks(chunks, embedder, batch_size=batch_size)
        if dry_run:
            click.echo("Dry run: skipping database write")
            return

        with psycopg.connect(database_url) as conn:
            register_vector(conn)
            ensure_schema(conn)
            doc_id = upsert_document(conn, source, chunks_input.mode, chunks_input.total_chunks)
            insert_chunks(conn, doc_id, chunks, embeddings)
            conn.commit()

        click.echo(f"Done (document {doc_id})")
        return

    # --- file/directory path ---
    try:
        resolved = resolve_paths(input_paths, glob_pattern=file_glob, recursive=not no_recursive)
    except FileNotFoundError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if not resolved:
        click.echo("No eligible files found.", err=True)
        sys.exit(1)

    # Pre-flight: check text-chunker availability if any markdown files
    has_markdown = any(f.suffix.lower() in MARKDOWN_SUFFIXES for f in resolved)
    if has_markdown and not shutil.which("text-chunker"):
        click.echo("Error: text-chunker not found in PATH (needed for markdown files)", err=True)
        sys.exit(1)

    if source and len(resolved) > 1:
        click.echo("Warning: --source ignored with multiple files (each file uses its own path)", err=True)
        source = None

    click.echo(f"Found {len(resolved)} file(s) to ingest")

    embedder = BgeM3Embedder()

    successes = 0
    failures: list[tuple[Path, str]] = []
    for i, file_path in enumerate(resolved, 1):
        click.echo(f"\n[{i}/{len(resolved)}] {file_path}")
        try:
            result = ingest_one_file(
                file_path=file_path,
                source=source,
                embedder=embedder,
                split=not no_split,
                batch_size=batch_size,
                database_url=database_url,
                dry_run=dry_run,
            )
            click.echo(f"  OK: {result.num_chunks} chunks" +
                       (f", document {result.doc_id}" if result.doc_id else ""))
            successes += 1
        except Exception as e:
            click.echo(f"  FAILED: {e}", err=True)
            failures.append((file_path, str(e)))
            if fail_fast:
                sys.exit(1)
        finally:
            gc.collect()

    click.echo(f"\nDone: {successes} succeeded, {len(failures)} failed")
    if failures:
        sys.exit(1)


@main.command()
@click.argument("query_text")
@click.option("--top-k", default=10, type=int, help="Number of results to return")
@click.option(
    "--database-url",
    default="postgresql://localhost/chunk_embed",
    envvar="DATABASE_URL",
    help="PostgreSQL connection string",
)
@click.option("--source", default=None, help="Filter results by source path")
@click.option("--chunk-type", default=None, help="Filter results by chunk type")
@click.option("--threshold", default=0.0, type=float, help="Minimum similarity threshold")
@click.option("--json", "as_json", is_flag=True, help="Output results as JSON")
def query(query_text: str, top_k: int, database_url: str, source: str | None,
          chunk_type: str | None, threshold: float, as_json: bool) -> None:
    """Search chunks by semantic similarity to QUERY_TEXT."""
    embedder = BgeM3Embedder()
    query_embedding = embedder.embed([query_text])[0]

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        results = search_chunks(conn, query_embedding, top_k=top_k,
                                source_paths=[source] if source else None,
                                chunk_types=[chunk_type] if chunk_type else None,
                                threshold=threshold)

    if as_json:
        click.echo(format_results_json(results))
    else:
        click.echo(format_results_human(results))


@main.command()
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--database-url",
    default="postgresql://localhost/chunk_embed",
    envvar="DATABASE_URL",
    help="PostgreSQL connection string",
)
def docs(as_json: bool, database_url: str) -> None:
    """List all ingested documents."""
    with psycopg.connect(database_url) as conn:
        documents = list_documents(conn)

    if not documents:
        click.echo("No documents found.")
        return

    if as_json:
        click.echo(json.dumps([
            {
                "id": d.id,
                "source_path": d.source_path,
                "mode": d.mode,
                "total_chunks": d.total_chunks,
                "created_at": d.created_at.isoformat(),
            }
            for d in documents
        ], indent=2))
        return

    click.echo(f"{'ID':>6}  {'SOURCE':<50}  {'MODE':<10}  {'CHUNKS':>6}  CREATED")
    for d in documents:
        click.echo(
            f"{d.id:>6}  {d.source_path:<50}  {d.mode:<10}  {d.total_chunks:>6}  "
            f"{d.created_at:%Y-%m-%d %H:%M}"
        )


@main.command()
@click.argument("source_path")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
@click.option(
    "--database-url",
    default="postgresql://localhost/chunk_embed",
    envvar="DATABASE_URL",
    help="PostgreSQL connection string",
)
def show(source_path: str, as_json: bool, database_url: str) -> None:
    """Show document details and chunk breakdown for SOURCE_PATH."""
    with psycopg.connect(database_url) as conn:
        doc = get_document(conn, source_path)
        if doc is None:
            click.echo(f"Document not found: {source_path}", err=True)
            sys.exit(1)
        summaries = get_chunk_summary(conn, doc.id)

    if as_json:
        click.echo(json.dumps({
            "id": doc.id,
            "source_path": doc.source_path,
            "mode": doc.mode,
            "total_chunks": doc.total_chunks,
            "created_at": doc.created_at.isoformat(),
            "chunk_types": [
                {"chunk_type": s.chunk_type, "count": s.count, "total_chars": s.total_chars}
                for s in summaries
            ],
        }, indent=2))
        return

    click.echo(f"Document: {doc.source_path}")
    click.echo(f"ID: {doc.id}  Mode: {doc.mode}  Chunks: {doc.total_chunks}  Created: {doc.created_at:%Y-%m-%d %H:%M}")
    if summaries:
        click.echo(f"\n{'TYPE':<20}  {'COUNT':>6}  {'TOTAL CHARS':>12}")
        for s in summaries:
            click.echo(f"{s.chunk_type:<20}  {s.count:>6}  {s.total_chars:>12}")


@main.command()
@click.argument("source_paths", nargs=-1, required=True)
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt")
@click.option(
    "--database-url",
    default="postgresql://localhost/chunk_embed",
    envvar="DATABASE_URL",
    help="PostgreSQL connection string",
)
def remove(source_paths: tuple[str, ...], yes: bool, database_url: str) -> None:
    """Remove one or more documents (and their chunks) by SOURCE_PATH."""
    paths = list(source_paths)
    if not yes:
        click.echo(f"Will delete {len(paths)} document(s):")
        for p in paths:
            click.echo(f"  {p}")
        if not click.confirm("Proceed?"):
            click.echo("Aborted.")
            return

    with psycopg.connect(database_url) as conn:
        count = delete_documents(conn, paths)
        conn.commit()

    if count == 0:
        click.echo("No matching documents found.", err=True)
        sys.exit(1)

    click.echo(f"Deleted {count} document(s).")
