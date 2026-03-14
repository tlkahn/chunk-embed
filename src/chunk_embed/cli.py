from __future__ import annotations

import sys
from pathlib import Path

import click
import psycopg
from pgvector.psycopg import register_vector

from chunk_embed.embed import BgeM3Embedder, embed_chunks
from chunk_embed.format import format_results_human, format_results_json
from chunk_embed.parse import parse_chunks, ParseError
from chunk_embed.split import split_chunks
from chunk_embed.store import ensure_schema, upsert_document, insert_chunks, search_chunks


@click.group("chunk-embed")
def main() -> None:
    """Embed text-chunker JSON output into pgvector for semantic search."""


@main.command()
@click.argument("input_path", default="-", type=click.Path())
@click.option("--source", default=None, help="Source file path for metadata (required for stdin)")
@click.option("--batch-size", default=32, type=int, help="Embedding batch size")
@click.option(
    "--database-url",
    default="postgresql://localhost/chunk_embed",
    envvar="DATABASE_URL",
    help="PostgreSQL connection string",
)
@click.option("--dry-run", is_flag=True, help="Parse and embed without writing to DB")
@click.option("--lang", default="en", help="ISO 639-1 language code for sentence splitting")
@click.option("--no-split", is_flag=True, help="Disable sentence splitting")
def ingest(input_path: str, source: str | None, batch_size: int, database_url: str, dry_run: bool, lang: str, no_split: bool) -> None:
    """Ingest text-chunker JSON into pgvector."""
    if input_path == "-":
        if source is None:
            click.echo("Error: --source is required when reading from stdin", err=True)
            sys.exit(1)
        raw = click.get_text_stream("stdin").read()
    else:
        path = Path(input_path)
        raw = path.read_text()
        if source is None:
            source = str(path)

    try:
        chunks_input = parse_chunks(raw)
    except ParseError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"Parsed {chunks_input.total_chunks} chunks ({chunks_input.mode} mode)")

    chunks = chunks_input.chunks
    if not no_split:
        chunks = split_chunks(chunks, lang)
        click.echo(f"Split into {len(chunks)} sentence chunks")

    embedder = BgeM3Embedder()
    embeddings = embed_chunks(chunks, embedder, batch_size=batch_size)
    click.echo(f"Embedded {len(embeddings)} chunks")

    if dry_run:
        click.echo("Dry run: skipping database write")
        return

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        ensure_schema(conn)
        doc_id = upsert_document(conn, source, chunks_input.mode, chunks_input.total_chunks)
        insert_chunks(conn, doc_id, chunks, embeddings)
        conn.commit()

    click.echo(f"Stored document {doc_id} with {len(embeddings)} chunks")


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
                                source_path=source, chunk_type=chunk_type,
                                threshold=threshold)

    if as_json:
        click.echo(format_results_json(results))
    else:
        click.echo(format_results_human(results))
