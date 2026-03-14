from __future__ import annotations

import sys
from pathlib import Path

import click
import psycopg
from pgvector.psycopg import register_vector

from chunk_embed.embed import BgeM3Embedder, embed_chunks
from chunk_embed.parse import parse_chunks, ParseError
from chunk_embed.store import ensure_schema, upsert_document, insert_chunks


@click.command("chunk-embed")
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
def main(input_path: str, source: str | None, batch_size: int, database_url: str, dry_run: bool) -> None:
    """Embed text-chunker JSON output into pgvector for semantic search."""
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

    embedder = BgeM3Embedder()
    embeddings = embed_chunks(chunks_input.chunks, embedder, batch_size=batch_size)
    click.echo(f"Embedded {len(embeddings)} chunks")

    if dry_run:
        click.echo("Dry run: skipping database write")
        return

    with psycopg.connect(database_url) as conn:
        register_vector(conn)
        ensure_schema(conn)
        doc_id = upsert_document(conn, source, chunks_input.mode, chunks_input.total_chunks)
        insert_chunks(conn, doc_id, chunks_input.chunks, embeddings)
        conn.commit()

    click.echo(f"Stored document {doc_id} with {len(embeddings)} chunks")
