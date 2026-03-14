# CLAUDE.md

## Build & Test Commands

```bash
uv run pytest -m "not slow and not integration"   # fast tests only (no DB, no model)
uv run pytest -m "not slow"                        # includes integration tests (needs PostgreSQL + pgvector)
uv run pytest -m slow                              # model tests (~2GB download on first run)
uv run pytest                                      # all tests
```

### Database setup for integration tests

```bash
createdb chunk_embed_test
psql chunk_embed_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Architecture

Separate Python project that takes JSON output from text-chunker, embeds each chunk using BGE-M3, and stores in PostgreSQL + pgvector.

- **src/chunk_embed/models.py** — `ChunkData` and `ChunksInput` frozen dataclasses
- **src/chunk_embed/parse.py** — `parse_chunks()` JSON parsing with `ParseError`
- **src/chunk_embed/embed.py** — `Embedder` protocol, `BgeM3Embedder`, `embed_chunks()` batched embedding
- **src/chunk_embed/store.py** — `ensure_schema()`, `upsert_document()`, `insert_chunks()` for pgvector storage
- **src/chunk_embed/cli.py** — Click CLI entry point

## CLI Usage

```bash
# Pipe from text-chunker
text-chunker --json chunks file.md | chunk-embed --source file.md

# From file
chunk-embed chunks.json

# Dry run (parse + embed, skip DB)
chunk-embed chunks.json --dry-run
```

## Re-ingest Strategy

`UNIQUE(source_path)` on documents table. Re-ingesting the same source deletes the old document (cascading to chunks) and inserts fresh.
