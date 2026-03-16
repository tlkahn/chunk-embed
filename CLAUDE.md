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
- **src/chunk_embed/embed.py** — `Embedder` protocol, `BgeM3Embedder`, `embed_chunks()` batched embedding with length-sorted batching
- **src/chunk_embed/store.py** — `ensure_schema()`, `upsert_document()`, `insert_chunks()` for pgvector storage
- **src/chunk_embed/cli.py** — Click CLI entry point
- **src/chunk_embed/pipeline.py** — `ingest_one_file()`, `resolve_paths()` shared by CLI and GUI
- **src/chunk_embed/split.py** — sentence splitting via lingua language detection + sentenza CLI
- **src/chunk_embed/gui.py** — PySide6 GUI with Ingest/Search tabs

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

## Embedding Batch Optimization

`embed_chunks()` sorts chunk indices by text length before batching. This prevents a single long text (e.g. a 5000-char code block) from padding an entire batch of 32 short texts to its length, which freezes the process on CPU. Results are written directly to their original slot via `results[slot] = vec.copy()`, so no unsort step is needed (unlike `SentenceTransformer.encode()` which appends in sorted order then unscrambles at the end). Each batch is passed to `encode()` separately to support `on_progress` callbacks; since batches are pre-sorted, `encode()`'s internal length sort becomes a near-identity no-op.

## Memory Management (Batch Ingest)

Two measures prevent memory growth when ingesting many files in a loop:
1. `vec.copy()` in `embed_chunks()` — breaks numpy view references so each batch's source array can be freed immediately
2. `gc.collect()` in `cli.py` after each file (in a `finally` block) — forces reclamation of chunks, embeddings, and raw text between files

## Documentation

Detailed implementation notes (in Chinese) are in `resources/doc/Implementation-notes.md` (symlink to Obsidian vault).
