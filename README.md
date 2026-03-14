# chunk-embed

Embed [text-chunker](https://github.com/your-user/text-chunker) JSON output into PostgreSQL + pgvector for semantic search.

Takes structural chunks (headings, paragraphs, code blocks, math, etc.) from Markdown or LaTeX documents, encodes them with [BGE-M3](https://huggingface.co/BAAI/bge-m3) (1024-dim multilingual embeddings), and stores both the text and vectors in PostgreSQL. Supports semantic similarity search via the `query` subcommand.

## Installation

Requires Python 3.13+, PostgreSQL with [pgvector](https://github.com/pgvector/pgvector).

```bash
cd ~/Projects/chunk-embed
uv sync
```

### Database setup

```bash
createdb chunk_embed
psql chunk_embed -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

## Usage

### Ingest: pipe from text-chunker

```bash
text-chunker --json chunks document.md | chunk-embed ingest --source document.md
```

### Ingest: from a JSON file

```bash
text-chunker --json chunks document.md > chunks.json
chunk-embed ingest chunks.json
```

Source path is inferred from the file argument. Override with `--source`:

```bash
chunk-embed ingest chunks.json --source original/path/to/document.md
```

### Ingest: dry run

Parse and embed without writing to the database:

```bash
chunk-embed ingest chunks.json --dry-run
```

### Query: semantic search

Search ingested chunks by semantic similarity:

```bash
chunk-embed query "how does authentication work"
```

Output as JSON for programmatic use:

```bash
chunk-embed query "error handling patterns" --json
```

Filter by source file or chunk type:

```bash
chunk-embed query "database schema" --source docs/architecture.md
chunk-embed query "function signatures" --chunk-type code_block --top-k 5
```

Set a minimum similarity threshold:

```bash
chunk-embed query "yoga philosophy" --threshold 0.5
```

### Ingest options

| Option | Default | Description |
|--------|---------|-------------|
| `INPUT` | stdin | JSON file path, or `-` for stdin |
| `--source` | inferred from file | Source path stored as document metadata |
| `--batch-size` | 32 | Embedding batch size |
| `--database-url` | `postgresql://localhost/chunk_embed` | Connection string (env: `DATABASE_URL`) |
| `--dry-run` | off | Skip database write |

### Query options

| Option | Default | Description |
|--------|---------|-------------|
| `QUERY_TEXT` | (required) | Text to search for |
| `--top-k` | 10 | Number of results to return |
| `--source` | all | Filter results by source path |
| `--chunk-type` | all | Filter results by chunk type |
| `--threshold` | 0.0 | Minimum similarity score (-1 to 1) |
| `--json` | off | Output results as JSON |
| `--database-url` | `postgresql://localhost/chunk_embed` | Connection string (env: `DATABASE_URL`) |

## Input format

Expects JSON output from `text-chunker --json chunks`:

```json
{
  "total_chunks": 3,
  "mode": "document",
  "chunks": [
    {
      "text": "# Introduction",
      "chunk_type": "heading",
      "heading_context": ["Introduction"],
      "heading_level": 1,
      "page_number": null,
      "source_line_start": 1,
      "source_line_end": 1
    }
  ]
}
```

Supported chunk types: `heading`, `paragraph`, `list_item`, `code_block`, `table`, `block_quote`, `definition_item`, `math_block`, `theorem`.

## Re-ingestion

Running `chunk-embed ingest` on a document that was previously ingested replaces the old data. The `source_path` column has a unique constraint — the old document and all its chunks are deleted before inserting the new version.

## Database schema

Two tables:

- **documents** — one row per ingested source file (source path, mode, chunk count)
- **chunks** — one row per chunk (text, metadata, 1024-dim embedding vector)

Chunks reference their parent document with `ON DELETE CASCADE`. An HNSW index on the embedding column enables fast cosine similarity search.

## Architecture

```
text-chunker (Rust CLI)
    │
    │  JSON chunks on stdout
    ▼
chunk-embed (Python CLI)
    │
    ├── parse.py    → validate JSON, produce ChunkData objects
    ├── embed.py    → BGE-M3 encoding via sentence-transformers
    ├── store.py    → pgvector storage + cosine similarity search
    ├── format.py   → human-readable and JSON output formatters
    └── cli.py      → Click group: ingest + query subcommands
    │
    ▼
PostgreSQL + pgvector
```

## Development

```bash
# Fast tests (no DB, no model)
uv run pytest -m "not integration and not slow"

# Integration tests (needs PostgreSQL)
createdb chunk_embed_test
psql chunk_embed_test -c "CREATE EXTENSION IF NOT EXISTS vector;"
uv run pytest -m integration

# Model tests (downloads ~2GB on first run)
uv run pytest -m slow

# All tests
uv run pytest
```

## License

MIT
