# chunk-embed

Embed [text-chunker](https://github.com/your-user/text-chunker) JSON output into PostgreSQL + pgvector for semantic search.

Takes structural chunks (headings, paragraphs, code blocks, math, etc.) from Markdown or LaTeX documents, encodes them with [BGE-M3](https://huggingface.co/BAAI/bge-m3) (1024-dim multilingual embeddings), and stores both the text and vectors in PostgreSQL. Supports semantic similarity search via the `query` subcommand.

## Installation

Requires Python 3.13+, PostgreSQL with [pgvector](https://github.com/pgvector/pgvector). Optional: [Sentenza](https://github.com/your-user/sentenza) for sentence-level splitting.

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

Note: `--source` is **required** when reading from stdin, since there is no filename to infer from:

```bash
# stdin: no filename, so --source is required
text-chunker --json chunks file.md | chunk-embed ingest --source file.md

# without --source, this fails with an error
text-chunker --json chunks file.md | chunk-embed ingest
# → "Error: --source is required when reading from stdin"
```

When passing a file argument, `--source` defaults to that path automatically:

```bash
# --source defaults to "chunks.json"
chunk-embed ingest chunks.json
```

### Ingest: sentence splitting

By default, prose chunks (paragraphs, headings, block quotes, etc.) are split into individual sentences via [Sentenza](https://github.com/your-user/sentenza) before embedding. Each sentence gets its own vector, improving retrieval precision.

```bash
# Default: split English text into sentences
text-chunker --json chunks document.md | chunk-embed ingest --source document.md

# Specify language for sentence splitting
chunk-embed ingest chunks.json --lang de

# Disable sentence splitting (embed whole chunks)
chunk-embed ingest chunks.json --no-split
```

Splittable chunk types: `paragraph`, `block_quote`, `definition_item`, `theorem`, `list_item`, `heading`. Non-splittable types (`code_block`, `math_block`, `table`) pass through unchanged.

If the `sentenza` binary is not found, splitting is skipped gracefully and chunks are embedded whole.

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
| `--lang` | `en` | ISO 639-1 language code for sentence splitting |
| `--no-split` | off | Disable sentence splitting |
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
    ├── split.py    → sentence splitting via Sentenza CLI
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

# Sentenza tests (needs sentenza binary in PATH)
uv run pytest -m sentenza

# Model tests (downloads ~2GB on first run)
uv run pytest -m slow

# All tests
uv run pytest
```

## macOS App (.dmg)

A pre-built `.dmg` is available from the [Releases](https://github.com/tlkahn/chunk-embed/releases) page. The app is ad-hoc signed (not notarized by Apple), so macOS Gatekeeper will block it by default.

### Opening the app for the first time

After mounting the `.dmg` and dragging **Chunk Embed** into Applications:

1. **Double-click** the app. You will see a dialog: *"Chunk Embed" can't be opened because Apple cannot check it for malicious software.*
2. Open **System Settings > Privacy & Security**, scroll down — you will see a message about "Chunk Embed" being blocked. Click **Open Anyway**.
3. Alternatively, **right-click** (or Control-click) the app and choose **Open**, then click **Open** in the dialog.

You only need to do this once. After the first launch, macOS remembers your choice.

### Prerequisites

The app bundles Python and all dependencies. You still need:

- **PostgreSQL** with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- An internet connection on first launch (to download the BGE-M3 model, ~2 GB)

The Setup tab inside the app checks dependency status and provides install guidance.

## License

MIT
