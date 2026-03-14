"""
Smoke test for chunk-embed: end-to-end with real BGE-M3 model and PostgreSQL.

Exercises the full pipeline: parse → embed → store → verify via SQL.

Requires:
    - Local BGE-M3 model at ./local_bge_m3/BAAI/bge-m3
    - PostgreSQL with pgvector: createdb chunk_embed_test && psql chunk_embed_test -c "CREATE EXTENSION vector;"

Usage:
    cd ~/Projects/chunk-embed
    uv run python smoke_test.py
"""

import json
import sys

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from chunk_embed.parse import parse_chunks
from chunk_embed.embed import BgeM3Embedder, embed_chunks
from chunk_embed.store import ensure_schema, upsert_document, insert_chunks, search_chunks

DB_URL = "postgresql://localhost/chunk_embed_test"

# --- Real-world test data ---

SANSKRIT_DOC = {
    "total_chunks": 4,
    "mode": "document",
    "chunks": [
        {
            "text": "# नासदीयसूक्तम् (Nāsadīya Sūkta)",
            "chunk_type": "heading",
            "heading_context": ["नासदीयसूक्तम् (Nāsadīya Sūkta)"],
            "heading_level": 1,
            "page_number": None,
            "source_line_start": 1,
            "source_line_end": 1,
        },
        {
            "text": "नासदासीन्नो सदासीत्तदानीं नासीद्रजो नो व्योमा परो यत् ।\nकिमावरीवः कुह कस्य शर्मन्नम्भः किमासीद्गहनं गभीरम् ॥",
            "chunk_type": "paragraph",
            "heading_context": ["नासदीयसूक्तम् (Nāsadīya Sūkta)"],
            "heading_level": None,
            "page_number": None,
            "source_line_start": 3,
            "source_line_end": 4,
        },
        {
            "text": "There was neither non-existence nor existence then;\nthere was neither the realm of space nor the sky which is beyond.",
            "chunk_type": "block_quote",
            "heading_context": ["नासदीयसूक्तम् (Nāsadīya Sūkta)"],
            "heading_level": None,
            "page_number": None,
            "source_line_start": 6,
            "source_line_end": 7,
        },
        {
            "text": "$\\text{RV 10.129}$: This hymn is one of the oldest cosmogonic speculations in world literature.",
            "chunk_type": "math_block",
            "heading_context": ["नासदीयसूक्तम् (Nāsadīya Sūkta)"],
            "heading_level": None,
            "page_number": None,
            "source_line_start": 9,
            "source_line_end": 9,
        },
    ],
}

CODE_DOC = {
    "total_chunks": 3,
    "mode": "document",
    "chunks": [
        {
            "text": "## Quickstart",
            "chunk_type": "heading",
            "heading_context": ["Quickstart"],
            "heading_level": 2,
            "page_number": None,
            "source_line_start": 1,
            "source_line_end": 1,
        },
        {
            "text": "```python\nfrom sentence_transformers import SentenceTransformer\nmodel = SentenceTransformer('BAAI/bge-m3')\nembeddings = model.encode(['Hello world'])\n```",
            "chunk_type": "code_block",
            "heading_context": ["Quickstart"],
            "heading_level": None,
            "page_number": None,
            "source_line_start": 3,
            "source_line_end": 7,
        },
        {
            "text": "| Model | Dimensions | Languages |\n|-------|-----------|----------|\n| BGE-M3 | 1024 | 100+ |",
            "chunk_type": "table",
            "heading_context": ["Quickstart"],
            "heading_level": None,
            "page_number": None,
            "source_line_start": 9,
            "source_line_end": 11,
        },
    ],
}

PERPAGE_DOC = {
    "total_chunks": 2,
    "mode": "per_page",
    "chunks": [
        {
            "text": "The Yogasūtra of Patañjali begins with the definition: yogaś citta-vṛtti-nirodhaḥ.",
            "chunk_type": "paragraph",
            "heading_context": ["Chapter 1"],
            "heading_level": None,
            "page_number": 1,
            "source_line_start": 1,
            "source_line_end": 1,
        },
        {
            "text": "- **citta**: mind-stuff\n- **vṛtti**: fluctuations\n- **nirodha**: cessation",
            "chunk_type": "definition_item",
            "heading_context": ["Chapter 1", "Key Terms"],
            "heading_level": None,
            "page_number": 2,
            "source_line_start": 3,
            "source_line_end": 5,
        },
    ],
}


def run_smoke_test():
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: {detail}")
            failed += 1

    # --- 1. Load model ---
    print("Loading BGE-M3 model...")
    embedder = BgeM3Embedder()
    check("Model loaded", embedder.dimension == 1024)

    # --- 2. Parse + embed each document ---
    test_cases = [
        ("sanskrit", SANSKRIT_DOC, "/test/nasadiya.md"),
        ("code", CODE_DOC, "/test/quickstart.md"),
        ("perpage", PERPAGE_DOC, "/test/yogasutra.md"),
    ]

    all_results = {}
    for label, doc, source in test_cases:
        print(f"\n--- {label} document ---")
        raw = json.dumps(doc)
        parsed = parse_chunks(raw)
        check(f"Parse {label}", parsed.total_chunks == doc["total_chunks"])

        embeddings = embed_chunks(parsed.chunks, embedder, batch_size=32)
        check(f"Embed {label} count", len(embeddings) == len(parsed.chunks))
        check(f"Embed {label} shape", all(v.shape == (1024,) for v in embeddings))
        check(f"Embed {label} finite", all(np.all(np.isfinite(v)) for v in embeddings))
        norms = [float(np.linalg.norm(v)) for v in embeddings]
        check(f"Embed {label} normalized", all(abs(n - 1.0) < 1e-4 for n in norms),
              f"norms={norms}")

        all_results[label] = (parsed, embeddings, source)

    # --- 3. Semantic similarity sanity check ---
    print("\n--- Semantic similarity ---")
    sanskrit_embs = all_results["sanskrit"][1]
    # The Sanskrit verse and its English translation should be more similar to each other
    # than the Sanskrit verse is to the code snippet
    sanskrit_verse = sanskrit_embs[1]  # Devanagari verse
    english_trans = sanskrit_embs[2]   # English translation
    code_emb = all_results["code"][1][1]  # Python code block

    sim_verse_trans = float(np.dot(sanskrit_verse, english_trans))
    sim_verse_code = float(np.dot(sanskrit_verse, code_emb))
    check("Sanskrit-English similarity > Sanskrit-Code similarity",
          sim_verse_trans > sim_verse_code,
          f"verse↔translation={sim_verse_trans:.3f}, verse↔code={sim_verse_code:.3f}")
    print(f"    verse ↔ translation: {sim_verse_trans:.4f}")
    print(f"    verse ↔ code block:  {sim_verse_code:.4f}")

    # --- 4. Store in database ---
    print("\n--- Database storage ---")
    with psycopg.connect(DB_URL, autocommit=False) as conn:
        register_vector(conn)
        ensure_schema(conn)

        for label, (parsed, embeddings, source) in all_results.items():
            doc_id = upsert_document(conn, source, parsed.mode, parsed.total_chunks)
            insert_chunks(conn, doc_id, parsed.chunks, embeddings)

        # Verify row counts
        doc_count = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT count(*) FROM chunks").fetchone()[0]
        check("Document count", doc_count == 3, f"got {doc_count}")
        total_expected = sum(d["total_chunks"] for _, d, _ in test_cases)
        check("Chunk count", chunk_count == total_expected, f"got {chunk_count}, expected {total_expected}")

        # Verify Unicode round-trip
        row = conn.execute(
            "SELECT text FROM chunks WHERE text LIKE '%नासदासीन्नो%'"
        ).fetchone()
        check("Unicode round-trip", row is not None and "नासदासीन्नो" in row[0])

        # Verify vector round-trip
        row = conn.execute(
            "SELECT embedding FROM chunks WHERE document_id = "
            "(SELECT id FROM documents WHERE source_path = %s) ORDER BY chunk_index LIMIT 1",
            ("/test/nasadiya.md",),
        ).fetchone()
        stored = np.array(row[0], dtype=np.float32)
        original = all_results["sanskrit"][1][0]
        check("Vector round-trip", np.allclose(stored, original, atol=1e-5))

        # Verify per-page mode stored correctly
        page_rows = conn.execute(
            "SELECT page_number FROM chunks WHERE document_id = "
            "(SELECT id FROM documents WHERE source_path = %s) ORDER BY chunk_index",
            ("/test/yogasutra.md",),
        ).fetchall()
        check("Per-page page_numbers", [r[0] for r in page_rows] == [1, 2])

        # --- 5. Semantic search via search_chunks ---
        print("\n--- Semantic search ---")

        # Query with a cosmogonic theme — Sanskrit chunks should rank highest
        query_text = "Sanskrit cosmogonic verse about creation"
        query_emb = embedder.embed([query_text])[0]

        results = search_chunks(conn, query_emb, top_k=5)
        check("Search returns results", len(results) > 0)
        # Sanskrit document chunks should rank above code doc chunks
        top_source = results[0].source_path
        check("Top result is Sanskrit doc", top_source == "/test/nasadiya.md",
              f"got {top_source}")

        # Source filter: restrict to code doc only
        code_results = search_chunks(conn, query_emb, source_path="/test/quickstart.md")
        check("Source filter returns only code doc",
              all(r.source_path == "/test/quickstart.md" for r in code_results))
        check("Source filter has results", len(code_results) > 0)

        # Chunk type filter: headings only
        heading_results = search_chunks(conn, query_emb, chunk_type="heading")
        check("Chunk type filter returns only headings",
              all(r.chunk_type == "heading" for r in heading_results))
        check("Heading filter has results", len(heading_results) > 0)

        # --- 6. Re-ingest test ---
        print("\n--- Re-ingest ---")
        old_id = conn.execute(
            "SELECT id FROM documents WHERE source_path = %s", ("/test/nasadiya.md",)
        ).fetchone()[0]
        new_id = upsert_document(conn, "/test/nasadiya.md", "document", 1)
        single_chunk = all_results["sanskrit"][0].chunks[:1]
        single_emb = all_results["sanskrit"][1][:1]
        insert_chunks(conn, new_id, single_chunk, single_emb)

        check("Re-ingest new ID", new_id != old_id)
        old_chunks = conn.execute(
            "SELECT id FROM chunks WHERE document_id = %s", (old_id,)
        ).fetchall()
        check("Old chunks deleted", len(old_chunks) == 0)
        new_chunks = conn.execute(
            "SELECT id FROM chunks WHERE document_id = %s", (new_id,)
        ).fetchall()
        check("New chunks inserted", len(new_chunks) == 1)

        # Rollback everything — don't leave test data
        conn.rollback()

    # --- Summary ---
    total = passed + failed
    print(f"\n{'=' * 40}")
    print(f"Smoke test: {passed}/{total} checks passed")
    if failed:
        print(f"FAILED: {failed} check(s)")
        sys.exit(1)
    else:
        print("All checks passed!")


if __name__ == "__main__":
    run_smoke_test()
