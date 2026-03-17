import numpy as np
import pytest

from chunk_embed.models import ChunkSummary, DocumentInfo, SearchResult
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
from tests.conftest import make_chunk, make_embedding

pytestmark = pytest.mark.integration


def test_ensure_schema_creates_tables(conn):
    cur = conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name IN ('documents', 'chunks') "
        "ORDER BY table_name"
    )
    tables = [row[0] for row in cur.fetchall()]
    assert tables == ["chunks", "documents"]


def test_insert_document(conn):
    doc_id = upsert_document(conn, "/test/file.md", "document", 3)
    assert isinstance(doc_id, int)
    row = conn.execute("SELECT source_path, mode, total_chunks FROM documents WHERE id = %s", (doc_id,)).fetchone()
    assert row == ("/test/file.md", "document", 3)


def test_insert_chunks(conn):
    doc_id = upsert_document(conn, "/test/file.md", "document", 3)
    chunks = [make_chunk(text=f"chunk {i}", source_line_start=i, source_line_end=i) for i in range(3)]
    embeddings = [make_embedding() for _ in range(3)]
    insert_chunks(conn, doc_id, chunks, embeddings)

    rows = conn.execute(
        "SELECT chunk_index, text, chunk_type FROM chunks WHERE document_id = %s ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    assert len(rows) == 3
    assert rows[0] == (0, "chunk 0", "paragraph")
    assert rows[2] == (2, "chunk 2", "paragraph")


def test_insert_chunks_with_page_numbers(conn):
    doc_id = upsert_document(conn, "/test/perpage.md", "per_page", 2)
    chunks = [
        make_chunk(text="page 1 content", page_number=1),
        make_chunk(text="page 2 content", page_number=2),
    ]
    embeddings = [make_embedding() for _ in range(2)]
    insert_chunks(conn, doc_id, chunks, embeddings)

    rows = conn.execute(
        "SELECT page_number FROM chunks WHERE document_id = %s ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    assert [r[0] for r in rows] == [1, 2]


def test_reingest_replaces_document(conn):
    doc_id_1 = upsert_document(conn, "/test/file.md", "document", 2)
    chunks = [make_chunk(text=f"old {i}") for i in range(2)]
    embeddings = [make_embedding() for _ in range(2)]
    insert_chunks(conn, doc_id_1, chunks, embeddings)

    doc_id_2 = upsert_document(conn, "/test/file.md", "document", 1)
    new_chunks = [make_chunk(text="new 0")]
    new_embeddings = [make_embedding()]
    insert_chunks(conn, doc_id_2, new_chunks, new_embeddings)

    assert doc_id_2 != doc_id_1
    old_rows = conn.execute("SELECT id FROM chunks WHERE document_id = %s", (doc_id_1,)).fetchall()
    assert old_rows == []
    new_rows = conn.execute("SELECT text FROM chunks WHERE document_id = %s", (doc_id_2,)).fetchall()
    assert len(new_rows) == 1
    assert new_rows[0][0] == "new 0"


def test_delete_document_cascades(conn):
    doc_id = upsert_document(conn, "/test/cascade.md", "document", 1)
    insert_chunks(conn, doc_id, [make_chunk()], [make_embedding()])
    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    rows = conn.execute("SELECT id FROM chunks WHERE document_id = %s", (doc_id,)).fetchall()
    assert rows == []


def test_vector_stored_correctly(conn):
    doc_id = upsert_document(conn, "/test/vec.md", "document", 1)
    original = make_embedding()
    insert_chunks(conn, doc_id, [make_chunk()], [original])

    row = conn.execute("SELECT embedding FROM chunks WHERE document_id = %s", (doc_id,)).fetchone()
    stored = np.array(row[0], dtype=np.float32)
    np.testing.assert_allclose(stored, original, atol=1e-5)


# --- search_chunks tests ---


def _seed_search_data(conn):
    """Insert two documents with distinct embeddings for search tests."""
    doc1 = upsert_document(conn, "/test/alpha.md", "document", 2)
    insert_chunks(conn, doc1, [
        make_chunk(text="alpha chunk one", chunk_type="paragraph",
                   heading_context=["Intro"], source_line_start=1, source_line_end=2),
        make_chunk(text="alpha chunk two", chunk_type="heading",
                   heading_context=["Intro", "Details"], heading_level=2,
                   page_number=3, source_line_start=3, source_line_end=4),
    ], [make_embedding(seed=10), make_embedding(seed=20)])

    doc2 = upsert_document(conn, "/test/beta.md", "document", 1)
    insert_chunks(conn, doc2, [
        make_chunk(text="beta chunk one", chunk_type="code_block",
                   source_line_start=1, source_line_end=5),
    ], [make_embedding(seed=30)])


def test_search_chunks_returns_results(conn):
    _seed_search_data(conn)
    results = search_chunks(conn, make_embedding(seed=10))
    assert isinstance(results, list)
    assert len(results) > 0
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_chunks_ordered_by_similarity(conn):
    _seed_search_data(conn)
    # Query with seed=10 embedding — should match alpha chunk one (seed=10) first
    results = search_chunks(conn, make_embedding(seed=10))
    assert results[0].text == "alpha chunk one"
    assert results[0].similarity > 0.99  # near-perfect match


def test_search_chunks_top_k(conn):
    _seed_search_data(conn)
    results = search_chunks(conn, make_embedding(seed=10), top_k=2)
    assert len(results) <= 2


def test_search_chunks_filter_source(conn):
    _seed_search_data(conn)
    results = search_chunks(conn, make_embedding(seed=10), source_paths=["/test/beta.md"])
    assert all(r.source_path == "/test/beta.md" for r in results)
    assert len(results) == 1


def test_search_chunks_filter_chunk_type(conn):
    _seed_search_data(conn)
    results = search_chunks(conn, make_embedding(seed=10), chunk_types=["heading"])
    assert all(r.chunk_type == "heading" for r in results)
    assert len(results) == 1


def test_search_chunks_threshold(conn):
    _seed_search_data(conn)
    # High threshold should exclude non-exact matches
    results = search_chunks(conn, make_embedding(seed=10), threshold=0.99)
    assert len(results) == 1
    assert results[0].text == "alpha chunk one"


def test_search_chunks_empty_table(conn):
    results = search_chunks(conn, make_embedding(seed=10))
    assert results == []


# --- list_documents / get_document / get_chunk_summary / delete_documents tests ---


def test_list_documents_empty(conn):
    assert list_documents(conn) == []


def test_list_documents_returns_all(conn):
    upsert_document(conn, "/test/a.md", "document", 2)
    upsert_document(conn, "/test/b.md", "per_page", 5)
    docs = list_documents(conn)
    assert len(docs) == 2
    assert all(isinstance(d, DocumentInfo) for d in docs)
    # Most recent first
    assert docs[0].source_path == "/test/b.md"
    assert docs[1].source_path == "/test/a.md"


def test_get_document_found(conn):
    upsert_document(conn, "/test/found.md", "document", 3)
    doc = get_document(conn, "/test/found.md")
    assert doc is not None
    assert isinstance(doc, DocumentInfo)
    assert doc.source_path == "/test/found.md"
    assert doc.mode == "document"
    assert doc.total_chunks == 3


def test_get_document_not_found(conn):
    assert get_document(conn, "/no/such/path.md") is None


def test_get_chunk_summary(conn):
    doc_id = upsert_document(conn, "/test/summary.md", "document", 3)
    chunks = [
        make_chunk(text="paragraph one", chunk_type="paragraph"),
        make_chunk(text="paragraph two longer", chunk_type="paragraph"),
        make_chunk(text="a heading", chunk_type="heading"),
    ]
    embeddings = [make_embedding(seed=i) for i in range(3)]
    insert_chunks(conn, doc_id, chunks, embeddings)

    summaries = get_chunk_summary(conn, doc_id)
    assert all(isinstance(s, ChunkSummary) for s in summaries)
    by_type = {s.chunk_type: s for s in summaries}
    assert by_type["paragraph"].count == 2
    assert by_type["paragraph"].total_chars == len("paragraph one") + len("paragraph two longer")
    assert by_type["heading"].count == 1
    assert by_type["heading"].total_chars == len("a heading")


def test_delete_documents_single(conn):
    upsert_document(conn, "/test/del.md", "document", 1)
    count = delete_documents(conn, ["/test/del.md"])
    assert count == 1
    assert get_document(conn, "/test/del.md") is None


def test_delete_documents_multiple(conn):
    upsert_document(conn, "/test/d1.md", "document", 1)
    upsert_document(conn, "/test/d2.md", "document", 1)
    count = delete_documents(conn, ["/test/d1.md", "/test/d2.md"])
    assert count == 2
    assert list_documents(conn) == []


def test_delete_documents_not_found(conn):
    count = delete_documents(conn, ["/no/such/path.md"])
    assert count == 0
