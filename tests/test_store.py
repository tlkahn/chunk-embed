import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

from chunk_embed.store import ensure_schema, upsert_document, insert_chunks
from tests.conftest import make_chunk

pytestmark = pytest.mark.integration

TEST_DB_URL = "postgresql://localhost/chunk_embed_test"


@pytest.fixture
def conn():
    with psycopg.connect(TEST_DB_URL, autocommit=False) as c:
        register_vector(c)
        ensure_schema(c)
        yield c
        c.rollback()


def _make_embedding(dim: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(0)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


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
    embeddings = [_make_embedding() for _ in range(3)]
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
    embeddings = [_make_embedding() for _ in range(2)]
    insert_chunks(conn, doc_id, chunks, embeddings)

    rows = conn.execute(
        "SELECT page_number FROM chunks WHERE document_id = %s ORDER BY chunk_index",
        (doc_id,),
    ).fetchall()
    assert [r[0] for r in rows] == [1, 2]


def test_reingest_replaces_document(conn):
    doc_id_1 = upsert_document(conn, "/test/file.md", "document", 2)
    chunks = [make_chunk(text=f"old {i}") for i in range(2)]
    embeddings = [_make_embedding() for _ in range(2)]
    insert_chunks(conn, doc_id_1, chunks, embeddings)

    doc_id_2 = upsert_document(conn, "/test/file.md", "document", 1)
    new_chunks = [make_chunk(text="new 0")]
    new_embeddings = [_make_embedding()]
    insert_chunks(conn, doc_id_2, new_chunks, new_embeddings)

    assert doc_id_2 != doc_id_1
    old_rows = conn.execute("SELECT id FROM chunks WHERE document_id = %s", (doc_id_1,)).fetchall()
    assert old_rows == []
    new_rows = conn.execute("SELECT text FROM chunks WHERE document_id = %s", (doc_id_2,)).fetchall()
    assert len(new_rows) == 1
    assert new_rows[0][0] == "new 0"


def test_delete_document_cascades(conn):
    doc_id = upsert_document(conn, "/test/cascade.md", "document", 1)
    insert_chunks(conn, doc_id, [make_chunk()], [_make_embedding()])
    conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    rows = conn.execute("SELECT id FROM chunks WHERE document_id = %s", (doc_id,)).fetchall()
    assert rows == []


def test_vector_stored_correctly(conn):
    doc_id = upsert_document(conn, "/test/vec.md", "document", 1)
    original = _make_embedding()
    insert_chunks(conn, doc_id, [make_chunk()], [original])

    row = conn.execute("SELECT embedding FROM chunks WHERE document_id = %s", (doc_id,)).fetchone()
    stored = np.array(row[0], dtype=np.float32)
    np.testing.assert_allclose(stored, original, atol=1e-5)
