"""Smoke tests for the GUI IngestWorker batch behavior.

These tests call ``_run_pipeline()`` (or ``run()``) directly, bypassing the
QThread event loop.  A session-scoped ``QApplication`` is enough for Qt
signals to work.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from chunk_embed.pipeline import IngestResult


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# -- helpers -----------------------------------------------------------------

def _make_worker(file_paths: list[str], **kw):
    from chunk_embed.gui import IngestWorker

    defaults = dict(split=False, database_url="postgresql://localhost/test", dry_run=True)
    defaults.update(kw)
    return IngestWorker(file_paths=file_paths, **defaults)


def _ok_result(path: str) -> IngestResult:
    return IngestResult(
        source_path=path, doc_id=1, num_chunks=5, num_embeddings=5, dry_run=True,
    )


# -- tests -------------------------------------------------------------------

def test_ingest_worker_single_file(qapp):
    worker = _make_worker(["/tmp/a.json"])
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with (
        patch("chunk_embed.embed.BgeM3Embedder") as MockEmb,
        patch("chunk_embed.pipeline.ingest_one_file", return_value=_ok_result("/tmp/a.json")),
    ):
        MockEmb.return_value = MagicMock()
        worker._run_pipeline()

    finished.assert_called_once()
    msg = finished.call_args[0][0]
    assert "1 succeeded" in msg
    error.assert_not_called()


def test_ingest_worker_multiple_files(qapp):
    paths = ["/tmp/a.json", "/tmp/b.json", "/tmp/c.json"]
    worker = _make_worker(paths)
    finished = MagicMock()
    worker.finished.connect(finished)

    with (
        patch("chunk_embed.embed.BgeM3Embedder") as MockEmb,
        patch("chunk_embed.pipeline.ingest_one_file", side_effect=[_ok_result(p) for p in paths]),
    ):
        MockEmb.return_value = MagicMock()
        worker._run_pipeline()

    msg = finished.call_args[0][0]
    assert "3 succeeded" in msg
    assert "3 files" in msg


def test_ingest_worker_partial_failure(qapp):
    paths = ["/tmp/a.json", "/tmp/b.json", "/tmp/c.json"]
    worker = _make_worker(paths)
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with (
        patch("chunk_embed.embed.BgeM3Embedder") as MockEmb,
        patch("chunk_embed.pipeline.ingest_one_file", side_effect=[
            _ok_result(paths[0]),
            RuntimeError("parse failed"),
            _ok_result(paths[2]),
        ]),
    ):
        MockEmb.return_value = MagicMock()
        worker._run_pipeline()

    msg = finished.call_args[0][0]
    assert "2 succeeded" in msg
    assert "1 failed" in msg
    assert "3 files" in msg
    error.assert_not_called()


def test_ingest_worker_all_fail(qapp):
    paths = ["/tmp/a.json", "/tmp/b.json", "/tmp/c.json"]
    worker = _make_worker(paths)
    finished = MagicMock()
    worker.finished.connect(finished)

    with (
        patch("chunk_embed.embed.BgeM3Embedder") as MockEmb,
        patch("chunk_embed.pipeline.ingest_one_file", side_effect=RuntimeError("boom")),
    ):
        MockEmb.return_value = MagicMock()
        worker._run_pipeline()

    msg = finished.call_args[0][0]
    assert "0 succeeded" in msg
    assert "3 failed" in msg


def test_ingest_worker_log_prefix(qapp):
    paths = ["/tmp/a.json", "/tmp/b.json", "/tmp/c.json"]
    worker = _make_worker(paths)
    log_msgs: list[str] = []
    worker.log.connect(log_msgs.append)

    with (
        patch("chunk_embed.embed.BgeM3Embedder") as MockEmb,
        patch("chunk_embed.pipeline.ingest_one_file", side_effect=[_ok_result(p) for p in paths]),
    ):
        MockEmb.return_value = MagicMock()
        worker._run_pipeline()

    prefixed = [m for m in log_msgs if m.startswith("[")]
    assert any("[1/3]" in m for m in prefixed)
    assert any("[2/3]" in m for m in prefixed)
    assert any("[3/3]" in m for m in prefixed)


def test_ingest_worker_embedder_loaded_once(qapp):
    paths = ["/tmp/a.json", "/tmp/b.json", "/tmp/c.json"]
    worker = _make_worker(paths)

    with (
        patch("chunk_embed.embed.BgeM3Embedder") as MockEmb,
        patch("chunk_embed.pipeline.ingest_one_file", side_effect=[_ok_result(p) for p in paths]),
    ):
        MockEmb.return_value = MagicMock()
        worker._run_pipeline()

    MockEmb.assert_called_once()


def test_ingest_worker_fatal_embedder_error(qapp):
    worker = _make_worker(["/tmp/a.json"])
    error = MagicMock()
    finished = MagicMock()
    worker.error.connect(error)
    worker.finished.connect(finished)

    with patch("chunk_embed.embed.BgeM3Embedder", side_effect=RuntimeError("GPU OOM")):
        worker.run()

    error.assert_called_once()
    assert "GPU OOM" in error.call_args[0][0]
    finished.assert_not_called()


def test_ingest_worker_passes_allowed_types(qapp):
    """allowed_types flows through to ingest_one_file."""
    types = frozenset({"paragraph", "heading"})
    worker = _make_worker(["/tmp/a.json"], allowed_types=types)
    finished = MagicMock()
    worker.finished.connect(finished)

    with (
        patch("chunk_embed.embed.BgeM3Embedder") as MockEmb,
        patch("chunk_embed.pipeline.ingest_one_file", return_value=_ok_result("/tmp/a.json")) as mock_ingest,
    ):
        MockEmb.return_value = MagicMock()
        worker._run_pipeline()

    mock_ingest.assert_called_once()
    assert mock_ingest.call_args.kwargs.get("allowed_types") == types


def test_ingest_worker_allowed_types_none_by_default(qapp):
    """Without allowed_types, it defaults to None."""
    worker = _make_worker(["/tmp/a.json"])
    assert worker.allowed_types is None


# -- DocsListWorker tests ---------------------------------------------------

def test_docs_list_worker(qapp):
    from chunk_embed.gui import DocsListWorker
    from chunk_embed.models import DocumentInfo
    from datetime import datetime

    fake_docs = [
        DocumentInfo(id=1, source_path="/a.md", mode="document", total_chunks=10, created_at=datetime(2026, 1, 1)),
        DocumentInfo(id=2, source_path="/b.md", mode="document", total_chunks=5, created_at=datetime(2026, 1, 2)),
    ]

    worker = DocsListWorker(database_url="postgresql://localhost/test")
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with patch("psycopg.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)
        with patch("chunk_embed.store.list_documents", return_value=fake_docs):
            worker.run()

    finished.assert_called_once_with(fake_docs)
    error.assert_not_called()


def test_docs_list_worker_error(qapp):
    from chunk_embed.gui import DocsListWorker

    worker = DocsListWorker(database_url="postgresql://localhost/test")
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with patch("psycopg.connect", side_effect=RuntimeError("connection refused")):
        worker.run()

    error.assert_called_once()
    assert "connection refused" in error.call_args[0][0]
    finished.assert_not_called()


# -- DocsDetailWorker tests -------------------------------------------------

def test_docs_detail_worker(qapp):
    from chunk_embed.gui import DocsDetailWorker
    from chunk_embed.models import ChunkSummary

    fake_summaries = [
        ChunkSummary(chunk_type="paragraph", count=8, total_chars=2000),
        ChunkSummary(chunk_type="heading", count=3, total_chars=120),
    ]

    worker = DocsDetailWorker(database_url="postgresql://localhost/test", document_id=42)
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with patch("psycopg.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)
        with patch("chunk_embed.store.get_chunk_summary", return_value=fake_summaries):
            worker.run()

    finished.assert_called_once_with(42, fake_summaries)
    error.assert_not_called()


# -- DocsDeleteWorker tests -------------------------------------------------

def test_docs_delete_worker(qapp):
    from chunk_embed.gui import DocsDeleteWorker

    worker = DocsDeleteWorker(
        database_url="postgresql://localhost/test",
        source_paths=["/a.md", "/b.md"],
    )
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with patch("psycopg.connect") as mock_connect:
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_connect.return_value.__exit__ = MagicMock(return_value=False)
        with patch("chunk_embed.store.delete_documents", return_value=2):
            worker.run()

    finished.assert_called_once_with(2)
    mock_conn.commit.assert_called_once()
    error.assert_not_called()


def test_docs_delete_worker_error(qapp):
    from chunk_embed.gui import DocsDeleteWorker

    worker = DocsDeleteWorker(
        database_url="postgresql://localhost/test",
        source_paths=["/a.md"],
    )
    finished = MagicMock()
    error = MagicMock()
    worker.finished.connect(finished)
    worker.error.connect(error)

    with patch("psycopg.connect", side_effect=RuntimeError("connection refused")):
        worker.run()

    error.assert_called_once()
    assert "connection refused" in error.call_args[0][0]
    finished.assert_not_called()
