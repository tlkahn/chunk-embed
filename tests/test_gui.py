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
