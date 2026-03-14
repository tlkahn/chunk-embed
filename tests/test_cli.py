import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from chunk_embed.cli import main
from tests.conftest import MockEmbedder

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_deps():
    """Patch BgeM3Embedder and DB connection for CLI tests."""
    mock_emb = MockEmbedder()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.psycopg.connect", return_value=mock_conn),
        patch("chunk_embed.cli.register_vector"),
        patch("chunk_embed.cli.ensure_schema"),
        patch("chunk_embed.cli.upsert_document", return_value=1),
        patch("chunk_embed.cli.insert_chunks"),
    ):
        yield


def test_cli_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "chunk-embed" in result.output.lower() or "source" in result.output.lower()


def test_cli_stdin_pipe(runner, mock_deps):
    sample = (FIXTURES / "sample_chunks.json").read_text()
    result = runner.invoke(main, ["--source", "test.md"], input=sample)
    assert result.exit_code == 0, result.output


def test_cli_file_input(runner, mock_deps, tmp_path):
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    result = runner.invoke(main, [str(src)])
    assert result.exit_code == 0, result.output


def test_cli_missing_source_for_stdin(runner, mock_deps):
    sample = (FIXTURES / "sample_chunks.json").read_text()
    result = runner.invoke(main, ["-"], input=sample)
    assert result.exit_code != 0
    assert "source" in result.output.lower() or "required" in result.output.lower()


def test_cli_file_input_infers_source(runner, mock_deps, tmp_path):
    src = tmp_path / "my_doc.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    with patch("chunk_embed.cli.upsert_document", return_value=1) as mock_upsert:
        with (
            patch("chunk_embed.cli.BgeM3Embedder", return_value=MockEmbedder()),
            patch("chunk_embed.cli.psycopg.connect") as mc,
            patch("chunk_embed.cli.register_vector"),
            patch("chunk_embed.cli.ensure_schema"),
            patch("chunk_embed.cli.insert_chunks"),
        ):
            mc.return_value.__enter__ = MagicMock(return_value=mc.return_value)
            mc.return_value.__exit__ = MagicMock(return_value=False)
            result = runner.invoke(main, [str(src)])
            assert result.exit_code == 0, result.output
            mock_upsert.assert_called_once()
            assert str(src) in mock_upsert.call_args[0][1]


def test_cli_invalid_json(runner, mock_deps):
    result = runner.invoke(main, ["--source", "x.md"], input="not json {{{")
    assert result.exit_code == 1


def test_cli_dry_run(runner, tmp_path):
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    mock_emb = MockEmbedder()
    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.psycopg.connect") as mock_connect,
        patch("chunk_embed.cli.insert_chunks") as mock_insert,
    ):
        result = runner.invoke(main, [str(src), "--dry-run"])
        assert result.exit_code == 0, result.output
        mock_connect.assert_not_called()
        mock_insert.assert_not_called()
        assert "dry" in result.output.lower() or "skip" in result.output.lower()
