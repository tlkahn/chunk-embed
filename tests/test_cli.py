import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from click.testing import CliRunner

from chunk_embed.cli import main
from chunk_embed.models import SearchResult
from tests.conftest import MockEmbedder

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def mock_deps():
    """Patch BgeM3Embedder and DB connection for CLI ingest tests."""
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


def test_cli_group_help(runner):
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "ingest" in result.output
    assert "query" in result.output


def test_cli_help(runner):
    result = runner.invoke(main, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "source" in result.output.lower()


def test_cli_stdin_pipe(runner, mock_deps):
    sample = (FIXTURES / "sample_chunks.json").read_text()
    result = runner.invoke(main, ["ingest", "--source", "test.md"], input=sample)
    assert result.exit_code == 0, result.output


def test_cli_file_input(runner, mock_deps, tmp_path):
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    result = runner.invoke(main, ["ingest", str(src)])
    assert result.exit_code == 0, result.output


def test_cli_missing_source_for_stdin(runner, mock_deps):
    sample = (FIXTURES / "sample_chunks.json").read_text()
    result = runner.invoke(main, ["ingest", "-"], input=sample)
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
            result = runner.invoke(main, ["ingest", str(src)])
            assert result.exit_code == 0, result.output
            mock_upsert.assert_called_once()
            assert str(src) in mock_upsert.call_args[0][1]


def test_cli_invalid_json(runner, mock_deps):
    result = runner.invoke(main, ["ingest", "--source", "x.md"], input="not json {{{")
    assert result.exit_code == 1


def test_cli_help_shows_no_split_but_not_lang(runner):
    result = runner.invoke(main, ["ingest", "--help"])
    assert "--lang" not in result.output
    assert "--no-split" in result.output


def test_cli_no_split_skips_split_chunks(runner, mock_deps, tmp_path):
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    with patch("chunk_embed.cli.split_chunks") as mock_split:
        result = runner.invoke(main, ["ingest", str(src), "--no-split"])
        assert result.exit_code == 0, result.output
        mock_split.assert_not_called()


def test_cli_dry_run(runner, tmp_path):
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    mock_emb = MockEmbedder()
    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.psycopg.connect") as mock_connect,
        patch("chunk_embed.cli.insert_chunks") as mock_insert,
    ):
        result = runner.invoke(main, ["ingest", str(src), "--dry-run"])
        assert result.exit_code == 0, result.output
        mock_connect.assert_not_called()
        mock_insert.assert_not_called()
        assert "dry" in result.output.lower() or "skip" in result.output.lower()


# --- query subcommand tests ---


_SAMPLE_RESULTS = [
    SearchResult(
        similarity=0.8742,
        source_path="/test/file.md",
        text="Hello world",
        chunk_type="paragraph",
        heading_context=["Intro"],
        heading_level=None,
        page_number=None,
        source_line_start=1,
        source_line_end=2,
    ),
]


@pytest.fixture
def mock_query_deps():
    """Patch embedder, DB, and search_chunks for query CLI tests."""
    mock_emb = MockEmbedder()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.psycopg.connect", return_value=mock_conn),
        patch("chunk_embed.cli.register_vector"),
        patch("chunk_embed.cli.search_chunks", return_value=_SAMPLE_RESULTS) as mock_search,
    ):
        yield mock_search


def test_query_help(runner):
    result = runner.invoke(main, ["query", "--help"])
    assert result.exit_code == 0
    for term in ["QUERY_TEXT", "--top-k", "--source", "--chunk-type", "--threshold", "--json"]:
        assert term in result.output


def test_query_basic(runner, mock_query_deps):
    result = runner.invoke(main, ["query", "hello world"])
    assert result.exit_code == 0, result.output
    assert "Hello world" in result.output


def test_query_json_flag(runner, mock_query_deps):
    result = runner.invoke(main, ["query", "hello world", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert isinstance(data, list)
    assert data[0]["text"] == "Hello world"


def test_query_top_k(runner, mock_query_deps):
    mock_search = mock_query_deps
    runner.invoke(main, ["query", "test", "--top-k", "5"])
    mock_search.assert_called_once()
    assert mock_search.call_args.kwargs.get("top_k") == 5 or mock_search.call_args[1].get("top_k") == 5


def test_query_source_filter(runner, mock_query_deps):
    mock_search = mock_query_deps
    runner.invoke(main, ["query", "test", "--source", "foo.md"])
    assert mock_search.call_args.kwargs.get("source_path") == "foo.md" or \
           mock_search.call_args[1].get("source_path") == "foo.md"


def test_query_no_results(runner):
    mock_emb = MockEmbedder()
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)

    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.psycopg.connect", return_value=mock_conn),
        patch("chunk_embed.cli.register_vector"),
        patch("chunk_embed.cli.search_chunks", return_value=[]),
    ):
        result = runner.invoke(main, ["query", "nothing"])
        assert result.exit_code == 0
        assert "No results found." in result.output
