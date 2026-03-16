import json
from pathlib import Path
from unittest.mock import patch, MagicMock

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


@pytest.fixture
def mock_batch_deps():
    """Patch embedder and ingest_one_file for file/directory CLI tests."""
    mock_emb = MockEmbedder()
    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.ingest_one_file") as mock_ingest,
    ):
        from chunk_embed.pipeline import IngestResult
        mock_ingest.return_value = IngestResult(
            source_path="test", doc_id=1, num_chunks=3, num_embeddings=3, dry_run=False,
        )
        yield mock_ingest


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


def test_cli_file_input(runner, mock_batch_deps, tmp_path):
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    result = runner.invoke(main, ["ingest", str(src)])
    assert result.exit_code == 0, result.output


def test_cli_missing_source_for_stdin(runner, mock_deps):
    sample = (FIXTURES / "sample_chunks.json").read_text()
    result = runner.invoke(main, ["ingest", "-"], input=sample)
    assert result.exit_code != 0
    assert "source" in result.output.lower() or "required" in result.output.lower()


def test_cli_file_input_infers_source(runner, tmp_path):
    """File path is passed as source=None to ingest_one_file, which infers it."""
    src = tmp_path / "my_doc.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    from chunk_embed.pipeline import IngestResult
    mock_emb = MockEmbedder()
    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.ingest_one_file") as mock_ingest,
    ):
        mock_ingest.return_value = IngestResult(
            source_path=str(src), doc_id=1, num_chunks=3, num_embeddings=3, dry_run=False,
        )
        result = runner.invoke(main, ["ingest", str(src)])
        assert result.exit_code == 0, result.output
        mock_ingest.assert_called_once()
        # source should be None (ingest_one_file infers from file_path)
        assert mock_ingest.call_args.kwargs.get("source") is None


def test_cli_invalid_json(runner, mock_deps):
    result = runner.invoke(main, ["ingest", "--source", "x.md"], input="not json {{{")
    assert result.exit_code == 1


def test_cli_help_shows_no_split_but_not_lang(runner):
    result = runner.invoke(main, ["ingest", "--help"])
    assert "--lang" not in result.output
    assert "--no-split" in result.output


def test_cli_no_split_skips_split_chunks(runner, mock_batch_deps, tmp_path):
    """--no-split passes split=False to ingest_one_file."""
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    result = runner.invoke(main, ["ingest", str(src), "--no-split"])
    assert result.exit_code == 0, result.output
    mock_batch_deps.assert_called_once()
    assert mock_batch_deps.call_args.kwargs.get("split") is False


def test_cli_dry_run(runner, tmp_path):
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    from chunk_embed.pipeline import IngestResult
    mock_emb = MockEmbedder()
    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.ingest_one_file") as mock_ingest,
    ):
        mock_ingest.return_value = IngestResult(
            source_path=str(src), doc_id=None, num_chunks=3, num_embeddings=3, dry_run=True,
        )
        result = runner.invoke(main, ["ingest", str(src), "--dry-run"])
        assert result.exit_code == 0, result.output
        mock_ingest.assert_called_once()
        assert mock_ingest.call_args.kwargs.get("dry_run") is True


# --- batch ingest tests ---


def test_cli_multiple_files(runner, mock_batch_deps, tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    for f in (f1, f2):
        f.write_text((FIXTURES / "sample_chunks.json").read_text())
    result = runner.invoke(main, ["ingest", str(f1), str(f2)])
    assert result.exit_code == 0, result.output
    assert mock_batch_deps.call_count == 2


def test_cli_directory(runner, mock_batch_deps, tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.md"
    txt = tmp_path / "c.txt"  # should be ignored
    for f in (f1, f2, txt):
        f.write_text("x")
    result = runner.invoke(main, ["ingest", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert mock_batch_deps.call_count == 2


def test_cli_empty_directory(runner, mock_batch_deps, tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    result = runner.invoke(main, ["ingest", str(d)])
    assert result.exit_code != 0
    assert "no eligible files" in result.output.lower()


def test_cli_continue_on_error(runner, tmp_path):
    """Default: continues past failures, reports summary, exits 1."""
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    for f in (f1, f2):
        f.write_text((FIXTURES / "sample_chunks.json").read_text())

    from chunk_embed.pipeline import IngestResult
    ok = IngestResult(source_path="a", doc_id=1, num_chunks=3, num_embeddings=3, dry_run=False)
    mock_emb = MockEmbedder()

    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.ingest_one_file", side_effect=[Exception("boom"), ok]),
    ):
        result = runner.invoke(main, ["ingest", str(f1), str(f2)])
    assert result.exit_code == 1
    assert "1 succeeded" in result.output
    assert "1 failed" in result.output


def test_cli_fail_fast(runner, tmp_path):
    """--fail-fast stops on first error."""
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.json"
    for f in (f1, f2):
        f.write_text((FIXTURES / "sample_chunks.json").read_text())

    mock_emb = MockEmbedder()
    with (
        patch("chunk_embed.cli.BgeM3Embedder", return_value=mock_emb),
        patch("chunk_embed.cli.ingest_one_file", side_effect=Exception("boom")) as mock_ingest,
    ):
        result = runner.invoke(main, ["ingest", str(f1), str(f2), "--fail-fast"])
    assert result.exit_code == 1
    assert mock_ingest.call_count == 1  # stopped after first


def test_cli_glob_filter(runner, mock_batch_deps, tmp_path):
    md = tmp_path / "a.md"
    json_ = tmp_path / "b.json"
    for f in (md, json_):
        f.write_text("x")
    result = runner.invoke(main, ["ingest", str(tmp_path), "--glob", "*.md"])
    assert result.exit_code == 0, result.output
    assert mock_batch_deps.call_count == 1


def test_cli_no_recursive(runner, mock_batch_deps, tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    top = tmp_path / "top.md"
    nested = sub / "nested.md"
    for f in (top, nested):
        f.write_text("x")
    result = runner.invoke(main, ["ingest", str(tmp_path), "--no-recursive"])
    assert result.exit_code == 0, result.output
    assert mock_batch_deps.call_count == 1  # only top-level


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
    assert mock_search.call_args.kwargs.get("source_paths") == ["foo.md"] or \
           mock_search.call_args[1].get("source_paths") == ["foo.md"]


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
