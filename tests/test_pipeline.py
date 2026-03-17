from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from chunk_embed.models import ALL_CHUNK_TYPES, TEXTUAL_TYPES
from chunk_embed.pipeline import (
    filter_chunks,
    resolve_ingest_types,
    resolve_paths,
    read_or_chunk_file,
    ingest_one_file,
    IngestResult,
)
from tests.conftest import MockEmbedder, make_chunk


def test_resolve_paths_single_json_file(tmp_path):
    f = tmp_path / "chunks.json"
    f.write_text("{}")
    assert resolve_paths([str(f)]) == [f]


def test_resolve_paths_single_md_file(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# hello")
    assert resolve_paths([str(f)]) == [f]


def test_resolve_paths_explicit_file_any_extension(tmp_path):
    """Explicit file paths are always included regardless of extension."""
    f = tmp_path / "notes.txt"
    f.write_text("hello")
    assert resolve_paths([str(f)]) == [f]


def test_resolve_paths_directory_filters_by_extension(tmp_path):
    md = tmp_path / "a.md"
    json_ = tmp_path / "b.json"
    txt = tmp_path / "c.txt"
    for f in (md, json_, txt):
        f.write_text("x")
    result = resolve_paths([str(tmp_path)])
    assert md in result
    assert json_ in result
    assert txt not in result


def test_resolve_paths_directory_sorted(tmp_path):
    b = tmp_path / "b.md"
    a = tmp_path / "a.md"
    for f in (b, a):
        f.write_text("x")
    result = resolve_paths([str(tmp_path)])
    assert result == [a, b]


def test_resolve_paths_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    top = tmp_path / "top.md"
    nested = sub / "nested.md"
    for f in (top, nested):
        f.write_text("x")
    result = resolve_paths([str(tmp_path)], recursive=True)
    assert nested in result
    assert top in result


def test_resolve_paths_non_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    top = tmp_path / "top.md"
    nested = sub / "nested.md"
    for f in (top, nested):
        f.write_text("x")
    result = resolve_paths([str(tmp_path)], recursive=False)
    assert top in result
    assert nested not in result


def test_resolve_paths_glob_pattern(tmp_path):
    md = tmp_path / "a.md"
    json_ = tmp_path / "b.json"
    for f in (md, json_):
        f.write_text("x")
    result = resolve_paths([str(tmp_path)], glob_pattern="*.md")
    assert result == [md]


def test_resolve_paths_mixed_files_and_dirs(tmp_path):
    d = tmp_path / "docs"
    d.mkdir()
    f1 = tmp_path / "standalone.json"
    f2 = d / "chapter.md"
    for f in (f1, f2):
        f.write_text("x")
    result = resolve_paths([str(f1), str(d)])
    assert f1 in result
    assert f2 in result


def test_resolve_paths_empty_directory(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    assert resolve_paths([str(d)]) == []


def test_resolve_paths_nonexistent_path():
    with pytest.raises(FileNotFoundError, match="Path not found"):
        resolve_paths(["/nonexistent/path/to/nowhere"])


def test_resolve_paths_multiple_files(tmp_path):
    f1 = tmp_path / "a.json"
    f2 = tmp_path / "b.md"
    for f in (f1, f2):
        f.write_text("x")
    result = resolve_paths([str(f1), str(f2)])
    assert result == [f1, f2]


# --- read_or_chunk_file ---


def test_read_or_chunk_file_json(tmp_path):
    """JSON files are read directly."""
    f = tmp_path / "chunks.json"
    f.write_text('{"total_chunks": 0}')
    assert read_or_chunk_file(f) == '{"total_chunks": 0}'


def test_read_or_chunk_file_markdown(tmp_path):
    """Markdown files trigger text-chunker subprocess."""
    f = tmp_path / "doc.md"
    f.write_text("# Hello")
    fake_json = '{"total_chunks": 1, "mode": "document", "chunks": []}'
    mock_result = MagicMock(stdout=fake_json)
    with patch("chunk_embed.pipeline.subprocess.run", return_value=mock_result) as mock_run:
        result = read_or_chunk_file(f)
        assert result == fake_json
        mock_run.assert_called_once_with(
            ["text-chunker", "--json", "chunks", str(f)],
            capture_output=True, text=True, check=True,
        )


def test_read_or_chunk_file_markdown_text_chunker_missing(tmp_path):
    """Raises FileNotFoundError when text-chunker is not installed."""
    f = tmp_path / "doc.md"
    f.write_text("# Hello")
    with patch("chunk_embed.pipeline.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            read_or_chunk_file(f)


def test_read_or_chunk_file_markdown_text_chunker_fails(tmp_path):
    """Raises RuntimeError when text-chunker exits non-zero."""
    import subprocess as sp
    f = tmp_path / "doc.md"
    f.write_text("# Hello")
    with patch(
        "chunk_embed.pipeline.subprocess.run",
        side_effect=sp.CalledProcessError(1, "text-chunker", stderr="bad input"),
    ):
        with pytest.raises(RuntimeError, match="text-chunker failed"):
            read_or_chunk_file(f)


# --- ingest_one_file ---

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mock_pipeline_db():
    """Patch DB dependencies for ingest_one_file tests."""
    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    with (
        patch("chunk_embed.pipeline.psycopg.connect", return_value=mock_conn),
        patch("chunk_embed.pipeline.register_vector"),
        patch("chunk_embed.pipeline.ensure_schema"),
        patch("chunk_embed.pipeline.upsert_document", return_value=42) as mock_upsert,
        patch("chunk_embed.pipeline.insert_chunks"),
    ):
        yield mock_upsert


def test_ingest_one_file_json(tmp_path, mock_pipeline_db):
    """Full pipeline: JSON file → parse → embed → store."""
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    embedder = MockEmbedder()
    result = ingest_one_file(
        file_path=src,
        source=None,
        embedder=embedder,
        split=False,
        database_url="postgresql://localhost/test",
    )
    assert isinstance(result, IngestResult)
    assert result.source_path == str(src)
    assert result.doc_id == 42
    assert result.num_chunks == 3
    assert result.num_embeddings == 3
    assert result.dry_run is False


def test_ingest_one_file_dry_run(tmp_path):
    """Dry run skips DB entirely."""
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    embedder = MockEmbedder()
    with patch("chunk_embed.pipeline.psycopg.connect") as mock_connect:
        result = ingest_one_file(
            file_path=src,
            source="test.md",
            embedder=embedder,
            split=False,
            dry_run=True,
        )
        mock_connect.assert_not_called()
    assert result.doc_id is None
    assert result.dry_run is True
    assert result.num_embeddings == 3


def test_ingest_one_file_custom_source(tmp_path, mock_pipeline_db):
    """Explicit source overrides file path."""
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    embedder = MockEmbedder()
    mock_upsert = mock_pipeline_db
    result = ingest_one_file(
        file_path=src,
        source="custom/source.md",
        embedder=embedder,
        split=False,
        database_url="postgresql://localhost/test",
    )
    assert result.source_path == "custom/source.md"
    mock_upsert.assert_called_once()
    assert mock_upsert.call_args[0][1] == "custom/source.md"


def test_ingest_one_file_parse_error(tmp_path):
    """ParseError from invalid JSON propagates."""
    src = tmp_path / "bad.json"
    src.write_text("not valid json {{{")
    embedder = MockEmbedder()
    from chunk_embed.parse import ParseError
    with pytest.raises(ParseError):
        ingest_one_file(
            file_path=src,
            source=None,
            embedder=embedder,
            split=False,
            dry_run=True,
        )


def test_ingest_one_file_progress_callbacks(tmp_path):
    """on_embed_progress and on_store_progress are forwarded."""
    src = tmp_path / "chunks.json"
    src.write_text((FIXTURES / "sample_chunks.json").read_text())
    embedder = MockEmbedder()
    embed_calls: list[tuple[int, int]] = []
    store_calls: list[tuple[int, int]] = []

    mock_conn = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    with (
        patch("chunk_embed.pipeline.psycopg.connect", return_value=mock_conn),
        patch("chunk_embed.pipeline.register_vector"),
        patch("chunk_embed.pipeline.ensure_schema"),
        patch("chunk_embed.pipeline.upsert_document", return_value=1),
        patch("chunk_embed.pipeline.insert_chunks") as mock_insert,
        patch("chunk_embed.pipeline.embed_chunks") as mock_embed,
    ):
        mock_embed.return_value = [MagicMock() for _ in range(3)]
        result = ingest_one_file(
            file_path=src,
            source=None,
            embedder=embedder,
            split=False,
            database_url="postgresql://localhost/test",
            on_embed_progress=lambda d, t: embed_calls.append((d, t)),
            on_store_progress=lambda d, t: store_calls.append((d, t)),
        )
    # embed_chunks should have been called with on_progress
    mock_embed.assert_called_once()
    assert mock_embed.call_args.kwargs.get("on_progress") is not None
    # insert_chunks should have been called with on_progress
    mock_insert.assert_called_once()
    assert mock_insert.call_args.kwargs.get("on_progress") is not None


# --- filter_chunks ---


def test_filter_chunks_keeps_matching():
    chunks = [
        make_chunk("hello", chunk_type="paragraph"),
        make_chunk("world", chunk_type="code_block"),
        make_chunk("heading", chunk_type="heading"),
    ]
    result = filter_chunks(chunks, frozenset({"paragraph", "heading"}))
    assert len(result) == 2
    assert all(c.chunk_type in {"paragraph", "heading"} for c in result)


def test_filter_chunks_empty_allowed():
    chunks = [make_chunk("hello", chunk_type="paragraph")]
    result = filter_chunks(chunks, frozenset())
    assert result == []


def test_filter_chunks_all_allowed():
    chunks = [
        make_chunk("a", chunk_type="paragraph"),
        make_chunk("b", chunk_type="code_block"),
        make_chunk("c", chunk_type="table"),
    ]
    result = filter_chunks(chunks, ALL_CHUNK_TYPES)
    assert len(result) == 3


def test_filter_chunks_default_textual():
    chunks = [
        make_chunk("a", chunk_type="paragraph"),
        make_chunk("b", chunk_type="code_block"),
        make_chunk("c", chunk_type="math_block"),
        make_chunk("d", chunk_type="table"),
        make_chunk("e", chunk_type="heading"),
    ]
    result = filter_chunks(chunks, TEXTUAL_TYPES)
    assert len(result) == 2
    assert {c.chunk_type for c in result} == {"paragraph", "heading"}


# --- resolve_ingest_types ---


def test_resolve_ingest_types_default():
    result = resolve_ingest_types()
    assert result == TEXTUAL_TYPES


def test_resolve_ingest_types_all():
    result = resolve_ingest_types(all_types=True)
    assert result == ALL_CHUNK_TYPES


def test_resolve_ingest_types_include():
    result = resolve_ingest_types(include_types=("code_block", "table"))
    assert result == frozenset({"code_block", "table"})


def test_resolve_ingest_types_exclude():
    result = resolve_ingest_types(exclude_types=("heading",))
    assert result == TEXTUAL_TYPES - {"heading"}


def test_resolve_ingest_types_all_minus_exclude():
    result = resolve_ingest_types(exclude_types=("code_block",), all_types=True)
    assert result == ALL_CHUNK_TYPES - {"code_block"}


def test_resolve_ingest_types_include_conflicts_with_exclude():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_ingest_types(include_types=("paragraph",), exclude_types=("heading",))


def test_resolve_ingest_types_include_conflicts_with_all():
    with pytest.raises(ValueError, match="cannot be combined"):
        resolve_ingest_types(include_types=("paragraph",), all_types=True)


def test_resolve_ingest_types_unknown_include():
    with pytest.raises(ValueError, match="Unknown chunk type"):
        resolve_ingest_types(include_types=("nonexistent",))


def test_resolve_ingest_types_unknown_exclude():
    with pytest.raises(ValueError, match="Unknown chunk type"):
        resolve_ingest_types(exclude_types=("nonexistent",))


# --- ingest_one_file with allowed_types ---

MIXED_FIXTURE = FIXTURES / "sample_chunks_mixed.json"


def test_ingest_one_file_default_filters_non_textual(tmp_path):
    """Default allowed_types=None filters out code_block, math_block, table."""
    src = tmp_path / "mixed.json"
    src.write_text(MIXED_FIXTURE.read_text())
    embedder = MockEmbedder()
    with patch("chunk_embed.pipeline.psycopg.connect") as mock_connect:
        result = ingest_one_file(
            file_path=src,
            source="test.md",
            embedder=embedder,
            split=False,
            dry_run=True,
        )
        mock_connect.assert_not_called()
    # mixed fixture has 5 chunks, 2 are textual (heading + paragraph)
    assert result.num_chunks == 2
    assert result.num_embeddings == 2


def test_ingest_one_file_all_types(tmp_path):
    """allowed_types=ALL_CHUNK_TYPES keeps everything."""
    src = tmp_path / "mixed.json"
    src.write_text(MIXED_FIXTURE.read_text())
    embedder = MockEmbedder()
    result = ingest_one_file(
        file_path=src,
        source="test.md",
        embedder=embedder,
        split=False,
        dry_run=True,
        allowed_types=ALL_CHUNK_TYPES,
    )
    assert result.num_chunks == 5
    assert result.num_embeddings == 5
