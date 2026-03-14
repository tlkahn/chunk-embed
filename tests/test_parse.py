import json
from pathlib import Path

import pytest

from chunk_embed.models import ChunkData, ChunksInput
from chunk_embed.parse import parse_chunks, ParseError

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_valid_document_mode():
    raw = (FIXTURES / "sample_chunks.json").read_text()
    result = parse_chunks(raw)
    assert isinstance(result, ChunksInput)
    assert result.total_chunks == 3
    assert result.mode == "document"
    assert len(result.chunks) == 3
    assert result.chunks[0].text == "# Introduction"
    assert result.chunks[0].chunk_type == "heading"
    assert result.chunks[0].heading_context == ["Introduction"]
    assert result.chunks[0].heading_level == 1
    assert result.chunks[0].page_number is None
    assert result.chunks[0].source_line_start == 1
    assert result.chunks[0].source_line_end == 1


def test_parse_per_page_mode():
    raw = (FIXTURES / "sample_chunks_perpage.json").read_text()
    result = parse_chunks(raw)
    assert result.mode == "per_page"
    assert result.chunks[0].page_number == 1
    assert result.chunks[1].page_number == 2


def test_parse_invalid_json_raises():
    with pytest.raises(ParseError, match="Invalid JSON"):
        parse_chunks("not json {{{")


def test_parse_missing_chunks_key_raises():
    with pytest.raises(ParseError, match="Missing required"):
        parse_chunks(json.dumps({"total_chunks": 1, "mode": "document"}))


def test_parse_chunk_count_mismatch_raises():
    data = {
        "total_chunks": 5,
        "mode": "document",
        "chunks": [
            {
                "text": "hello",
                "chunk_type": "paragraph",
                "heading_context": [],
                "heading_level": None,
                "page_number": None,
                "source_line_start": 1,
                "source_line_end": 1,
            }
        ],
    }
    with pytest.raises(ParseError, match="mismatch"):
        parse_chunks(json.dumps(data))


def test_parse_preserves_unicode():
    data = {
        "total_chunks": 1,
        "mode": "document",
        "chunks": [
            {
                "text": "धर्मक्षेत्रे कुरुक्षेत्रे",
                "chunk_type": "paragraph",
                "heading_context": ["अध्याय १"],
                "heading_level": None,
                "page_number": None,
                "source_line_start": 1,
                "source_line_end": 1,
            }
        ],
    }
    result = parse_chunks(json.dumps(data))
    assert result.chunks[0].text == "धर्मक्षेत्रे कुरुक्षेत्रे"
    assert result.chunks[0].heading_context == ["अध्याय १"]


def test_parse_empty_chunks_list():
    data = {"total_chunks": 0, "mode": "document", "chunks": []}
    result = parse_chunks(json.dumps(data))
    assert result.total_chunks == 0
    assert result.chunks == []


def test_parse_all_chunk_types():
    chunk_types = [
        "heading", "paragraph", "list_item", "code_block", "table",
        "block_quote", "definition_item", "math_block", "theorem",
    ]
    chunks = [
        {
            "text": f"text for {ct}",
            "chunk_type": ct,
            "heading_context": [],
            "heading_level": None,
            "page_number": None,
            "source_line_start": i,
            "source_line_end": i,
        }
        for i, ct in enumerate(chunk_types, 1)
    ]
    data = {"total_chunks": len(chunks), "mode": "document", "chunks": chunks}
    result = parse_chunks(json.dumps(data))
    assert [c.chunk_type for c in result.chunks] == chunk_types
