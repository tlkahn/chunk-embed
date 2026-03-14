import json

from chunk_embed.format import format_results_human, format_results_json
from chunk_embed.models import SearchResult


def _result(**overrides) -> SearchResult:
    defaults = dict(
        similarity=0.8742,
        source_path="/path/to/file.md",
        text="Hello world chunk text.",
        chunk_type="paragraph",
        heading_context=["Introduction", "Getting Started"],
        heading_level=2,
        page_number=2,
        source_line_start=3,
        source_line_end=5,
    )
    defaults.update(overrides)
    return SearchResult(**defaults)


def test_format_human_single():
    out = format_results_human([_result()])
    assert "[1]" in out
    assert "0.8742" in out
    assert "/path/to/file.md" in out
    assert "Hello world chunk text." in out


def test_format_human_multiple():
    out = format_results_human([_result(similarity=0.9), _result(similarity=0.7)])
    assert "[1]" in out
    assert "[2]" in out
    assert out.index("[1]") < out.index("[2]")


def test_format_human_truncation():
    long_text = "x" * 300
    out = format_results_human([_result(text=long_text)])
    assert "..." in out
    # Should not contain the full 300-char text
    assert "x" * 300 not in out


def test_format_human_page_number():
    with_page = format_results_human([_result(page_number=5)])
    assert "page 5" in with_page

    without_page = format_results_human([_result(page_number=None)])
    assert "page" not in without_page


def test_format_human_heading_context():
    out = format_results_human([_result(heading_context=["A", "B"])])
    assert "A > B" in out


def test_format_human_empty():
    out = format_results_human([])
    assert out == "No results found."


def test_format_json_structure():
    out = format_results_json([_result()])
    data = json.loads(out)
    assert isinstance(data, list)
    assert len(data) == 1
    item = data[0]
    expected_keys = {"rank", "similarity", "source_path", "chunk_type",
                     "heading_context", "page_number", "source_line_start",
                     "source_line_end", "text"}
    assert set(item.keys()) == expected_keys
    assert item["rank"] == 1


def test_format_json_full_text():
    long_text = "x" * 300
    out = format_results_json([_result(text=long_text)])
    data = json.loads(out)
    assert data[0]["text"] == long_text  # NOT truncated


def test_format_json_empty():
    out = format_results_json([])
    assert json.loads(out) == []
