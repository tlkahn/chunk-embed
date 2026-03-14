import subprocess
from dataclasses import replace
from unittest.mock import patch, MagicMock

import pytest

from chunk_embed.models import ChunkData
from chunk_embed.split import SPLITTABLE_TYPES, split_text, split_chunks
from tests.conftest import make_chunk


# --- SPLITTABLE_TYPES ---


def test_splittable_types_contains_expected():
    for t in ("paragraph", "block_quote", "definition_item", "theorem", "list_item", "heading"):
        assert t in SPLITTABLE_TYPES


def test_splittable_types_excludes_non_splittable():
    for t in ("code_block", "math_block", "table"):
        assert t not in SPLITTABLE_TYPES


# --- split_text ---


def test_split_text_calls_sentenza_with_correct_args():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="First sentence.\nSecond sentence.\n")
        split_text("First sentence. Second sentence.", "en")
        mock_run.assert_called_once_with(
            ["sentenza", "en"],
            input="First sentence. Second sentence.",
            capture_output=True,
            text=True,
            check=True,
        )


def test_split_text_returns_sentences_from_stdout():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Hello.\nWorld.\n")
        result = split_text("Hello. World.", "en")
        assert result == ["Hello.", "World."]


def test_split_text_filters_empty_lines():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Hello.\n\nWorld.\n\n")
        result = split_text("Hello. World.", "en")
        assert result == ["Hello.", "World."]


def test_split_text_empty_input_returns_without_subprocess():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        result = split_text("", "en")
        assert result == [""]
        mock_run.assert_not_called()


def test_split_text_whitespace_only_returns_without_subprocess():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        result = split_text("   \n  ", "en")
        assert result == ["   \n  "]
        mock_run.assert_not_called()


def test_split_text_file_not_found_falls_back():
    with patch("chunk_embed.split.subprocess.run", side_effect=FileNotFoundError):
        result = split_text("Some text.", "en")
        assert result == ["Some text."]


def test_split_text_called_process_error_falls_back():
    with patch(
        "chunk_embed.split.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "sentenza"),
    ):
        result = split_text("Some text.", "en")
        assert result == ["Some text."]


def test_split_text_passes_lang_code():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="Satz eins.\nSatz zwei.\n")
        split_text("Satz eins. Satz zwei.", "de")
        assert mock_run.call_args[0][0] == ["sentenza", "de"]


# --- split_chunks ---


def test_split_chunks_expands_paragraph():
    chunk = make_chunk("First. Second.", chunk_type="paragraph")
    with patch("chunk_embed.split.split_text", return_value=["First.", "Second."]):
        result = split_chunks([chunk], "en")
    assert len(result) == 2
    assert result[0].text == "First."
    assert result[1].text == "Second."


def test_split_chunks_skips_code_block():
    chunk = make_chunk("x = 1", chunk_type="code_block")
    with patch("chunk_embed.split.split_text") as mock_split:
        result = split_chunks([chunk], "en")
    mock_split.assert_not_called()
    assert result == [chunk]


def test_split_chunks_skips_math_block():
    chunk = make_chunk("E = mc^2", chunk_type="math_block")
    with patch("chunk_embed.split.split_text") as mock_split:
        result = split_chunks([chunk], "en")
    mock_split.assert_not_called()
    assert result == [chunk]


def test_split_chunks_skips_table():
    chunk = make_chunk("| a | b |", chunk_type="table")
    with patch("chunk_embed.split.split_text") as mock_split:
        result = split_chunks([chunk], "en")
    mock_split.assert_not_called()
    assert result == [chunk]


def test_split_chunks_preserves_metadata():
    chunk = make_chunk(
        "Sentence one. Sentence two.",
        chunk_type="paragraph",
        heading_context=["Chapter 1", "Section A"],
        heading_level=2,
        page_number=5,
        source_line_start=10,
        source_line_end=12,
    )
    with patch("chunk_embed.split.split_text", return_value=["Sentence one.", "Sentence two."]):
        result = split_chunks([chunk], "en")

    for r in result:
        assert r.heading_context == ["Chapter 1", "Section A"]
        assert r.heading_level == 2
        assert r.page_number == 5
        assert r.source_line_start == 10
        assert r.source_line_end == 12
        assert r.chunk_type == "paragraph"


def test_split_chunks_single_sentence():
    chunk = make_chunk("Only one sentence.", chunk_type="paragraph")
    with patch("chunk_embed.split.split_text", return_value=["Only one sentence."]):
        result = split_chunks([chunk], "en")
    assert len(result) == 1
    assert result[0].text == "Only one sentence."


def test_split_chunks_mixed_types():
    para = make_chunk("A. B.", chunk_type="paragraph")
    code = make_chunk("print(1)", chunk_type="code_block")
    heading = make_chunk("Title here. Subtitle.", chunk_type="heading")

    def fake_split(text, lang):
        if text == "A. B.":
            return ["A.", "B."]
        if text == "Title here. Subtitle.":
            return ["Title here.", "Subtitle."]
        return [text]

    with patch("chunk_embed.split.split_text", side_effect=fake_split):
        result = split_chunks([para, code, heading], "en")

    assert len(result) == 5  # 2 + 1 + 2
    assert result[0].text == "A."
    assert result[1].text == "B."
    assert result[2].text == "print(1)"
    assert result[3].text == "Title here."
    assert result[4].text == "Subtitle."


def test_split_chunks_empty_list():
    result = split_chunks([], "en")
    assert result == []


# --- integration test (requires sentenza binary) ---


@pytest.mark.sentenza
def test_split_text_real_sentenza():
    result = split_text("The quick brown fox jumped. The lazy dog slept.", "en")
    assert len(result) == 2
    assert "fox" in result[0]
    assert "dog" in result[1]
