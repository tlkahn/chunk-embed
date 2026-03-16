import subprocess
from unittest.mock import patch, MagicMock

import pytest

from chunk_embed.split import SPLITTABLE_TYPES, split_text, split_chunks, detect_lang
from tests.conftest import make_chunk


# --- SPLITTABLE_TYPES ---


def test_splittable_types_contains_expected():
    for t in ("paragraph", "block_quote", "definition_item", "theorem", "list_item", "heading"):
        assert t in SPLITTABLE_TYPES


def test_splittable_types_excludes_non_splittable():
    for t in ("code_block", "math_block", "table"):
        assert t not in SPLITTABLE_TYPES


# --- detect_lang (mocked) ---


def test_detect_lang_none_fallback():
    mock_detector = MagicMock()
    mock_detector.detect_language_of.return_value = None
    with patch("chunk_embed.split._get_detector", return_value=mock_detector):
        assert detect_lang("???") == "en"


def test_detect_lang_passes_through_iso_code():
    """detect_lang returns the ISO 639-1 code unchanged — no remapping."""
    from lingua import Language
    for lang, expected in [
        (Language.ENGLISH, "en"),
        (Language.GERMAN, "de"),
        (Language.FRENCH, "fr"),
        (Language.HINDI, "hi"),
        (Language.MARATHI, "mr"),
    ]:
        mock_detector = MagicMock()
        mock_detector.detect_language_of.return_value = lang
        with patch("chunk_embed.split._get_detector", return_value=mock_detector):
            assert detect_lang("x") == expected, f"expected {expected} for {lang}"


# --- detect_lang smoke tests (real lingua detector, no mocks) ---


@pytest.mark.parametrize("text, expected", [
    ("The quick brown fox jumps over the lazy dog.", "en"),
    ("Der schnelle braune Fuchs springt über den faulen Hund.", "de"),
    ("Le rapide renard brun saute par-dessus le chien paresseux.", "fr"),
    # Devanagari Sanskrit is detected as an Indic language (hi/mr);
    # sentencex handles dandas for any language code, so no remapping needed.
    ("धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः", {"hi", "mr", "sa"}),
])
def test_detect_lang_smoke(text, expected):
    result = detect_lang(text)
    if isinstance(expected, set):
        assert result in expected, f"got {result!r}, expected one of {expected}"
    else:
        assert result == expected


# --- split_text ---


def test_split_text_calls_sentenza_with_correct_args():
    with (
        patch("chunk_embed.split.detect_lang", return_value="en"),
        patch("chunk_embed.split.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="First sentence.\nSecond sentence.\n")
        split_text("First sentence. Second sentence.")
        mock_run.assert_called_once_with(
            ["sentenza", "en"],
            input="First sentence. Second sentence.",
            capture_output=True,
            text=True,
            check=True,
        )


def test_split_text_returns_sentences_from_stdout():
    with (
        patch("chunk_embed.split.detect_lang", return_value="en"),
        patch("chunk_embed.split.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="Hello.\nWorld.\n")
        result = split_text("Hello. World.")
        assert result == ["Hello.", "World."]


def test_split_text_filters_empty_lines():
    with (
        patch("chunk_embed.split.detect_lang", return_value="en"),
        patch("chunk_embed.split.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="Hello.\n\nWorld.\n\n")
        result = split_text("Hello. World.")
        assert result == ["Hello.", "World."]


def test_split_text_empty_input_returns_without_subprocess():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        result = split_text("")
        assert result == [""]
        mock_run.assert_not_called()


def test_split_text_whitespace_only_returns_without_subprocess():
    with patch("chunk_embed.split.subprocess.run") as mock_run:
        result = split_text("   \n  ")
        assert result == ["   \n  "]
        mock_run.assert_not_called()


def test_split_text_file_not_found_falls_back():
    with (
        patch("chunk_embed.split.detect_lang", return_value="en"),
        patch("chunk_embed.split.subprocess.run", side_effect=FileNotFoundError),
    ):
        result = split_text("Some text.")
        assert result == ["Some text."]


def test_split_text_called_process_error_falls_back():
    with (
        patch("chunk_embed.split.detect_lang", return_value="en"),
        patch(
            "chunk_embed.split.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "sentenza"),
        ),
    ):
        result = split_text("Some text.")
        assert result == ["Some text."]


def test_split_text_uses_detected_lang():
    with (
        patch("chunk_embed.split.detect_lang", return_value="de") as mock_detect,
        patch("chunk_embed.split.subprocess.run") as mock_run,
    ):
        mock_run.return_value = MagicMock(stdout="Satz eins.\nSatz zwei.\n")
        split_text("Satz eins. Satz zwei.")
        mock_detect.assert_called_once_with("Satz eins. Satz zwei.")
        assert mock_run.call_args[0][0] == ["sentenza", "de"]


# --- split_chunks ---


def test_split_chunks_expands_paragraph():
    chunk = make_chunk("First. Second.", chunk_type="paragraph")
    with (
        patch("chunk_embed.split._get_detector"),
        patch("chunk_embed.split.split_text", return_value=["First.", "Second."]),
    ):
        result = split_chunks([chunk])
    assert len(result) == 2
    assert result[0].text == "First."
    assert result[1].text == "Second."


def test_split_chunks_skips_code_block():
    chunk = make_chunk("x = 1", chunk_type="code_block")
    with (
        patch("chunk_embed.split._get_detector"),
        patch("chunk_embed.split.split_text") as mock_split,
    ):
        result = split_chunks([chunk])
    mock_split.assert_not_called()
    assert result == [chunk]


def test_split_chunks_skips_math_block():
    chunk = make_chunk("E = mc^2", chunk_type="math_block")
    with (
        patch("chunk_embed.split._get_detector"),
        patch("chunk_embed.split.split_text") as mock_split,
    ):
        result = split_chunks([chunk])
    mock_split.assert_not_called()
    assert result == [chunk]


def test_split_chunks_skips_table():
    chunk = make_chunk("| a | b |", chunk_type="table")
    with (
        patch("chunk_embed.split._get_detector"),
        patch("chunk_embed.split.split_text") as mock_split,
    ):
        result = split_chunks([chunk])
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
    with (
        patch("chunk_embed.split._get_detector"),
        patch("chunk_embed.split.split_text", return_value=["Sentence one.", "Sentence two."]),
    ):
        result = split_chunks([chunk])

    for r in result:
        assert r.heading_context == ["Chapter 1", "Section A"]
        assert r.heading_level == 2
        assert r.page_number == 5
        assert r.source_line_start == 10
        assert r.source_line_end == 12
        assert r.chunk_type == "paragraph"


def test_split_chunks_single_sentence():
    chunk = make_chunk("Only one sentence.", chunk_type="paragraph")
    with (
        patch("chunk_embed.split._get_detector"),
        patch("chunk_embed.split.split_text", return_value=["Only one sentence."]),
    ):
        result = split_chunks([chunk])
    assert len(result) == 1
    assert result[0].text == "Only one sentence."


def test_split_chunks_mixed_types():
    para = make_chunk("A. B.", chunk_type="paragraph")
    code = make_chunk("print(1)", chunk_type="code_block")
    heading = make_chunk("Title here. Subtitle.", chunk_type="heading")

    def fake_split(text):
        if text == "A. B.":
            return ["A.", "B."]
        if text == "Title here. Subtitle.":
            return ["Title here.", "Subtitle."]
        return [text]

    with (
        patch("chunk_embed.split._get_detector"),
        patch("chunk_embed.split.split_text", side_effect=fake_split),
    ):
        result = split_chunks([para, code, heading])

    assert len(result) == 5  # 2 + 1 + 2
    assert result[0].text == "A."
    assert result[1].text == "B."
    assert result[2].text == "print(1)"
    assert result[3].text == "Title here."
    assert result[4].text == "Subtitle."


def test_split_chunks_empty_list():
    result = split_chunks([])
    assert result == []


# --- integration tests (requires sentenza binary + lingua) ---


@pytest.mark.sentenza
def test_split_text_real_sentenza():
    result = split_text("The quick brown fox jumped. The lazy dog slept.")
    assert len(result) == 2
    assert "fox" in result[0]
    assert "dog" in result[1]


@pytest.mark.sentenza
def test_split_text_devanagari_dandas():
    """Pure Devanagari text splits on single dandas."""
    result = split_text("धर्मक्षेत्रे कुरुक्षेत्रे। समवेता युयुत्सवः।")
    assert len(result) == 2
    assert "धर्मक्षेत्रे" in result[0]
    assert "समवेता" in result[1]


@pytest.mark.sentenza
def test_split_text_devanagari_double_dandas():
    """Pure Devanagari text splits on double dandas (verse separators)."""
    result = split_text("प्रथमः श्लोकः॥ द्वितीयः श्लोकः॥ तृतीयः श्लोकः॥")
    assert len(result) == 3


@pytest.mark.sentenza
def test_split_text_mixed_english_devanagari():
    """Mixed-script text splits on both periods and dandas."""
    result = split_text("This is English. यह हिन्दी है। More English here.")
    assert len(result) == 3
    assert "English" in result[0]
    assert "हिन्दी" in result[1]
    assert "More" in result[2]


@pytest.mark.sentenza
def test_split_text_devanagari_abbreviation():
    """Abbreviations like 'Dr.' must not cause false splits in Indic text."""
    result = split_text("Dr. Kumar ने कहा। वह गया।")
    assert len(result) == 2
    assert "Dr." in result[0]
    assert "वह" in result[1]


@pytest.mark.sentenza
def test_split_text_devanagari_question_mark():
    """Question marks split correctly alongside dandas."""
    result = split_text("किम् एतत्? उत्तरम् अस्ति।")
    assert len(result) == 2
    assert "?" in result[0]
    assert "।" in result[1]


@pytest.mark.sentenza
def test_split_text_iast_romanized_sanskrit():
    """IAST romanized Sanskrit looks like English — splits on periods."""
    result = split_text("dharma-kṣetre kuru-kṣetre. samavetā yuyutsavaḥ.")
    assert len(result) == 2
