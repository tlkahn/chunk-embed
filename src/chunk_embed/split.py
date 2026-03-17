from __future__ import annotations

import logging
import subprocess
from dataclasses import replace

from chunk_embed.models import ChunkData, TEXTUAL_TYPES

logger = logging.getLogger(__name__)

SPLITTABLE_TYPES = TEXTUAL_TYPES

_detector = None


def _get_detector():
    """Return a lazily-initialised lingua LanguageDetector singleton."""
    global _detector
    if _detector is None:
        from lingua import LanguageDetectorBuilder
        _detector = LanguageDetectorBuilder.from_all_languages().build()
    return _detector


def detect_lang(text: str) -> str:
    """Detect the language of *text* and return an ISO 639-1 code.

    The detected code is passed straight to sentenza/sentencex, which
    handles dandas (।॥) for all language codes — no Devanagari remapping
    needed.

    Returns ``"en"`` when detection fails.
    """
    detected = _get_detector().detect_language_of(text)
    if detected is None:
        return "en"
    return detected.iso_code_639_1.name.lower()


def split_text(text: str) -> list[str]:
    """Split text into sentences using the sentenza CLI."""
    if not text.strip():
        return [text]

    lang = detect_lang(text)

    try:
        result = subprocess.run(
            ["sentenza", lang],
            input=text,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError:
        logger.warning("sentenza binary not found; returning text unsplit")
        return [text]
    except subprocess.CalledProcessError as e:
        logger.warning("sentenza failed (exit %d); returning text unsplit", e.returncode)
        return [text]

    sentences = [line for line in result.stdout.splitlines() if line]
    return sentences if sentences else [text]


def split_chunks(chunks: list[ChunkData]) -> list[ChunkData]:
    """Split splittable chunks into per-sentence ChunkData objects."""
    if chunks:
        _get_detector()  # eagerly initialise detector
    out: list[ChunkData] = []
    for chunk in chunks:
        if chunk.chunk_type not in SPLITTABLE_TYPES:
            out.append(chunk)
            continue
        sentences = split_text(chunk.text)
        for sentence in sentences:
            out.append(replace(chunk, text=sentence))
    return out
