from __future__ import annotations

import logging
import subprocess
from dataclasses import replace

from chunk_embed.models import ChunkData

logger = logging.getLogger(__name__)

SPLITTABLE_TYPES: frozenset[str] = frozenset({
    "paragraph",
    "block_quote",
    "definition_item",
    "theorem",
    "list_item",
    "heading",
})


def split_text(text: str, lang: str) -> list[str]:
    """Split text into sentences using the sentenza CLI."""
    if not text.strip():
        return [text]

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


def split_chunks(chunks: list[ChunkData], lang: str) -> list[ChunkData]:
    """Split splittable chunks into per-sentence ChunkData objects."""
    out: list[ChunkData] = []
    for chunk in chunks:
        if chunk.chunk_type not in SPLITTABLE_TYPES:
            out.append(chunk)
            continue
        sentences = split_text(chunk.text, lang)
        for sentence in sentences:
            out.append(replace(chunk, text=sentence))
    return out
