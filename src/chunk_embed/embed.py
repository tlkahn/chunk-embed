from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm import tqdm

from chunk_embed.models import ChunkData

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


_MODEL_DIR = Path(__file__).resolve().parents[2] / "local_bge_m3" / "BAAI" / "bge-m3"
MODEL_DIR = _MODEL_DIR  # public alias for use by gui.py dependency check


class BgeM3Embedder:
    def __init__(self) -> None:
        logger.info("Loading sentence-transformers…")
        from sentence_transformers import SentenceTransformer

        if _MODEL_DIR.is_dir():
            logger.info("Loading local model: %s", _MODEL_DIR)
            self._model = SentenceTransformer(str(_MODEL_DIR))
        else:
            logger.info("Local model not found, downloading BAAI/bge-m3 from HuggingFace")
            self._model = SentenceTransformer("BAAI/bge-m3")

    @property
    def dimension(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def embed_chunks(
    chunks: list[ChunkData],
    embedder: Embedder,
    batch_size: int = 32,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[np.ndarray]:
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    total = len(texts)
    results: list[np.ndarray] = []

    use_tqdm = on_progress is None
    pbar = tqdm(total=total, desc="Embedding", unit="chunk") if use_tqdm else None
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.embed(batch)
        for vec in batch_embeddings:
            results.append(vec)
        done = min(i + len(batch), total)
        if pbar is not None:
            pbar.update(len(batch))
        if on_progress is not None:
            on_progress(done, total)
    if pbar is not None:
        pbar.close()

    return results
