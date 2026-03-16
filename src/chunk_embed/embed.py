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
    results: list[np.ndarray] = [np.empty(0)] * total

    # Sort indices by text length so similar-sized texts are batched together.
    # This minimises padding waste — a single long text no longer forces every
    # other text in the batch to be padded to its length.
    sorted_indices = sorted(range(total), key=lambda i: len(texts[i]))

    use_tqdm = on_progress is None
    pbar = tqdm(total=total, desc="Embedding", unit="chunk") if use_tqdm else None
    done = 0
    for batch_start in range(0, total, batch_size):
        batch_indices = sorted_indices[batch_start : batch_start + batch_size]
        batch_texts = [texts[i] for i in batch_indices]
        batch_embeddings = embedder.embed(batch_texts)
        for slot, vec in zip(batch_indices, batch_embeddings):
            results[slot] = vec.copy()
        done += len(batch_indices)
        if pbar is not None:
            pbar.update(len(batch_indices))
        if on_progress is not None:
            on_progress(done, total)
    if pbar is not None:
        pbar.close()

    return results
