from __future__ import annotations

from typing import Protocol

import numpy as np

from chunk_embed.models import ChunkData


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


class BgeM3Embedder:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer("./local_bge_m3/BAAI/bge-m3")

    @property
    def dimension(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True)


def embed_chunks(
    chunks: list[ChunkData],
    embedder: Embedder,
    batch_size: int = 32,
) -> list[np.ndarray]:
    if not chunks:
        return []

    texts = [c.text for c in chunks]
    results: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.embed(batch)
        for vec in batch_embeddings:
            results.append(vec)

    return results
