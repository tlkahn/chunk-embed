from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm import tqdm

from chunk_embed.models import ChunkData


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray: ...

    @property
    def dimension(self) -> int: ...


_MODEL_DIR = Path(__file__).resolve().parents[2] / "local_bge_m3" / "BAAI" / "bge-m3"


class BgeM3Embedder:
    def __init__(self) -> None:
        import click

        click.echo("Loading sentence-transformers…")
        from sentence_transformers import SentenceTransformer

        if _MODEL_DIR.is_dir():
            click.echo(f"Loading local model: {_MODEL_DIR}")
            self._model = SentenceTransformer(str(_MODEL_DIR))
        else:
            click.echo("Local model not found, downloading BAAI/bge-m3 from HuggingFace")
            self._model = SentenceTransformer("BAAI/bge-m3")

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

    pbar = tqdm(total=len(texts), desc="Embedding", unit="chunk")
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = embedder.embed(batch)
        for vec in batch_embeddings:
            results.append(vec)
        pbar.update(len(batch))
    pbar.close()

    return results
