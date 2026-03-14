import numpy as np
import psycopg
import pytest
from pgvector.psycopg import register_vector

from chunk_embed.models import ChunkData
from chunk_embed.store import ensure_schema


TEST_DB_URL = "postgresql://localhost/chunk_embed_test"


class MockEmbedder:
    @property
    def dimension(self) -> int:
        return 1024

    def embed(self, texts: list[str]) -> np.ndarray:
        rng = np.random.default_rng(42)
        vecs = rng.standard_normal((len(texts), 1024)).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / norms


@pytest.fixture
def mock_embedder():
    return MockEmbedder()


def make_chunk(text: str = "hello", chunk_type: str = "paragraph", **kwargs) -> ChunkData:
    defaults = dict(
        heading_context=[],
        heading_level=None,
        page_number=None,
        source_line_start=1,
        source_line_end=1,
    )
    defaults.update(kwargs)
    return ChunkData(text=text, chunk_type=chunk_type, **defaults)


def make_embedding(dim: int = 1024, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


@pytest.fixture
def conn():
    with psycopg.connect(TEST_DB_URL, autocommit=False) as c:
        register_vector(c)
        ensure_schema(c)
        yield c
        c.rollback()
