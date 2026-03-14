import numpy as np
import pytest

from chunk_embed.models import ChunkData


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
