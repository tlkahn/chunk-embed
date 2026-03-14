import numpy as np
import pytest

from tests.conftest import MockEmbedder, make_chunk
from chunk_embed.embed import embed_chunks


def test_mock_embedder_returns_correct_shape():
    emb = MockEmbedder()
    result = emb.embed(["hello", "world", "test"])
    assert result.shape == (3, 1024)
    assert result.dtype == np.float32


def test_batch_embed_chunks(mock_embedder):
    chunks = [make_chunk(text=f"chunk {i}") for i in range(100)]
    results = embed_chunks(chunks, mock_embedder, batch_size=32)
    assert len(results) == 100
    assert all(v.shape == (1024,) for v in results)


def test_batch_embed_empty_list(mock_embedder):
    results = embed_chunks([], mock_embedder, batch_size=32)
    assert results == []


def test_batch_embed_single_chunk(mock_embedder):
    chunks = [make_chunk(text="only one")]
    results = embed_chunks(chunks, mock_embedder, batch_size=32)
    assert len(results) == 1
    assert results[0].shape == (1024,)


@pytest.mark.slow
def test_bge_m3_embedder_dimension():
    from chunk_embed.embed import BgeM3Embedder
    emb = BgeM3Embedder()
    assert emb.dimension == 1024


@pytest.mark.slow
def test_bge_m3_embedder_encode():
    from chunk_embed.embed import BgeM3Embedder
    emb = BgeM3Embedder()
    result = emb.embed(["Hello world", "Test sentence"])
    assert result.shape == (2, 1024)
    assert np.all(np.isfinite(result))
    norms = np.linalg.norm(result, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
