import numpy as np
import pytest

from search import vector_backends
from search.vector_backends import FaissVectorIndex, MpsVectorIndex, resolve_vector_backend


def test_resolve_backend_explicit_faiss(monkeypatch):
    monkeypatch.setenv("CODE_SEARCH_VECTOR_BACKEND", "faiss")
    backend, explicit = resolve_vector_backend(None)
    assert backend == "faiss"
    assert explicit is True


def test_resolve_backend_mps_unavailable(monkeypatch):
    monkeypatch.delenv("CODE_SEARCH_VECTOR_BACKEND", raising=False)
    monkeypatch.setattr(vector_backends, "_mps_is_available", lambda: False)
    backend, explicit = resolve_vector_backend("mps")
    assert backend == "faiss"
    assert explicit is True


@pytest.mark.skipif(not vector_backends._mps_is_available(), reason="MPS not available")
def test_mps_search_matches_faiss_top1():
    try:
        import faiss  # noqa: F401
    except Exception:
        pytest.skip("faiss not installed")

    rng = np.random.default_rng(42)
    embeddings = rng.random((64, 32), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    query = embeddings[0]

    faiss_index = FaissVectorIndex.create(32, index_type="flat")
    faiss_index.add(embeddings)

    mps_index = MpsVectorIndex.create(32, device="mps")
    mps_index.add(embeddings)

    faiss_scores, faiss_indices = faiss_index.search(query.reshape(1, -1), k=5)
    mps_scores, mps_indices = mps_index.search(query.reshape(1, -1), k=5)

    assert int(faiss_indices[0][0]) == 0
    assert int(mps_indices[0][0]) == 0
    assert np.isclose(faiss_scores[0][0], mps_scores[0][0], atol=1e-4)
