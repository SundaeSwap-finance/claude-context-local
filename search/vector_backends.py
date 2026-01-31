"""Vector index backends (FAISS CPU/GPU and MPS brute-force)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple
import logging
import os
import platform

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mps_is_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def resolve_vector_backend(requested: Optional[str]) -> Tuple[str, bool]:
    """Resolve backend name and whether it was explicitly requested."""
    req = (requested or os.getenv("CODE_SEARCH_VECTOR_BACKEND") or "auto").strip().lower()
    explicit = req not in ("", "auto")

    if req in ("mps", "metal"):
        if _mps_is_available():
            return "mps", explicit
        logger.warning("MPS requested but not available. Falling back to FAISS.")
        return "faiss", explicit
    if req in ("faiss", "cpu"):
        return "faiss", explicit
    if req in ("auto", ""):
        if _is_apple_silicon() and _mps_is_available():
            return "mps", explicit
        return "faiss", explicit

    logger.warning(f"Unknown vector backend '{req}'. Falling back to FAISS.")
    return "faiss", explicit


class VectorIndex(ABC):
    """Abstract vector index backend interface."""

    @property
    @abstractmethod
    def ntotal(self) -> int:
        pass

    @property
    @abstractmethod
    def d(self) -> int:
        pass

    @property
    @abstractmethod
    def index_type(self) -> str:
        pass

    @abstractmethod
    def add(self, embeddings: np.ndarray) -> None:
        pass

    @abstractmethod
    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def reconstruct(self, index_id: int) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass


class FaissVectorIndex(VectorIndex):
    """FAISS backend with optional CUDA acceleration."""

    def __init__(self, index):
        self._index = index
        self._on_gpu = False
        self._maybe_move_to_gpu()

    @classmethod
    def create(cls, embedding_dim: int, index_type: str = "flat") -> "FaissVectorIndex":
        import faiss

        if index_type == "flat":
            index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(embedding_dim)
            n_centroids = min(100, max(10, embedding_dim // 8))
            index = faiss.IndexIVFFlat(quantizer, embedding_dim, n_centroids)
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        return cls(index)

    @classmethod
    def load(cls, path: Path) -> "FaissVectorIndex":
        import faiss

        index = faiss.read_index(str(path))
        return cls(index)

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    @property
    def d(self) -> int:
        return self._index.d

    @property
    def index_type(self) -> str:
        suffix = "+gpu" if self._on_gpu else ""
        return f"{type(self._index).__name__}{suffix}"

    def add(self, embeddings: np.ndarray) -> None:
        if hasattr(self._index, "is_trained") and not self._index.is_trained:
            self._index.train(embeddings)
        self._index.add(embeddings)

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self._index.search(query, k)

    def reconstruct(self, index_id: int) -> np.ndarray:
        return self._index.reconstruct(index_id)

    def save(self, path: Path) -> None:
        import faiss

        index_to_write = self._index
        if self._on_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            index_to_write = faiss.index_gpu_to_cpu(self._index)
        faiss.write_index(index_to_write, str(path))

    def _gpu_is_available(self) -> bool:
        import faiss

        if not hasattr(faiss, "StandardGpuResources"):
            return False
        get_num_gpus = getattr(faiss, "get_num_gpus", None)
        if get_num_gpus is None:
            return False
        return get_num_gpus() > 0

    def _maybe_move_to_gpu(self) -> None:
        if self._on_gpu:
            return
        if not self._gpu_is_available():
            return
        try:
            import faiss

            self._index = faiss.index_cpu_to_all_gpus(self._index)
            self._on_gpu = True
            logger.info("FAISS index moved to GPU(s)")
        except Exception as exc:
            logger.warning(f"Failed to move FAISS index to GPU, continuing on CPU: {exc}")


class MpsVectorIndex(VectorIndex):
    """Brute-force vector search backend using torch on MPS."""

    def __init__(self, embedding_dim: int, device: str = "mps", embeddings: Optional[np.ndarray] = None):
        if device == "mps" and not _mps_is_available():
            raise RuntimeError("MPS requested but not available.")

        self._device = torch.device(device)
        self._embedding_dim = embedding_dim
        self._embeddings_cpu: Optional[np.ndarray] = None
        self._embeddings_torch: Optional[torch.Tensor] = None
        self._chunk_size = max(1, int(os.getenv("CODE_SEARCH_MPS_CHUNK_SIZE", "20000")))

        if embeddings is not None:
            self._embeddings_cpu = np.asarray(embeddings, dtype=np.float32)
            if self._embeddings_cpu.ndim == 2 and self._embeddings_cpu.shape[1] > 0:
                self._embedding_dim = self._embeddings_cpu.shape[1]

    @classmethod
    def create(cls, embedding_dim: int, device: str = "mps") -> "MpsVectorIndex":
        return cls(embedding_dim=embedding_dim, device=device)

    @classmethod
    def load(cls, path: Path, device: str = "mps") -> "MpsVectorIndex":
        with open(path, "rb") as handle:
            data = np.load(handle, allow_pickle=False)
            if "embeddings" not in data.files:
                raise ValueError("Missing embeddings in MPS index file.")
            embeddings = data["embeddings"]
        return cls(embedding_dim=embeddings.shape[1], device=device, embeddings=embeddings)

    @property
    def ntotal(self) -> int:
        return 0 if self._embeddings_cpu is None else int(self._embeddings_cpu.shape[0])

    @property
    def d(self) -> int:
        if self._embeddings_cpu is not None and self._embeddings_cpu.size > 0:
            return int(self._embeddings_cpu.shape[1])
        return int(self._embedding_dim)

    @property
    def index_type(self) -> str:
        return "mps_flat"

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if self._embeddings_cpu is None:
            self._embeddings_cpu = embeddings
        else:
            self._embeddings_cpu = np.concatenate([self._embeddings_cpu, embeddings], axis=0)
        self._embeddings_torch = None

    def search(self, query: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            return np.empty((1, 0), dtype=np.float32), np.empty((1, 0), dtype=np.int64)

        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query[None, :]

        k = min(k, self.ntotal)
        if k == 0:
            return np.empty((query.shape[0], 0), dtype=np.float32), np.empty((query.shape[0], 0), dtype=np.int64)

        with torch.inference_mode():
            query_t = torch.from_numpy(query).to(self._device)
            embeddings_t = self._ensure_embeddings_torch()

            if embeddings_t is None:
                return np.empty((query.shape[0], 0), dtype=np.float32), np.empty((query.shape[0], 0), dtype=np.int64)

            if self.ntotal <= self._chunk_size:
                scores = torch.matmul(query_t, embeddings_t.T)
                top_scores, top_indices = torch.topk(scores, k, dim=1)
                return top_scores.cpu().numpy(), top_indices.cpu().numpy()

            scores_out = np.empty((query.shape[0], k), dtype=np.float32)
            indices_out = np.empty((query.shape[0], k), dtype=np.int64)

            for row_idx in range(query_t.shape[0]):
                best_scores, best_indices = self._search_single(query_t[row_idx], embeddings_t, k)
                scores_out[row_idx] = best_scores.cpu().numpy()
                indices_out[row_idx] = best_indices.cpu().numpy()

            return scores_out, indices_out

    def reconstruct(self, index_id: int) -> np.ndarray:
        if self._embeddings_cpu is None:
            raise ValueError("Index is empty.")
        return self._embeddings_cpu[index_id]

    def save(self, path: Path) -> None:
        embeddings = self._embeddings_cpu
        if embeddings is None:
            embeddings = np.empty((0, self._embedding_dim), dtype=np.float32)
        with open(path, "wb") as handle:
            np.savez(handle, embeddings=embeddings)

    def _ensure_embeddings_torch(self) -> Optional[torch.Tensor]:
        if self._embeddings_cpu is None:
            return None
        if self._embeddings_torch is None or self._embeddings_torch.shape[0] != self._embeddings_cpu.shape[0]:
            self._embeddings_torch = torch.from_numpy(self._embeddings_cpu).to(self._device)
        return self._embeddings_torch

    def _search_single(self, query_vec: torch.Tensor, embeddings_t: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        best_scores = None
        best_indices = None
        n_total = embeddings_t.shape[0]

        for start in range(0, n_total, self._chunk_size):
            end = min(start + self._chunk_size, n_total)
            chunk = embeddings_t[start:end]
            scores = torch.mv(chunk, query_vec)
            local_k = min(k, scores.numel())
            local_scores, local_indices = torch.topk(scores, local_k)
            local_indices = local_indices + start

            if best_scores is None:
                best_scores = local_scores
                best_indices = local_indices
                continue

            merged_scores = torch.cat([best_scores, local_scores])
            merged_indices = torch.cat([best_indices, local_indices])
            top_scores, top_pos = torch.topk(merged_scores, k)
            best_scores = top_scores
            best_indices = merged_indices[top_pos]

        return best_scores, best_indices
