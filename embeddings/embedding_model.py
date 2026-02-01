"""Abstract base class for embedding models."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import os
import numpy as np
import torch


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    def __init__(self, device: str):
        """Initialize with device resolution."""
        self._device = self._resolve_device(device)

    @abstractmethod
    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            **kwargs: Additional model-specific arguments

        Returns:
            Array of embeddings with shape (len(texts), embedding_dim)
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up model resources."""
        pass

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        try:
            self.cleanup()
        except Exception:
            pass

    def _resolve_device(self, requested: Optional[str]) -> str:
        """Resolve device string.

        Note: MPS (Apple Silicon GPU) has known instability issues with
        transformer attention operations (scaled_dot_product_attention crashes).
        By default, we use CPU for embeddings even on Apple Silicon for stability.
        Set CODE_SEARCH_EMBEDDING_DEVICE=mps to explicitly use MPS if desired.
        """
        # Check for explicit embedding device override
        env_device = os.getenv("CODE_SEARCH_EMBEDDING_DEVICE", "").strip().lower()
        if env_device:
            req = env_device
        else:
            req = (requested or "auto").lower()

        if req in ("auto", "none", ""):
            if torch.cuda.is_available():
                return "cuda"
            # NOTE: Intentionally NOT auto-selecting MPS for embeddings.
            # PyTorch MPS has instability issues with transformer attention
            # (scaled_dot_product_attention) that cause Metal assertion failures.
            # Users can explicitly set CODE_SEARCH_EMBEDDING_DEVICE=mps to try it.
            return "cpu"
        if req.startswith("cuda"):
            return "cuda" if torch.cuda.is_available() else "cpu"
        if req == "mps":
            try:
                if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
                return "cpu"
            except Exception:
                return "cpu"
        return "cpu"
