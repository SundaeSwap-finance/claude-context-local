"""SentenceTransformer embedding model implementation."""

from typing import Optional, Dict, Any
from pathlib import Path
from functools import cached_property
import os
import logging
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from embeddings.embedding_model import EmbeddingModel


class SentenceTransformerModel(EmbeddingModel):
    """SentenceTransformer embedding model with caching and device management."""

    def __init__(
        self,
        model_name: str,
        cache_dir: Optional[str] = None,
        device: str = "auto"
    ):
        """Initialize SentenceTransformerModel.

        Args:
            model_name: Name of the model to load
            cache_dir: Directory to cache the model
            device: Device to load model on
        """
        super().__init__(device=device)
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model_loaded = False
        self._logger = logging.getLogger(__name__)

    @cached_property
    def model(self):
        """Load and cache the SentenceTransformer model."""
        self._logger.info(f"Loading model: {self.model_name}")

        # If the model appears to be cached locally, enable offline mode
        local_model_dir = None
        try:
            if self._is_model_cached():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                self._logger.info("Model cache detected. Enabling offline mode for faster startup.")
                local_model_dir = self._find_local_model_dir()
                if local_model_dir:
                    self._logger.info(f"Loading model from local cache path: {local_model_dir}")
        except Exception as e:
            self._logger.debug(f"Offline mode detection skipped: {e}")

        try:
            model_source = str(local_model_dir) if local_model_dir else self.model_name
            model = SentenceTransformer(
                model_source,
                cache_folder=self.cache_dir,
                device=self._device,
                trust_remote_code=True
            )
            self._logger.info(f"Model loaded successfully on device: {model.device}")
            self._model_loaded = True
            return model
        except Exception as e:
            self._logger.error(f"Failed to load model: {e}")
            raise

    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode texts using SentenceTransformer.

        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments passed to SentenceTransformer.encode()

        Returns:
            Array of embeddings
        """
        return self.model.encode(texts, **kwargs)

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self._model_loaded:
            return {"status": "not_loaded"}

        return {
            "model_name": self.model_name,
            "embedding_dimension": self.get_embedding_dimension(),
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "device": str(self.model.device),
            "status": "loaded"
        }

    def cleanup(self):
        """Clean up model resources."""
        if not self._model_loaded:
            return

        try:
            model = self.model
            model.to('cpu')

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del model
            self._logger.info("Model cleaned up and memory freed")
        except Exception as e:
            self._logger.warning(f"Error during model cleanup: {e}")

    def _has_model_weights(self, directory: Path) -> bool:
        """Check if directory contains model weights."""
        weights_files = ['model.safetensors', 'pytorch_model.bin']
        for weights_file in weights_files:
            weights_path = directory / weights_file
            # Check if file exists and is not empty (symlinks are ok)
            if weights_path.exists():
                try:
                    # Resolve symlinks and check file size
                    resolved = weights_path.resolve()
                    if resolved.exists() and resolved.stat().st_size > 0:
                        return True
                except Exception:
                    pass
        return False

    def _is_model_cached(self) -> bool:
        """Check if model is cached locally with weights."""
        if not self.cache_dir:
            return False
        try:
            model_key = self.model_name.split('/')[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return False
            for path in cache_root.rglob('config_sentence_transformers.json'):
                parent = path.parent
                if model_key in str(parent).lower() and self._has_model_weights(parent):
                    return True
        except Exception:
            return False
        return False

    def _find_local_model_dir(self) -> Optional[Path]:
        """Locate the cached model directory with weights."""
        if not self.cache_dir:
            return None
        try:
            model_key = self.model_name.split('/')[-1].lower()
            cache_root = Path(self.cache_dir)
            if not cache_root.exists():
                return None
            for path in cache_root.rglob('config_sentence_transformers.json'):
                parent = path.parent
                if model_key in str(parent).lower() and self._has_model_weights(parent):
                    return parent
            return None
        except Exception:
            return None
