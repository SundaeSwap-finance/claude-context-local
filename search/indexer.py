"""Vector index management with FAISS and metadata storage."""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlitedict import SqliteDict
from embeddings.embedder import EmbeddingResult
from chunking.code_chunk import CodeChunk
from search.vector_backends import FaissVectorIndex, MpsVectorIndex, resolve_vector_backend


class CodeIndexManager:
    """Manages vector index backend and metadata storage for code chunks."""
    
    def __init__(self, storage_dir: str, backend: Optional[str] = None):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.index_path = self.storage_dir / "code.index"
        self.metadata_path = self.storage_dir / "metadata.db" 
        self.chunk_id_path = self.storage_dir / "chunk_ids.pkl"
        self.stats_path = self.storage_dir / "stats.json"
        self.index_meta_path = self.storage_dir / "index_meta.json"
        
        # Initialize components
        self._index = None
        self._metadata_db = None
        self._chunk_ids = []
        self._logger = logging.getLogger(__name__)
        self._backend_name, self._backend_explicit = resolve_vector_backend(backend)
        
    @property
    def index(self):
        """Lazy loading of FAISS index."""
        if self._index is None:
            self._load_index()
        return self._index
    
    @property
    def metadata_db(self):
        """Lazy loading of metadata database."""
        if self._metadata_db is None:
            self._metadata_db = SqliteDict(
                str(self.metadata_path), 
                autocommit=False,
                journal_mode="WAL"
            )
        return self._metadata_db
    
    def _load_index(self):
        """Load existing vector index or create new one."""
        if self.index_path.exists():
            self._logger.info(f"Loading existing index from {self.index_path}")
            preferred_backend = self._resolve_backend_for_existing_index()
            self._index = self._try_load_index(preferred_backend)

            if self._index is None:
                fallback_backend = "faiss" if preferred_backend == "mps" else "mps"
                self._logger.warning(
                    f"Failed to load index with backend '{preferred_backend}'. "
                    f"Attempting fallback backend '{fallback_backend}'."
                )
                self._index = self._try_load_index(fallback_backend)
                if self._index is not None:
                    self._backend_name = fallback_backend
            
            # Load chunk IDs
            if self.chunk_id_path.exists():
                with open(self.chunk_id_path, 'rb') as f:
                    self._chunk_ids = pickle.load(f)
            if self._index is None:
                self._logger.warning("Index file could not be loaded; resetting chunk IDs.")
                self._chunk_ids = []
        else:
            self._logger.info("Creating new index")
            # Create a new index - we'll initialize it when we get the first embedding
            self._index = None
            self._chunk_ids = []
    
    def create_index(self, embedding_dimension: int, index_type: str = "flat"):
        """Create a new vector index."""
        if self._backend_name == "mps":
            try:
                self._index = MpsVectorIndex.create(embedding_dimension, device="mps")
                self._logger.info(f"Created MPS index with dimension {embedding_dimension}")
            except Exception as exc:
                self._logger.warning(f"Failed to create MPS index, falling back to FAISS: {exc}")
                self._backend_name = "faiss"
                self._index = FaissVectorIndex.create(embedding_dimension, index_type=index_type)
                self._logger.info(f"Created {index_type} FAISS index with dimension {embedding_dimension}")
        else:
            self._index = FaissVectorIndex.create(embedding_dimension, index_type=index_type)
            self._logger.info(f"Created {index_type} FAISS index with dimension {embedding_dimension}")
        self._write_index_meta()
    
    def add_embeddings(self, embedding_results: List[EmbeddingResult]) -> None:
        """Add embeddings to the index and metadata to the database."""
        if not embedding_results:
            return
        
        # Initialize index if needed
        if self._index is None:
            embedding_dim = embedding_results[0].embedding.shape[0]
            # Default to flat index for better recall - only use IVF for very large datasets
            index_type = "ivf" if len(embedding_results) > 10000 else "flat"
            self.create_index(embedding_dim, index_type)
        
        # Prepare embeddings and metadata
        embeddings = np.array([result.embedding for result in embedding_results], dtype=np.float32)
        
        # Normalize embeddings for cosine similarity
        embeddings = self._normalize_embeddings(embeddings)
        
        # Add to index
        start_id = len(self._chunk_ids)
        self._index.add(embeddings)
        
        # Store metadata and update chunk IDs
        for i, result in enumerate(embedding_results):
            chunk_id = result.chunk_id
            self._chunk_ids.append(chunk_id)
            
            # Store in metadata database
            self.metadata_db[chunk_id] = {
                'index_id': start_id + i,
                'metadata': result.metadata
            }
        
        self._logger.info(f"Added {len(embedding_results)} embeddings to index")
        
        # Commit metadata in a single transaction for performance
        try:
            self.metadata_db.commit()
        except Exception:
            # If commit is unavailable for some reason, continue without failing
            pass
        
        # Update statistics
        self._update_stats()

    def _gpu_is_available(self) -> bool:
        """Deprecated: retained for compatibility. Use vector backend selection."""
        return False

    def _maybe_move_index_to_gpu(self) -> None:
        """Deprecated: retained for compatibility. GPU handled in FAISS backend."""
        return
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar code chunks."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Index manager search called with k={k}, filters={filters}")
        
        # Use property to trigger lazy loading
        index = self.index
        if index is None or index.ntotal == 0:
            logger.warning(f"Index is empty or None. Index: {index}, ntotal: {index.ntotal if index else 'N/A'}")
            return []
        
        logger.info(f"Index has {index.ntotal} total vectors")
        
        # Normalize query embedding
        query_embedding = self._normalize_query(query_embedding)
        
        # Search in index
        search_k = min(k * 3, index.ntotal)  # Get more results for filtering
        similarities, indices = index.search(query_embedding, search_k)
        
        results = []
        for i, (similarity, index_id) in enumerate(zip(similarities[0], indices[0])):
            if index_id == -1:  # No more results
                break
            
            chunk_id = self._chunk_ids[index_id]
            metadata_entry = self.metadata_db.get(chunk_id)
            
            if metadata_entry is None:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Apply filters
            if filters and not self._matches_filters(metadata, filters):
                continue
            
            results.append((chunk_id, float(similarity), metadata))
            
            if len(results) >= k:
                break
        
        return results
    
    def _matches_filters(self, metadata: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if metadata matches the provided filters."""
        for key, value in filters.items():
            if key == 'file_pattern':
                # Pattern matching for file paths
                if not any(pattern in metadata.get('relative_path', '') for pattern in value):
                    return False
            elif key == 'chunk_type':
                # Exact match for chunk type
                if metadata.get('chunk_type') != value:
                    return False
            elif key == 'tags':
                # Tag intersection
                chunk_tags = set(metadata.get('tags', []))
                required_tags = set(value if isinstance(value, list) else [value])
                if not required_tags.intersection(chunk_tags):
                    return False
            elif key == 'folder_structure':
                # Check if any of the required folders are in the path
                chunk_folders = set(metadata.get('folder_structure', []))
                required_folders = set(value if isinstance(value, list) else [value])
                if not required_folders.intersection(chunk_folders):
                    return False
            elif key in metadata:
                # Direct metadata comparison
                if metadata[key] != value:
                    return False
        
        return True
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve chunk metadata by ID."""
        metadata_entry = self.metadata_db.get(chunk_id)
        return metadata_entry['metadata'] if metadata_entry else None
    
    def get_similar_chunks(self, chunk_id: str, k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Find chunks similar to a given chunk."""
        metadata_entry = self.metadata_db.get(chunk_id)
        if not metadata_entry:
            return []
        
        index_id = metadata_entry['index_id']
        if self._index is None or index_id >= self._index.ntotal:
            return []
        
        # Get the embedding for this chunk
        embedding = self._index.reconstruct(index_id)
        
        # Search for similar chunks (excluding the original)
        results = self.search(embedding, k + 1)
        
        # Filter out the original chunk
        return [(cid, sim, meta) for cid, sim, meta in results if cid != chunk_id][:k]
    
    def remove_file_chunks(self, file_path: str, project_name: Optional[str] = None) -> int:
        """Remove all chunks from a specific file.
        
        Args:
            file_path: Path to the file (relative or absolute)
            project_name: Optional project name filter
            
        Returns:
            Number of chunks removed
        """
        chunks_to_remove = []
        
        # Find chunks to remove
        for chunk_id in self._chunk_ids:
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Check if this chunk belongs to the file
            chunk_file = metadata.get('file_path') or metadata.get('relative_path')
            if not chunk_file:
                continue
            
            # Check if paths match (handle both relative and absolute)
            if file_path in chunk_file or chunk_file in file_path:
                # Check project name if provided
                if project_name and metadata.get('project_name') != project_name:
                    continue
                chunks_to_remove.append(chunk_id)
        
        # Remove chunks from metadata
        for chunk_id in chunks_to_remove:
            del self.metadata_db[chunk_id]
        
        # Note: We don't remove from FAISS index directly as it's complex
        # Instead, we'll rebuild the index periodically or on demand
        
        self._logger.info(f"Removed {len(chunks_to_remove)} chunks from {file_path}")
        
        # Commit removals in batch
        try:
            self.metadata_db.commit()
        except Exception:
            pass
        return len(chunks_to_remove)
    
    def save_index(self):
        """Save the vector index and chunk IDs to disk."""
        if self._index is not None:
            try:
                self._index.save(self.index_path)
                self._logger.info(f"Saved index to {self.index_path}")
            except Exception as e:
                self._logger.error(f"Failed to save index: {e}")
        
        # Save chunk IDs
        with open(self.chunk_id_path, 'wb') as f:
            pickle.dump(self._chunk_ids, f)
        self._write_index_meta()
        
        self._update_stats()
    
    def _update_stats(self):
        """Update index statistics."""
        stats = {
            'total_chunks': len(self._chunk_ids),
            'index_size': self._index.ntotal if self._index else 0,
            'embedding_dimension': self._index.d if self._index else 0,
            'index_type': self._index.index_type if self._index else 'None',
            'vector_backend': self._backend_name if self._index else 'None'
        }
        
        # Add file and folder statistics
        file_counts = {}
        folder_counts = {}
        chunk_type_counts = {}
        tag_counts = {}
        
        for chunk_id in self._chunk_ids:
            metadata_entry = self.metadata_db.get(chunk_id)
            if not metadata_entry:
                continue
            
            metadata = metadata_entry['metadata']
            
            # Count by file
            file_path = metadata.get('relative_path', 'unknown')
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
            
            # Count by folder
            for folder in metadata.get('folder_structure', []):
                folder_counts[folder] = folder_counts.get(folder, 0) + 1
            
            # Count by chunk type
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1
            
            # Count by tags
            for tag in metadata.get('tags', []):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        stats.update({
            'files_indexed': len(file_counts),
            'top_folders': dict(sorted(folder_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'chunk_types': chunk_type_counts,
            'top_tags': dict(sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20])
        })
        
        # Save stats
        with open(self.stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        if self.stats_path.exists():
            with open(self.stats_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'total_chunks': 0,
                'index_size': 0,
                'embedding_dimension': 0,
                'files_indexed': 0
            }
    
    def get_index_size(self) -> int:
        """Get the number of chunks in the index."""
        return len(self._chunk_ids)
    
    def clear_index(self):
        """Clear the entire index and metadata."""
        # Close database connection
        self.close()
        
        # Remove files
        for file_path in [self.index_path, self.metadata_path, self.chunk_id_path, self.stats_path, self.index_meta_path]:
            if file_path.exists():
                file_path.unlink()
        
        # Reset in-memory state
        self._index = None
        self._chunk_ids = []
        
        self._logger.info("Index cleared")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        self.close()

    def close(self):
        """Close open resources."""
        if self._metadata_db is not None:
            try:
                self._metadata_db.close()
            except Exception:
                pass
            self._metadata_db = None

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (embeddings / norms).astype(np.float32, copy=False)

    def _normalize_query(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        norms = np.linalg.norm(query, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (query / norms).astype(np.float32, copy=False)

    def _resolve_backend_for_existing_index(self) -> str:
        if not self._backend_explicit:
            meta_backend = self._read_index_meta_backend()
            if meta_backend:
                return meta_backend
        return self._backend_name

    def _read_index_meta_backend(self) -> Optional[str]:
        if not self.index_meta_path.exists():
            return None
        try:
            with open(self.index_meta_path, "r") as handle:
                data = json.load(handle)
            backend = data.get("backend")
            return backend if backend in ("faiss", "mps") else None
        except Exception:
            return None

    def _write_index_meta(self) -> None:
        if not self._index:
            return
        meta = {
            "backend": self._backend_name,
            "index_type": self._index.index_type,
            "embedding_dimension": self._index.d
        }
        try:
            with open(self.index_meta_path, "w") as handle:
                json.dump(meta, handle, indent=2)
        except Exception as exc:
            self._logger.warning(f"Failed to write index metadata: {exc}")

    def _try_load_index(self, backend_name: str):
        try:
            if backend_name == "mps":
                self._backend_name = "mps"
                return MpsVectorIndex.load(self.index_path, device="mps")
            self._backend_name = "faiss"
            return FaissVectorIndex.load(self.index_path)
        except Exception as exc:
            self._logger.warning(f"Failed to load {backend_name} index: {exc}")
            return None
