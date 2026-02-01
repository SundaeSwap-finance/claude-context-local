# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Claude Embedding Search is an intelligent code search system that uses Google's EmbeddingGemma model and AST-based chunking to provide semantic search capabilities for Python codebases, integrated with Claude Code via MCP (Model Context Protocol).

## Key Commands

### Development Setup

```bash
# Install dependencies
uv sync

# Install in development mode
uv sync --dev
```

### Testing

```bash
# Run all tests
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --unit          # Unit tests only
python tests/run_tests.py --integration   # Integration tests only
python tests/run_tests.py --chunking      # Chunking tests only
python tests/run_tests.py --embeddings    # Embedding tests only
python tests/run_tests.py --search        # Search tests only
python tests/run_tests.py --mcp           # MCP server tests only

# Run tests with coverage
python tests/run_tests.py --coverage

# Run tests with verbose output
python tests/run_tests.py --verbose

# Run specific test files or patterns
python tests/run_tests.py unit/test_chunking.py

# Alternative: Direct pytest usage
python -m pytest                          # All tests
python -m pytest -m "unit"               # Unit tests only
python -m pytest -m "not slow"           # Skip slow tests
python -m pytest tests/unit/test_chunking.py -v  # Single test file
```

### Indexing & Usage

```bash
# Index a Python codebase
./scripts/index_codebase.py /path/to/project

# Index with custom storage location
./scripts/index_codebase.py /path/to/project --storage-dir /custom/location

# Clear existing index and reindex
./scripts/index_codebase.py /path/to/project --clear

# Enable verbose logging
./scripts/index_codebase.py /path/to/project --verbose
```

### MCP Server

```bash
# Run MCP server directly
uv run python mcp_server/server.py

# Add to Claude Code (global)
claude mcp add code-search --scope user -- uv run --directory /full/path/to/claude_embedding_search python mcp_server/server.py

# Add to Claude Code (project-specific)
claude mcp add code-search -- uv run --directory /full/path/to/claude_embedding_search python mcp_server/server.py
```

## Architecture

The codebase is organized into distinct modules with clear separation of concerns:

### Core Components

- **`chunking/`**: AST-based code parsing and chunking
  - `python_ast_chunker.py`: Breaks Python code into semantically meaningful chunks (functions, classes, modules)
  - `multi_language_chunker.py`: Tree-sitter based chunking for JavaScript, TypeScript, Go, Java, Rust, and Svelte
  - Preserves context and relationships between code elements
- **`embeddings/`**: Embedding generation using EmbeddingGemma
  - `embedder.py`: Handles model loading, caching, and batch embedding generation
  - Uses `google/embeddinggemma-300m` model with 768-dimensional embeddings
- **`search/`**: FAISS-based search and indexing
  - `indexer.py`: Manages FAISS indices, metadata storage (SQLite), and index persistence
  - `searcher.py`: Intelligent search with filtering, context-aware results, and similarity search
- **`mcp_server/`**: Claude Code integration via MCP

  - `server.py`: FastMCP server exposing search tools to Claude Code
  - Provides `search_code`, `index_directory`, `find_similar_code`, etc.

- **`merkle/`**: Incremental indexing support
  - `merkle_dag.py`: Merkle tree implementation for efficient change detection
  - `change_detector.py`: Detects file additions, modifications, and deletions
  - `snapshot_manager.py`: Manages snapshots for incremental indexing
- **`search/incremental_indexer.py`**: Orchestrates incremental indexing using Merkle tree change detection

### Storage Structure

Data is stored in `~/.claude_code_search/` (configurable via `CODE_SEARCH_STORAGE`):

```
~/.claude_code_search/
├── models/          # Downloaded EmbeddingGemma models
├── projects/        # Project-specific data
│   └── {project_name}_{hash}/
│       ├── project_info.json  # Project metadata
│       ├── index/             # FAISS indices and metadata
│       │   ├── code.index     # Vector index
│       │   ├── metadata.db    # Chunk metadata (SQLite)
│       │   └── stats.json     # Index statistics
│       └── snapshots/         # Merkle tree snapshots for incremental indexing
```

### Chunking Strategy

The system uses AST parsing to create semantically meaningful chunks:

- Complete functions with docstrings and decorators
- Full classes with methods as separate chunks
- Module-level code blocks and constants
- Rich metadata: file paths, semantic tags, complexity scores, relationships

## Testing Strategy

Tests are organized by component with pytest markers:

- `unit`: Fast, isolated unit tests
- `integration`: End-to-end workflow tests
- `chunking`: AST chunking functionality
- `embeddings`: Model loading and embedding generation
- `search`: Indexing and search functionality
- `mcp`: MCP server integration
- `slow`: Time-intensive tests (excluded by default)

## Development Notes

### Key Dependencies

- `sentence-transformers`: EmbeddingGemma model loading and inference
- `faiss-cpu`: Efficient vector similarity search (CPU backend)
- `fastmcp`: MCP server implementation for Claude Code integration
- `sqlitedict`: Persistent metadata storage
- `tree-sitter` & `tree-sitter-languages`: Multi-language parsing support
- `click`: Command-line interface utilities
- `pytest`: Testing framework with async support

### Performance Considerations

- Model size: ~300MB (EmbeddingGemma-300m)
- Embedding dimension: 768 (FAISS Flat index for small datasets, IVF for large)
- Batch processing: Configurable batch sizes for memory management
- Local processing: All embeddings computed locally, no API calls
- Incremental indexing: Only reprocesses changed files using Merkle tree snapshots

### Environment Variables

- `CODE_SEARCH_STORAGE`: Custom storage directory (default: `~/.claude_code_search`)
- `CODE_SEARCH_VECTOR_BACKEND`: `auto` (default), `mps`, or `faiss` - controls vector search backend
- `CODE_SEARCH_EMBEDDING_DEVICE`: `auto` (default=CPU), `cpu`, `cuda`, or `mps` - controls embedding model device
- `CODE_SEARCH_MPS_CHUNK_SIZE`: Chunk size for MPS search batches (default: `20000`)

**Note on Apple Silicon (MPS):** Embedding generation defaults to CPU even on Apple Silicon because PyTorch's MPS backend has known instability issues with transformer attention operations (`scaled_dot_product_attention`). The vector search backend can still use MPS safely. To explicitly try MPS for embeddings (not recommended), set `CODE_SEARCH_EMBEDDING_DEVICE=mps`.

## Common Tasks

### Adding New Chunk Types

1. Extend `python_ast_chunker.py` to handle new AST node types
2. Update metadata extraction in chunk creation
3. Add corresponding tests in `tests/unit/test_chunking.py`

### Modifying Search Behavior

1. Update `searcher.py` for new filtering/ranking logic
2. Modify MCP server tools in `server.py` if new parameters needed
3. Add integration tests in `tests/integration/test_full_flow.py`

### Testing Changes

Always run the full test suite before commits:

```bash
python tests/run_tests.py --coverage
```

For quick iteration during development:

```bash
python tests/run_tests.py --unit --verbose -x
```

### Multi-Language Support

The system now supports chunking and indexing multiple languages:

- Python (AST-based chunking)
- JavaScript/TypeScript (tree-sitter)
- JSX/TSX (React components)
- Go, Java, Rust (tree-sitter)
- Svelte components
- Markdown (section-based chunking by headers)
