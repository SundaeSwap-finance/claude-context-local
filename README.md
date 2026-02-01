# Claude Context Local

Forks the official [Claude Context Local](https://github.com/FarhanAliRaza/claude-context-local) repository and switches out EmbeddingGemma (which requires account and registration) for an open-source alternative: [nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1).

## How it works

- Uses an ML model (nomic-embed-text) to convert code into 768-dimensional vectors
- Stores vectors in FAISS for fast similarity search
- Requires indexing the codebase upfront (processes files, generates embeddings, stores them)

## Motivation

- Token usage (not API calls directly) - by returning targeted snippets instead of full files
- Embedding costs - zero, because it runs locally (vs paying for OpenAI embeddings)
- Iterations - semantic search finds relevant code faster, fewer back-and-forth turns

## Requirements

A decent GPU (or CPU as a fallback) with enough RAM (32GB+) to handle model context building. A small repository being indexed with an Apple M1 Max CPU took about 122 seconds as an example. You also need at least 500MB to download and install the embedding.

### Apple Silicon notes

- Embeddings run on PyTorch MPS automatically when available.
- Vector search can use an MPS backend on Apple Silicon (default when available). Override with `CODE_SEARCH_VECTOR_BACKEND=faiss` if you want CPU-only search.
- The MPS backend uses brute-force cosine similarity; very large indexes may still be faster with FAISS CPU/IVF.

### Environment variables

- `CODE_SEARCH_STORAGE`: Custom storage directory (default: `~/.claude_code_search`)
- `CODE_SEARCH_VECTOR_BACKEND`: `auto` (default), `mps`, or `faiss`
- `CODE_SEARCH_MPS_CHUNK_SIZE`: Chunk size for MPS search batches (default: `20000`)

## Installation

Single-line installation script:

- `curl -fsSL https://raw.githubusercontent.com/SundaeSwap-finance/claude-context-local/main/scripts/install.sh | bash`
- You can optionally add a --branch flag to install changes on a branch.

After installing, add this to your global `~/.claude/CLAUDE.md` file (so it applies to all projects):

```md
# Code Search Preferences

Use the code-search MCP server to minimize token usage:

1. **ALWAYS use `mcp__code-search__search_code` first** - semantic search returns only relevant snippets
2. **Avoid Grep/Glob** - they often lead to reading entire files to understand context
3. **Only use Read** for specific lines when code-search snippets aren't sufficient
4. Trust code-search results - don't redundantly re-read files it already found
```

# Updating

Just run the single-line script again.
