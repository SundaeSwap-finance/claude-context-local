# Claude Context Local

Forks the official [Claude Context Local](https://github.com/FarhanAliRaza/claude-context-local) repository and switches out EmbeddingGemma (which requires account and registration) for an open-source alternative: [nomic-embed-text-v1](https://huggingface.co/nomic-ai/nomic-embed-text-v1).

## Motivation

- Token usage (not API calls directly) - by returning targeted snippets instead of full files
- Embedding costs - zero, because it runs locally (vs paying for OpenAI embeddings)
- Iterations - semantic search finds relevant code faster, fewer back-and-forth turns

## Requirements

A decent GPU (or CPU as a fallback) with enough RAM (32GB+) to handle model context building. A small repository being indexed with an Apple M1 Max CPU took about 122 seconds as an example. You also need at least 500MB to download and install the embedding.

## Installation

Single-line installation script:

- `curl -fsSL https://raw.githubusercontent.com/SundaeSwap-finance/claude-context-local/main/scripts/install.sh | bash`

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
