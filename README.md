# Eros

**Semantic Code Intelligence** — meaning-based search for code and documentation.

Eros is a focused MCP server that adds semantic/vector search as an optional companion to [Julie](https://github.com/anortham/julie) (structural code intelligence) and [Goldfish](https://github.com/anortham/goldfish) (development memory).

Named after the station in *The Expanse* where the protomolecule first transformed raw data into understanding.

## What It Does

- **Semantic search**: Find code by meaning, not just text matching
- **Dual-model embeddings**: Code-optimized model for source, prose model for documentation
- **Similar code detection**: "Show me code that does something like this"
- **RAG diagnostics**: Understand why results ranked the way they did

## Architecture

```
Julie's SQLite → Code symbols
Disk files (.md) → Documentation
        ↓
Dual embedding models (code + prose)
        ↓
LanceDB vector store
        ↓
4 semantic MCP tools
```

## Quick Start

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
```

## Tools

| Tool | Purpose |
|------|---------|
| `semantic_search` | Find code/docs by meaning |
| `find_similar` | Find conceptually similar code |
| `semantic_index` | Manage vector index |
| `explain_retrieval` | RAG diagnostics |

## Requirements

- Python 3.12+
- [Julie](https://github.com/anortham/julie) running and indexed (provides code data)

## License

MIT
