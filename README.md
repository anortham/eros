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

## Setup

### Claude Code

Add to your MCP settings (e.g. `~/.claude/settings.json`):

```json
{
  "mcpServers": {
    "eros": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/eros", "python", "-m", "eros"],
      "env": {
        "EROS_WORKSPACE": "/path/to/your/project"
      }
    }
  }
}
```

Claude Code passes `cwd` correctly, so `EROS_WORKSPACE` is optional — Eros will use the current directory by default. Set it explicitly if you run Eros from a different directory than your project.

### VS Code / GitHub Copilot

Add to `.vscode/mcp.json` in your project:

```json
{
  "servers": {
    "eros": {
      "type": "stdio",
      "command": "uv",
      "args": ["run", "--directory", "/path/to/eros", "python", "-m", "eros"],
      "env": {
        "EROS_WORKSPACE": "${workspaceFolder}"
      }
    }
  }
}
```

**Important:** The `EROS_WORKSPACE` environment variable is **required** for VS Code to ensure Eros finds your project's `.julie/` data and creates its `.eros/` directory in the right place. VS Code's MCP integration doesn't pass `cwd` the way Claude Code does, so without this variable Eros will look in the wrong directory.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `EROS_WORKSPACE` | Current directory | Project root (where `.julie/` lives). **Required for VS Code.** |
| `EROS_CODE_MODEL` | `nomic-ai/CodeRankEmbed` | Code embedding model |
| `EROS_DOCS_MODEL` | `BAAI/bge-small-en-v1.5` | Documentation embedding model |
| `EROS_DATA_DIR` | `.eros` | Where to store vector data |

## License

MIT
