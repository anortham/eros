# Eros

**Semantic Code Intelligence** — meaning-based search for code and documentation.

Eros is an **optional add-on** MCP server that adds semantic/vector search to your AI coding workflow. It reads code structure from [Julie](https://github.com/anortham/julie) and makes it searchable by meaning, not just text matching.

Named after the station in *The Expanse* where the protomolecule first transformed raw data into understanding.

> **Prerequisite:** Eros requires [Julie](https://github.com/anortham/julie) to be installed, running, and indexed on your project. Julie provides the code structure data (symbols, files) that Eros embeds and searches. Without Julie, Eros has nothing to index.

## What It Does

- **Semantic search**: Find code by what it *does*, not what it's *named* ("function that retries HTTP requests")
- **Cross-project search**: Index reference workspaces from Julie to search across related codebases
- **Dual-model embeddings**: Code-optimized model for source, prose model for documentation
- **Similar code detection**: Find duplicate implementations, related patterns, or alternative approaches
- **RAG diagnostics**: Understand why results ranked the way they did

## How It Works

```
Julie's SQLite (.julie/indexes/{workspace}/db/symbols.db)
    → Eros reads symbols + file content
Disk files (.md, .txt, .rst)
    → Eros reads and chunks documentation
        ↓
Dual embedding models (code + prose)
        ↓
LanceDB vector store (.eros/vectors.lance/)
        ↓
4 semantic MCP tools
```

Eros is a **read-only consumer** of Julie's data. It never writes to `.julie/`.

## Quick Start

### 1. Install

```bash
git clone https://github.com/anortham/eros.git
cd eros
uv venv --python 3.12
uv pip install -e ".[dev]"
```

### 2. Make sure Julie is running

Eros reads Julie's SQLite databases at `{project}/.julie/`. If Julie hasn't indexed your project yet, run Julie first.

### 3. Configure your MCP client

#### Claude Code

Add to `~/.claude/settings.json`:

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

Claude Code passes `cwd` correctly, so `EROS_WORKSPACE` is optional if you're working in the project directory.

#### VS Code / GitHub Copilot

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

`EROS_WORKSPACE` is **required** for VS Code — its MCP integration doesn't pass `cwd`, so without it Eros won't find your `.julie/` data.

### 4. Build the index

Once Eros is running, build the semantic index:

```
semantic_index(operation="index")
```

This embeds all code symbols and documentation into the vector store. First run downloads the embedding models (~100MB) and takes a few minutes depending on project size.

### 5. Search

```
semantic_search("function that validates email addresses", scope="code")
semantic_search("how authentication works", scope="docs")
find_similar("authenticate")
```

## Tools

| Tool | Purpose |
|------|---------|
| `semantic_search` | Find code/docs by meaning. Supports `scope` (code/docs/all), `workspace` filter, `language` filter |
| `find_similar` | Find conceptually similar code to a symbol or snippet |
| `semantic_index` | Build/rebuild vector index. Operations: `index`, `refresh`, `stats`, `health` |
| `explain_retrieval` | RAG diagnostics — understand why results ranked as they did |

## Reference Workspaces

If Julie has reference workspaces configured, Eros indexes them too. This lets you search across related codebases:

```
# Search only a specific workspace
semantic_search("retry logic", workspace="shared-lib_abc123")

# Search all workspaces (default)
semantic_search("retry logic")
```

Use `semantic_index(operation="health")` to see which workspaces Eros has discovered.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `EROS_WORKSPACE` | Current directory | Project root (where `.julie/` lives). **Required for VS Code.** |
| `EROS_CODE_MODEL` | `nomic-ai/CodeRankEmbed` | Code embedding model |
| `EROS_DOCS_MODEL` | `BAAI/bge-small-en-v1.5` | Documentation embedding model |
| `EROS_DEVICE` | `cpu` | Compute device: `cpu`, `mps`, `cuda`, or `auto` |
| `EROS_DATA_DIR` | `.eros` | Where to store vector data |

### Swapping models

Eros supports A/B testing different embedding models for RAG research:

```bash
EROS_CODE_MODEL=jinaai/jina-embeddings-v2-base-code python -m eros
```

After switching models, rebuild the index with `semantic_index(operation="index")`.

## Logging

Eros writes logs to `.eros/logs/eros.log` (rotating, 5MB max, 3 backups). Console output is INFO-level; the log file captures DEBUG-level detail including embedding throughput and model load times.

## Development

```bash
# Run tests
pytest python/tests/ -v

# Lint
ruff check python/

# Run server directly
python -m eros
```

## Requirements

- Python 3.12+
- [Julie](https://github.com/anortham/julie) installed and indexed on your project
- ~500MB disk for embedding models (downloaded on first run)
- ~100MB+ for vector index (varies with project size)

## Part of the Expanse Stack

Eros is one piece of a modular AI coding toolkit:

| Tool | Role | Required? |
|------|------|-----------|
| [Julie](https://github.com/anortham/julie) | Structural code intelligence (symbols, refs, definitions) | **Yes** — Eros depends on Julie's data |
| **Eros** | Semantic code intelligence (meaning-based search) | Optional add-on to Julie |
| [Goldfish](https://github.com/anortham/goldfish) | Development memory (checkpoints, plans, recall) | Independent — no dependency |

Each tool works as a standalone MCP server. Julie is the foundation that Eros builds on.

## License

MIT
