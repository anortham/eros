# Eros — Agent Onboarding Guide

## What Is Eros?

A pure-Python semantic code intelligence MCP server. Adds meaning-based search as an optional companion to Julie (structural intelligence) and Goldfish (development memory).

Named after the station in *The Expanse* where the protomolecule first transformed raw data into understanding.

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

## Key Architecture Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Pure Python | No Rust needed — Julie handles extraction |
| Julie dependency | Hard requirement | Reads Julie's SQLite. Zero duplicate parsing. |
| Vector store | LanceDB | Proven, supports hybrid search + FTS |
| Embedding | Dual-model (code + prose) | RAG experiment: code-specific vs general-purpose |
| MCP framework | FastMCP | Battle-tested async patterns |

---

## 🔴 Critical Rules

### 1. TDD Is Mandatory
No code without tests. Red → Green → Refactor.

### 2. File Size Limit: 500 Lines Max
Target 200-300 lines. Refactor if approaching 500.

### 3. Don't Break Lazy Loading
Heavy ML imports (torch, sentence-transformers) MUST use `asyncio.to_thread()` in `lifecycle.py`. The MCP handshake must complete in <100ms.

### 4. Julie Data Is Read-Only
Never write to Julie's `.julie/` databases. Eros is a consumer, not a producer.

---

## Build & Test

```bash
# Setup
uv venv --python 3.12
uv pip install -e ".[dev]"

# Tests
pytest python/tests/ -v

# Run server
python -m eros
```

---

## File Layout

```
eros/
├── python/
│   ├── eros/
│   │   ├── __init__.py        # Minimal — just version
│   │   ├── server.py          # 4 MCP tools
│   │   ├── embeddings.py      # Dual-model manager (code + prose)
│   │   ├── julie_reader.py    # Read Julie's SQLite data
│   │   ├── chunking.py        # Code + doc chunking strategies
│   │   ├── retrieval.py       # Search routing, RRF fusion
│   │   ├── storage.py         # LanceDB vector store
│   │   ├── lifecycle.py       # Startup, lazy model loading
│   │   └── config.py          # Model selection, paths, env vars
│   └── tests/
│       ├── conftest.py        # Julie mock fixtures
│       ├── test_julie_reader.py
│       ├── test_chunking.py
│       ├── test_embeddings.py
│       └── test_retrieval.py
├── .eros/                     # Runtime data (gitignored)
│   └── vectors.lance/
├── pyproject.toml
└── CLAUDE.md                  # This file
```

---

## MCP Tools

### `semantic_search`
Find code or documentation by meaning.
- `scope`: "code", "docs", or "all"
- `explain=true`: Show score breakdown

### `find_similar`
Find conceptually similar code to a symbol or snippet.

### `semantic_index`
Manage the vector index.
- Operations: `index`, `refresh`, `stats`, `health`

### `explain_retrieval`
RAG diagnostic tool — understand why results ranked as they did.

---

## Configuration (Environment Variables)

| Variable | Default | Purpose |
|---|---|---|
| `EROS_WORKSPACE` | Current directory | Project root (where .julie/ lives). **Required for VS Code.** |
| `EROS_CODE_MODEL` | `nomic-ai/CodeRankEmbed` | Code embedding model |
| `EROS_DOCS_MODEL` | `BAAI/bge-small-en-v1.5` | Documentation embedding model |
| `EROS_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Result reranker |
| `EROS_PROJECT_ROOT` | (fallback for EROS_WORKSPACE) | Legacy alias for EROS_WORKSPACE |
| `EROS_DATA_DIR` | `.eros` | Where to store vector data |

---

## How Julie Integration Works

Julie stores data **per-project** in `{project_root}/.julie/`:

```
my-project/
├── .julie/
│   ├── workspace_registry.json    ← Eros reads this to find workspace ID
│   └── indexes/
│       └── {workspace_id}/
│           └── db/
│               └── symbols.db    ← Eros reads symbols + files tables
└── .eros/
    └── vectors.lance/            ← Eros stores embeddings here
```

Discovery flow:
1. Read `.julie/workspace_registry.json`
2. Extract `primary_workspace.directory_name`
3. Open `.julie/indexes/{id}/db/symbols.db` (read-only)
4. Query `symbols` and `files` tables

---

## RAG Experiments

Swap models via environment variables and compare search quality:

```bash
# A/B test code models
EROS_CODE_MODEL=jinaai/jina-embeddings-v2-base-code python -m eros
EROS_CODE_MODEL=BAAI/bge-small-en-v1.5 python -m eros

# After switching, rebuild index:
semantic_index(operation="index")
```
