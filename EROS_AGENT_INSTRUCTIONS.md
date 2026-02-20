# Eros — Semantic Code Intelligence Instructions

Eros adds **meaning-based search** to your toolkit. While Julie finds code by structure (names, references, definitions), Eros finds code by **what it does** — using embedding models that understand intent.

## When to Use Eros vs Julie

| Need | Use | Why |
|---|---|---|
| Find a function by name | Julie `fast_search` | Exact match, instant |
| Find who calls a symbol | Julie `fast_refs` | Reference graph, complete |
| Find code that "validates email" | Eros `semantic_search` | Meaning-based, fuzzy |
| Find similar implementations | Eros `find_similar` | Embedding similarity |
| Understand a symbol before changing it | Julie `deep_dive` | Callers, callees, types |

**Rule of thumb:** Try Julie first (faster, more precise for exact matches). Use Eros when you're searching by concept, not by name.

## Tool Guide

### semantic_search — Find by Meaning

Use when text search returns noise or you're looking for conceptual matches.

```
semantic_search("function that retries failed HTTP requests", scope="code")
semantic_search("how authentication works", scope="docs")
semantic_search("error handling patterns", scope="all", language="python")
```

- `scope="code"` searches implementation (functions, classes, methods)
- `scope="docs"` searches documentation (markdown, text files)
- `scope="all"` searches both and fuses results with Reciprocal Rank Fusion
- Use `language` and `file_pattern` to narrow noisy results
- Set `explain=true` to see score breakdowns when debugging search quality

### find_similar — Find Related Code

Use when you have a symbol or snippet and want to find conceptually similar code elsewhere.

```
find_similar("authenticate")           # Find code similar to the authenticate function
find_similar("retry with backoff")     # Find retry patterns by description
```

- Good for discovering duplicate implementations, related patterns, or alternative approaches
- `scope="code"` (default) searches implementation; `scope="docs"` searches documentation

### semantic_index — Manage the Index

The vector index must be built before search works. Operations:

- `"stats"` — Show index size and model status (default)
- `"health"` — Check project setup (Julie data, chunk counts)
- `"index"` — Full rebuild from Julie's data + documentation files
- `"refresh"` — Incremental update (same as index currently)

**When to rebuild:**
- After Julie re-indexes the project (new/changed symbols)
- After adding or editing documentation files
- If search quality seems degraded

```
semantic_index(operation="health")     # Quick check
semantic_index(operation="index")      # Full rebuild
```

### explain_retrieval — Debug Search Quality

RAG diagnostic tool. Shows why results ranked the way they did.

```
explain_retrieval("validate user input")                    # Explain top result
explain_retrieval("validate user input", result_text="...")  # Explain specific result
```

Use this when search results seem wrong — it reveals the embedding similarity score and helps you understand what the model "thinks" about a query.

## What Gets Indexed

**Code** (from Julie's symbol database):
- Functions, classes, methods, structs, traits, interfaces
- Excludes imports, variables, and constants (low semantic value)
- Each symbol becomes one chunk: signature + doc comment + body

**Documentation** (auto-discovered):
- Root-level files: README.md, CLAUDE.md, CHANGELOG.md, etc.
- Doc directories: `docs/`, `doc/`, `documentation/`
- Split by markdown headings with configurable overlap

## Key Principles

- **Julie first, Eros second** — structural search is faster and more precise for exact matches
- **Rebuild after changes** — the index is a snapshot; run `semantic_index(operation="index")` after significant code changes
- **Trust the scores** — higher scores mean stronger semantic matches; use `explain=true` to understand ranking
- **Scope narrows results** — use `scope="code"` or `scope="docs"` when you know what you're looking for
