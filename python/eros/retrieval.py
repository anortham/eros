"""
Search routing, RRF fusion, and result formatting.

Handles:
  - Scope-based routing (code → code model, docs → prose model, all → both + fusion)
  - Reciprocal Rank Fusion (RRF) for merging results from multiple sources
  - Result formatting for MCP tool output
  - Explain mode for RAG diagnostics
"""

import asyncio
import logging
from dataclasses import dataclass, field

from eros.config import ErosConfig
from eros.embeddings import DualEmbeddingManager
from eros.incremental import sql_quote
from eros.indexing import build_index
from eros.julie_reader import JulieReader
from eros.storage import VectorStorage

logger = logging.getLogger("eros.retrieval")


@dataclass
class SearchResult:
    """A single search result with score and metadata."""

    text: str
    score: float
    collection: str  # "code" or "docs"
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# RRF Fusion
# ---------------------------------------------------------------------------


def rrf_fuse(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF score = sum(1 / (k + rank_i)) across all lists where the item appears.
    Items appearing in multiple lists get boosted.

    Args:
        ranked_lists: List of ranked result lists
        k: RRF constant (higher = less emphasis on top ranks, default 60)
    """
    if not ranked_lists:
        return []

    # Score accumulator keyed by text (deduplication key)
    scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results):
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed, RRF uses 1-indexed
            key = result.text
            scores[key] = scores.get(key, 0.0) + rrf_score
            # Keep the result with more metadata
            if key not in best_result or len(result.metadata) > len(best_result[key].metadata):
                best_result[key] = result

    # Sort by accumulated RRF score
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    return [
        SearchResult(
            text=best_result[key].text,
            score=scores[key],
            collection=best_result[key].collection,
            metadata=best_result[key].metadata,
        )
        for key in sorted_keys
    ]


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def format_results(results: list[SearchResult], explain: bool = False) -> str:
    """Format search results as readable text for MCP output."""
    if not results:
        return "No results found."

    lines = [f"{len(results)} results:\n"]

    for i, r in enumerate(results, 1):
        if r.collection == "code":
            name = r.metadata.get("symbol_name", "?")
            kind = r.metadata.get("kind", "")
            lang = r.metadata.get("language", "")
            path = r.metadata.get("file_path", "")
            line = r.metadata.get("start_line", "")
            loc = f"{path}:{line}" if line else path
            lines.append(f"  {i}. {name} ({kind}, {lang}) — {loc}")
        else:
            path = r.metadata.get("file_path", "")
            section = r.metadata.get("section", "")
            lines.append(f"  {i}. [{section}] — {path}")

        if explain:
            lines.append(f"     score: {r.score:.4f}  collection: {r.collection}")

        # Show a preview of the text
        preview = r.text[:120].replace("\n", " ")
        if len(r.text) > 120:
            preview += "..."
        lines.append(f"     {preview}")
        lines.append("")

    return "\n".join(lines)


def format_explain(query: str, result: SearchResult) -> str:
    """Format detailed explanation of why a result ranked as it did."""
    lines = [
        f"Query: {query}",
        f"Result: {result.metadata.get('symbol_name', result.metadata.get('section', '?'))}",
        f"Collection: {result.collection}",
        f"Score: {result.score:.4f}",
        f"File: {result.metadata.get('file_path', '?')}",
        "",
        "Score breakdown:",
        f"  Embedding similarity: {result.score:.4f}",
        "",
        f"Text preview: {result.text[:200]}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXCLUDE_KEYS = frozenset(("vector", "text", "_distance"))


def _to_results(raw: list[dict], collection: str) -> list[SearchResult]:
    """Convert raw LanceDB results to SearchResult objects."""
    return [
        SearchResult(
            text=r.get("text", ""),
            score=1.0 / (1.0 + r.get("_distance", 0.0)),
            collection=collection,
            metadata={k: v for k, v in r.items() if k not in _EXCLUDE_KEYS},
        )
        for r in raw
    ]


# ---------------------------------------------------------------------------
# MCP Tool Implementations
# ---------------------------------------------------------------------------


async def search(
    query: str,
    scope: str,
    language: str | None,
    file_pattern: str | None,
    limit: int,
    explain: bool,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    config: ErosConfig,
    workspace: str | None = None,
) -> str:
    """Implementation of the semantic_search tool."""
    results: list[SearchResult] = []

    # Resolve workspace to a filter value for WHERE clauses.
    # None means no filter (search all).
    ws_filter = workspace  # pass through directly; None = no filter

    if scope in ("code", "all"):
        code_results = await _search_collection(
            query,
            "code",
            language,
            file_pattern,
            limit,
            embeddings,
            storage,
            workspace_id=ws_filter,
        )
        results.extend(code_results)

    if scope in ("docs", "all"):
        doc_results = await _search_collection(
            query,
            "docs",
            None,
            file_pattern,
            limit,
            embeddings,
            storage,
            workspace_id=ws_filter,
        )
        results.extend(doc_results)

    # If both scopes, fuse with RRF
    if scope == "all" and results:
        code_results = [r for r in results if r.collection == "code"]
        doc_results = [r for r in results if r.collection == "docs"]
        results = rrf_fuse([code_results, doc_results], k=config.rrf_k)

    # Apply overall limit
    results = results[:limit]

    return format_results(results, explain=explain)


async def _search_collection(
    query: str,
    collection: str,
    language: str | None,
    file_pattern: str | None,
    limit: int,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    workspace_id: str | None = None,
) -> list[SearchResult]:
    """Search a single collection."""
    # Embed the query with the appropriate model
    query_vec = await asyncio.to_thread(embeddings.embed_query, query, collection)

    # Build WHERE clause for filtering
    where_parts = []
    if language and collection == "code":
        where_parts.append(f"language = '{sql_quote(language)}'")
    if file_pattern:
        # Simple glob-to-SQL: convert * to %
        pattern = sql_quote(file_pattern.replace("*", "%"))
        where_parts.append(f"file_path LIKE '{pattern}'")
    if workspace_id:
        where_parts.append(f"workspace_id = '{sql_quote(workspace_id)}'")

    where = " AND ".join(where_parts) if where_parts else None

    raw_results = await asyncio.to_thread(storage.search, collection, query_vec, limit, where)

    return _to_results(raw_results, collection)


async def find_similar_code(
    symbol: str,
    scope: str,
    limit: int,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    config: ErosConfig,
    workspace: str | None = None,
) -> str:
    """Implementation of the find_similar tool."""
    collection = "code" if scope == "code" else "docs"
    query_vec = await asyncio.to_thread(embeddings.embed_query, symbol, collection)

    where = f"workspace_id = '{sql_quote(workspace)}'" if workspace else None
    raw_results = await asyncio.to_thread(storage.search, collection, query_vec, limit, where)

    return format_results(_to_results(raw_results, collection), explain=False)


async def manage_index(
    operation: str,
    workspace: str | None,
    doc_paths: list[str] | None,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    config: ErosConfig,
) -> str:
    """Implementation of the semantic_index tool."""
    if operation == "stats":
        stats = storage.stats()
        lines = ["Eros Index Statistics:", ""]
        for name, data in stats.items():
            lines.append(f"  {name}: {data['count']} chunks")
        code_model = embeddings.code_model
        docs_model = embeddings.docs_model
        lines.append("")
        lines.append(
            f"  Code model: {code_model.model_name} ({'loaded' if code_model.is_loaded else 'not loaded'})"
        )
        lines.append(
            f"  Docs model: {docs_model.model_name} ({'loaded' if docs_model.is_loaded else 'not loaded'})"
        )
        return "\n".join(lines)

    elif operation == "health":
        stats = storage.stats()
        lines = ["Eros Health Check:", ""]
        lines.append(f"  Project root: {config.project_root}")
        julie_exists = config.julie_dir.exists()
        lines.append(f"  Julie data: {'found' if julie_exists else 'NOT FOUND'}")

        # List workspaces
        if julie_exists:
            try:
                reader = JulieReader(config.project_root)
                workspaces = reader.list_workspaces()
                lines.append(f"  Workspaces: {len(workspaces)}")
                for ws in workspaces:
                    lines.append(f"    - {ws.display_name} ({ws.workspace_type}) [{ws.id}]")
            except (FileNotFoundError, ValueError) as e:
                lines.append(f"  Workspaces: error reading registry ({e})")

        lines.append(f"  Code chunks: {stats['code']['count']}")
        lines.append(f"  Doc chunks: {stats['docs']['count']}")

        if not julie_exists:
            lines.append("")
            lines.append(
                "  WARNING: No .julie directory found. Run Julie to index this project first."
            )

        # Check for stale index (missing workspace_id column)
        if stats["code"]["count"] > 0:
            try:
                code_table = storage._tables.get("code")
                if code_table is not None:
                    schema = code_table.schema
                    if "workspace_id" not in [f.name for f in schema]:
                        lines.append("")
                        lines.append(
                            "  WARNING: Index is missing workspace_id column (pre-workspace-support)."
                        )
                        lines.append("  Run semantic_index(operation='index') to rebuild.")
            except Exception:
                pass

        return "\n".join(lines)

    elif operation in ("index", "refresh"):
        return await _build_index(
            full_rebuild=(operation == "index"),
            workspace=workspace,
            doc_paths=doc_paths,
            embeddings=embeddings,
            storage=storage,
            config=config,
        )

    else:
        return f"Unknown operation: {operation}. Use: index, refresh, stats, health"


async def _build_index(
    full_rebuild: bool,
    workspace: str | None,
    doc_paths: list[str] | None,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    config: ErosConfig,
) -> str:
    return await build_index(
        full_rebuild=full_rebuild,
        workspace=workspace,
        doc_paths=doc_paths,
        embeddings=embeddings,
        storage=storage,
        config=config,
    )


async def explain(
    query: str,
    result_id: str | None,
    result_text: str | None,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    config: ErosConfig,
    workspace: str | None = None,
) -> str:
    """Implementation of the explain_retrieval tool."""
    where = f"workspace_id = '{sql_quote(workspace)}'" if workspace else None

    # Run the search to get results with scores
    code_vec = await asyncio.to_thread(embeddings.embed_query, query, "code")
    code_results = await asyncio.to_thread(storage.search, "code", code_vec, 10, where)

    docs_vec = await asyncio.to_thread(embeddings.embed_query, query, "docs")
    doc_results = await asyncio.to_thread(storage.search, "docs", docs_vec, 10, where)

    all_results = _to_results(code_results, "code") + _to_results(doc_results, "docs")

    # Find the specific result to explain
    target = None
    if result_text:
        for r in all_results:
            if result_text in r.text:
                target = r
                break
    elif result_id:
        for r in all_results:
            if r.metadata.get("symbol_id") == result_id:
                target = r
                break

    if target is None and all_results:
        target = all_results[0]  # Explain the top result

    if target is None:
        return "No results found for this query."

    return format_explain(query, target)
