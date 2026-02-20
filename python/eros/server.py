"""
Eros MCP Server — semantic code intelligence tools.

Tools:
  - semantic_search: Find code or documentation by meaning/intent
  - find_similar: Find conceptually similar code
  - semantic_index: Manage the vector index
  - explain_retrieval: RAG diagnostic tool
"""

import logging
from pathlib import Path

from fastmcp import FastMCP

from eros.lifecycle import lifespan

logger = logging.getLogger("eros")


def _load_instructions() -> str | None:
    """Load agent instructions from EROS_AGENT_INSTRUCTIONS.md."""
    candidates = [
        Path("EROS_AGENT_INSTRUCTIONS.md"),
        Path(__file__).parent.parent.parent / "EROS_AGENT_INSTRUCTIONS.md",
    ]
    for path in candidates:
        try:
            return path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            continue
    return None


mcp = FastMCP(
    "Eros Semantic Code Intelligence",
    instructions=_load_instructions(),
    lifespan=lifespan,
)


@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True},
    output_schema=None,
)
async def semantic_search(
    query: str,
    scope: str = "all",
    language: str | None = None,
    file_pattern: str | None = None,
    limit: int = 20,
    explain: bool = False,
    workspace: str | None = None,
) -> str:
    """Find code or documentation by meaning/intent.

    Use this when text search (fast_search) returns noise or you're searching
    by concept rather than exact names. Searches embedding vectors built from
    Julie's symbol data and project documentation.

    Args:
        query: Natural language description of what you're looking for
        scope: Search scope — "code" (implementation), "docs" (documentation), or "all" (both, fused with RRF)
        language: Filter by programming language (e.g., "python", "rust")
        file_pattern: Filter by file glob pattern (e.g., "src/**/*.py")
        limit: Maximum results to return (default: 20)
        explain: Show score breakdown and ranking details for debugging search quality
        workspace: Filter by workspace — specific workspace ID, or None to search all indexed workspaces

    Returns:
        Matching code or documentation ranked by semantic relevance.
    """
    from eros.lifecycle import wait_for_init

    await wait_for_init()

    from eros import lifecycle
    from eros.retrieval import search

    return await search(
        query=query,
        scope=scope,
        language=language,
        file_pattern=file_pattern,
        limit=limit,
        explain=explain,
        embeddings=lifecycle.embeddings,
        storage=lifecycle.storage,
        config=lifecycle.config,
        workspace=workspace,
    )


@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True},
    output_schema=None,
)
async def find_similar(
    symbol: str,
    scope: str = "code",
    limit: int = 10,
    workspace: str | None = None,
) -> str:
    """Find conceptually similar code to a given symbol or code snippet.

    Use this to discover duplicate implementations, related patterns, or
    alternative approaches. Pass a symbol name or a short code snippet.

    Args:
        symbol: Symbol name or code snippet to find similar code for
        scope: Search scope — "code" or "docs"
        limit: Maximum results to return (default: 10)
        workspace: Filter by workspace — specific workspace ID, or None to search all indexed workspaces

    Returns:
        Code or documentation that is semantically similar.
    """
    from eros.lifecycle import wait_for_init

    await wait_for_init()

    from eros import lifecycle
    from eros.retrieval import find_similar_code

    return await find_similar_code(
        symbol=symbol,
        scope=scope,
        limit=limit,
        embeddings=lifecycle.embeddings,
        storage=lifecycle.storage,
        config=lifecycle.config,
        workspace=workspace,
    )


@mcp.tool(output_schema=None)
async def semantic_index(
    operation: str = "stats",
    workspace: str | None = None,
    doc_paths: list[str] | None = None,
) -> str:
    """Manage the semantic vector index.

    The index must be built before search works. Rebuild after Julie
    re-indexes or after documentation changes.

    Args:
        operation: "index" (full rebuild), "refresh" (incremental), "stats" (index size + model status), "health" (project setup check)
        workspace: Julie workspace ID to index (default: auto-discover primary)
        doc_paths: Additional documentation paths to index (directories or individual files)

    Returns:
        Operation result — index statistics, health report, or indexing summary.
    """
    from eros.lifecycle import wait_for_init

    await wait_for_init()

    from eros import lifecycle
    from eros.retrieval import manage_index

    return await manage_index(
        operation=operation,
        workspace=workspace,
        doc_paths=doc_paths,
        embeddings=lifecycle.embeddings,
        storage=lifecycle.storage,
        config=lifecycle.config,
    )


@mcp.tool(
    annotations={"readOnlyHint": True, "idempotentHint": True},
    output_schema=None,
)
async def explain_retrieval(
    query: str,
    result_id: str | None = None,
    result_text: str | None = None,
    workspace: str | None = None,
) -> str:
    """RAG diagnostic tool — understand why results ranked the way they did.

    Use this when search results seem wrong or you want to understand the
    ranking. Shows embedding similarity scores for the top result or a
    specific result you identify.

    Args:
        query: The search query to analyze
        result_id: ID of a specific result to explain (from metadata)
        result_text: Text of a specific result to explain (alternative to result_id)
        workspace: Filter by workspace — specific workspace ID, or None to search all indexed workspaces

    Returns:
        Score breakdown showing embedding similarity and ranking details.
    """
    from eros.lifecycle import wait_for_init

    await wait_for_init()

    from eros import lifecycle
    from eros.retrieval import explain

    return await explain(
        query=query,
        result_id=result_id,
        result_text=result_text,
        embeddings=lifecycle.embeddings,
        storage=lifecycle.storage,
        config=lifecycle.config,
        workspace=workspace,
    )


def _setup_logging(config) -> None:
    """Configure console (INFO) + rotating file (DEBUG) logging."""
    from logging.handlers import RotatingFileHandler

    root = logging.getLogger("eros")
    root.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    log_dir = config.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "eros.log", maxBytes=5_000_000, backupCount=3,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)


def main():
    """Entry point for `python -m eros.server`."""
    from eros.config import ErosConfig

    config = ErosConfig()
    _setup_logging(config)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
