"""
Eros MCP Server — 4 semantic search tools.

Tools:
  - semantic_search: Find code or documentation by meaning/intent
  - find_similar: Find conceptually similar code
  - semantic_index: Manage the vector index
  - explain_retrieval: RAG diagnostic tool
"""

import logging

from fastmcp import FastMCP

from eros.lifecycle import lifespan

logger = logging.getLogger("eros")

mcp = FastMCP(
    "Eros Semantic Code Intelligence",
    lifespan=lifespan,
)


@mcp.tool()
async def semantic_search(
    query: str,
    scope: str = "all",
    language: str | None = None,
    file_pattern: str | None = None,
    limit: int = 20,
    explain: bool = False,
) -> str:
    """Find code or documentation by meaning/intent.

    Args:
        query: Natural language description of what you're looking for
        scope: Search scope — "code", "docs", or "all"
        language: Filter by programming language (e.g., "python", "rust")
        file_pattern: Filter by file glob pattern (e.g., "src/**/*.py")
        limit: Maximum results to return
        explain: Show score breakdown and ranking details

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
    )


@mcp.tool()
async def find_similar(
    symbol: str,
    scope: str = "code",
    limit: int = 10,
) -> str:
    """Find conceptually similar code to a given symbol or code snippet.

    Args:
        symbol: Symbol name or code snippet to find similar code for
        scope: Search scope — "code" or "docs"
        limit: Maximum results to return

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
    )


@mcp.tool()
async def semantic_index(
    operation: str = "stats",
    workspace: str | None = None,
    doc_paths: list[str] | None = None,
) -> str:
    """Manage the semantic vector index.

    Args:
        operation: "index" (build/rebuild), "refresh" (incremental), "stats", "health"
        workspace: Julie workspace ID to index (default: auto-discover primary)
        doc_paths: Additional documentation paths to index

    Returns:
        Operation result — index statistics, health report, etc.
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


@mcp.tool()
async def explain_retrieval(
    query: str,
    result_id: str | None = None,
    result_text: str | None = None,
) -> str:
    """RAG diagnostic tool — understand why results ranked the way they did.

    Args:
        query: The search query to analyze
        result_id: ID of a specific result to explain
        result_text: Text of a specific result to explain (alternative to result_id)

    Returns:
        Detailed score breakdown: embedding similarity, token overlap,
        reranker score, fusion weight.
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
    )


def main():
    """Entry point for `python -m eros.server`."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
