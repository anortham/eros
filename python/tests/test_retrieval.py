"""Tests for retrieval logic — search routing, RRF fusion, and reranking."""


import numpy as np
import pytest
from eros.config import ErosConfig
from eros.retrieval import (
    SearchResult,
    format_explain,
    format_results,
    manage_index,
    rrf_fuse,
    search,
)
from eros.storage import VectorStorage


class FakeEmbeddings:
    """Minimal fake that returns random vectors of a fixed dimension."""

    def __init__(self, dim: int = 8):
        self._dim = dim

        class FakeSlot:
            model_name = "fake-model"
            is_loaded = True

        self.code_model = FakeSlot()
        self.docs_model = FakeSlot()

    @property
    def code_dimensions(self):
        return self._dim

    @property
    def docs_dimensions(self):
        return self._dim

    def embed_code(self, texts, batch_size=32):
        return np.random.randn(len(texts), self._dim).astype(np.float32)

    def embed_docs(self, texts, batch_size=32):
        return np.random.randn(len(texts), self._dim).astype(np.float32)

    def embed_query(self, query, scope):
        return np.random.randn(self._dim).astype(np.float32)


class TestRRFFusion:
    """Test Reciprocal Rank Fusion for merging ranked result lists."""

    def test_single_list(self):
        """RRF of a single list should preserve order."""
        results = [
            SearchResult(text="a", score=0.9, collection="code", metadata={"id": "1"}),
            SearchResult(text="b", score=0.5, collection="code", metadata={"id": "2"}),
        ]
        fused = rrf_fuse([results], k=60)
        assert len(fused) == 2
        assert fused[0].text == "a"
        assert fused[1].text == "b"

    def test_two_lists_merge(self):
        """RRF should merge two lists, boosting items that appear in both."""
        list_a = [
            SearchResult(text="shared", score=0.9, collection="code", metadata={"id": "1"}),
            SearchResult(text="only_a", score=0.5, collection="code", metadata={"id": "2"}),
        ]
        list_b = [
            SearchResult(text="only_b", score=0.8, collection="docs", metadata={"id": "3"}),
            SearchResult(text="shared", score=0.7, collection="code", metadata={"id": "1"}),
        ]
        fused = rrf_fuse([list_a, list_b], k=60)
        # "shared" appears in both lists, so it should rank highest
        assert fused[0].text == "shared"

    def test_empty_lists(self):
        """RRF of empty lists should return empty."""
        assert rrf_fuse([], k=60) == []
        assert rrf_fuse([[]], k=60) == []

    def test_deduplication_by_text(self):
        """Same text from different lists should be merged, not duplicated."""
        list_a = [
            SearchResult(text="same", score=0.9, collection="code", metadata={"id": "1"}),
        ]
        list_b = [
            SearchResult(text="same", score=0.8, collection="docs", metadata={"id": "1"}),
        ]
        fused = rrf_fuse([list_a, list_b], k=60)
        assert len(fused) == 1


class TestFormatResults:
    """Test formatting search results for MCP output."""

    def test_format_basic(self):
        """Should format results as readable text."""
        results = [
            SearchResult(
                text="def authenticate(user, pwd)",
                score=0.92,
                collection="code",
                metadata={
                    "symbol_name": "authenticate",
                    "kind": "function",
                    "language": "python",
                    "file_path": "src/auth.py",
                    "start_line": 10,
                },
            ),
        ]
        output = format_results(results, explain=False)
        assert "authenticate" in output
        assert "src/auth.py" in output
        assert "function" in output

    def test_format_with_explain(self):
        """Explain mode should include score details."""
        results = [
            SearchResult(
                text="some code",
                score=0.85,
                collection="code",
                metadata={"symbol_name": "x", "file_path": "x.py"},
            ),
        ]
        output = format_results(results, explain=True)
        assert "0.85" in output or "score" in output.lower()

    def test_format_empty(self):
        """Empty results should produce a meaningful message."""
        output = format_results([], explain=False)
        assert "no results" in output.lower() or "0 results" in output.lower()


class TestFormatExplain:
    """Test the explain_retrieval diagnostic output."""

    def test_explain_output(self):
        """Should show score breakdown for a result."""
        result = SearchResult(
            text="def validate(data)",
            score=0.88,
            collection="code",
            metadata={"symbol_name": "validate", "file_path": "val.py"},
        )
        output = format_explain("validate input", result)
        assert "validate" in output
        assert "0.88" in output or "score" in output.lower()


class TestMultiWorkspaceIndexing:
    """Test that _build_index indexes all workspaces and tags chunks."""

    @pytest.fixture
    def indexing_env(self, julie_project_with_refs, tmp_path):
        """Set up config, storage, and fake embeddings for indexing tests."""
        eros_dir = tmp_path / "eros_data"
        config = ErosConfig(
            project_root=julie_project_with_refs["project_root"],
            eros_data_dir=eros_dir,
        )
        storage = VectorStorage(config, embedding_dims={"code": 8, "docs": 8})
        embeddings = FakeEmbeddings(dim=8)
        return {
            "config": config,
            "storage": storage,
            "embeddings": embeddings,
            **julie_project_with_refs,
        }

    @pytest.mark.asyncio
    async def test_index_all_workspaces(self, indexing_env):
        """Indexing with workspace='all' should index primary + reference."""
        await manage_index(
            operation="index",
            workspace="all",
            doc_paths=None,
            embeddings=indexing_env["embeddings"],
            storage=indexing_env["storage"],
            config=indexing_env["config"],
        )
        # Should report chunks from both workspaces
        stats = indexing_env["storage"].stats()
        assert stats["code"]["count"] == 5  # 4 primary + 1 reference

    @pytest.mark.asyncio
    async def test_indexed_chunks_have_workspace_id(self, indexing_env):
        """Each code chunk should be tagged with its workspace_id."""
        await manage_index(
            operation="index",
            workspace="all",
            doc_paths=None,
            embeddings=indexing_env["embeddings"],
            storage=indexing_env["storage"],
            config=indexing_env["config"],
        )
        # Search for all results (no filter)
        query_vec = np.random.randn(8).astype(np.float32)
        results = indexing_env["storage"].search("code", query_vec, limit=10)
        workspace_ids = {r["workspace_id"] for r in results}
        assert indexing_env["primary_id"] in workspace_ids
        assert indexing_env["ref_id"] in workspace_ids

    @pytest.mark.asyncio
    async def test_index_primary_only(self, indexing_env):
        """Indexing with workspace='primary' should only index primary."""
        await manage_index(
            operation="index",
            workspace="primary",
            doc_paths=None,
            embeddings=indexing_env["embeddings"],
            storage=indexing_env["storage"],
            config=indexing_env["config"],
        )
        stats = indexing_env["storage"].stats()
        assert stats["code"]["count"] == 4  # Only primary symbols

    @pytest.mark.asyncio
    async def test_index_specific_workspace(self, indexing_env):
        """Indexing a specific workspace ID should only index that one."""
        ref_id = indexing_env["ref_id"]
        await manage_index(
            operation="index",
            workspace=ref_id,
            doc_paths=None,
            embeddings=indexing_env["embeddings"],
            storage=indexing_env["storage"],
            config=indexing_env["config"],
        )
        stats = indexing_env["storage"].stats()
        assert stats["code"]["count"] == 1  # Only reference symbol

    @pytest.mark.asyncio
    async def test_index_reports_timing(self, indexing_env):
        """Index output should include total and per-workspace timing."""
        import re

        result = await manage_index(
            operation="index",
            workspace="all",
            doc_paths=None,
            embeddings=indexing_env["embeddings"],
            storage=indexing_env["storage"],
            config=indexing_env["config"],
        )
        # Total time in summary line
        assert re.search(r"in \d+\.\d+s", result), f"No total timing in: {result}"
        # Per-workspace time in brackets
        assert re.search(r"\[\d+\.\d+s\]", result), f"No per-ws timing in: {result}"


class TestWorkspaceFilteredSearch:
    """Test that search() filters results by workspace."""

    @pytest.fixture
    async def indexed_env(self, julie_project_with_refs, tmp_path):
        """Index all workspaces and return the env for search tests."""
        eros_dir = tmp_path / "eros_data"
        config = ErosConfig(
            project_root=julie_project_with_refs["project_root"],
            eros_data_dir=eros_dir,
        )
        storage = VectorStorage(config, embedding_dims={"code": 8, "docs": 8})
        embeddings = FakeEmbeddings(dim=8)

        # Index all workspaces
        await manage_index(
            operation="index",
            workspace="all",
            doc_paths=None,
            embeddings=embeddings,
            storage=storage,
            config=config,
        )

        return {
            "config": config,
            "storage": storage,
            "embeddings": embeddings,
            **julie_project_with_refs,
        }

    @pytest.mark.asyncio
    async def test_search_primary_only(self, indexed_env):
        """Search with workspace='primary' returns only primary workspace results."""
        result = await search(
            query="authenticate user",
            scope="code",
            language=None,
            file_pattern=None,
            limit=20,
            explain=False,
            workspace=indexed_env["primary_id"],
            embeddings=indexed_env["embeddings"],
            storage=indexed_env["storage"],
            config=indexed_env["config"],
        )
        # Should not contain the reference workspace's Rust "retry" function
        assert "retry" not in result.lower()

    @pytest.mark.asyncio
    async def test_search_all_workspaces(self, indexed_env):
        """Search with workspace=None returns results from all workspaces."""
        result = await search(
            query="function",
            scope="code",
            language=None,
            file_pattern=None,
            limit=20,
            explain=False,
            workspace=None,
            embeddings=indexed_env["embeddings"],
            storage=indexed_env["storage"],
            config=indexed_env["config"],
        )
        # With no workspace filter, should return results (5 total indexed)
        assert "results" in result.lower()

    @pytest.mark.asyncio
    async def test_search_specific_workspace(self, indexed_env):
        """Search with a specific workspace ID returns only that workspace."""
        ref_id = indexed_env["ref_id"]
        result = await search(
            query="retry operation",
            scope="code",
            language=None,
            file_pattern=None,
            limit=20,
            explain=False,
            workspace=ref_id,
            embeddings=indexed_env["embeddings"],
            storage=indexed_env["storage"],
            config=indexed_env["config"],
        )
        # Should only contain reference workspace results
        assert "results" in result.lower()
