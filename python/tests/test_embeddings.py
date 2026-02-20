"""Tests for dual-model embedding manager and vector storage."""

import numpy as np
import pytest

from eros.chunking import Chunk
from eros.config import ErosConfig
from eros.storage import VectorStorage


class TestVectorStorage:
    """Test LanceDB vector store operations."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a VectorStorage with temp directory."""
        config = ErosConfig(eros_data_dir=tmp_path / ".eros")
        return VectorStorage(config, embedding_dims={"code": 8, "docs": 8})

    def test_create_storage(self, storage):
        """Should initialize without errors."""
        assert storage is not None

    def test_add_code_chunks(self, storage):
        """Should store code chunks with vectors."""
        chunks = [
            Chunk(
                text="def authenticate(user, password)",
                collection="code",
                metadata={
                    "symbol_id": "sym1",
                    "symbol_name": "authenticate",
                    "kind": "function",
                    "language": "python",
                    "file_path": "src/auth.py",
                    "start_line": 1,
                    "end_line": 5,
                },
            ),
            Chunk(
                text="class User",
                collection="code",
                metadata={
                    "symbol_id": "sym2",
                    "symbol_name": "User",
                    "kind": "class",
                    "language": "python",
                    "file_path": "src/models.py",
                    "start_line": 1,
                    "end_line": 10,
                },
            ),
        ]
        vectors = np.random.randn(2, 8).astype(np.float32)
        count = storage.add_chunks(chunks, vectors)
        assert count == 2

    def test_add_doc_chunks(self, storage):
        """Should store doc chunks with vectors."""
        chunks = [
            Chunk(
                text="# Authentication\n\nJWT-based auth system.",
                collection="docs",
                metadata={"file_path": "docs/auth.md", "section": "Authentication"},
            ),
        ]
        vectors = np.random.randn(1, 8).astype(np.float32)
        count = storage.add_chunks(chunks, vectors)
        assert count == 1

    def test_search_code(self, storage):
        """Should return nearest neighbors for code search."""
        chunks = [
            Chunk(
                text="def authenticate(user, password)",
                collection="code",
                metadata={
                    "symbol_id": "sym1",
                    "symbol_name": "authenticate",
                    "kind": "function",
                    "language": "python",
                    "file_path": "src/auth.py",
                    "start_line": 1,
                    "end_line": 5,
                },
            ),
        ]
        vec = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        storage.add_chunks(chunks, vec)

        # Search with a similar vector
        query_vec = np.array([1.0, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        results = storage.search("code", query_vec, limit=5)
        assert len(results) == 1
        assert results[0]["symbol_name"] == "authenticate"

    def test_search_docs(self, storage):
        """Should return nearest neighbors for docs search."""
        chunks = [
            Chunk(
                text="Authentication overview",
                collection="docs",
                metadata={"file_path": "auth.md", "section": "Overview"},
            ),
            Chunk(
                text="API reference endpoints",
                collection="docs",
                metadata={"file_path": "api.md", "section": "Endpoints"},
            ),
        ]
        vecs = np.array(
            [[1.0, 0, 0, 0, 0, 0, 0, 0], [0, 1.0, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        storage.add_chunks(chunks, vecs)

        query_vec = np.array([0.9, 0.1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        results = storage.search("docs", query_vec, limit=5)
        assert len(results) == 2
        # First result should be closer to the query
        assert results[0]["section"] == "Overview"

    def test_search_empty_collection(self, storage):
        """Should return empty list when searching empty collection."""
        query_vec = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        results = storage.search("code", query_vec, limit=5)
        assert results == []

    def test_clear_collection(self, storage):
        """Should be able to clear a specific collection."""
        chunks = [
            Chunk(
                text="some code",
                collection="code",
                metadata={
                    "symbol_id": "s1",
                    "symbol_name": "x",
                    "kind": "function",
                    "language": "python",
                    "file_path": "x.py",
                    "start_line": 1,
                    "end_line": 1,
                },
            ),
        ]
        vecs = np.random.randn(1, 8).astype(np.float32)
        storage.add_chunks(chunks, vecs)
        storage.clear_collection("code")

        query_vec = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        results = storage.search("code", query_vec, limit=5)
        assert results == []

    def test_stats(self, storage):
        """Should report collection statistics."""
        chunks = [
            Chunk(
                text="code chunk",
                collection="code",
                metadata={
                    "symbol_id": "s1",
                    "symbol_name": "x",
                    "kind": "function",
                    "language": "python",
                    "file_path": "x.py",
                    "start_line": 1,
                    "end_line": 1,
                },
            ),
        ]
        vecs = np.random.randn(1, 8).astype(np.float32)
        storage.add_chunks(chunks, vecs)

        stats = storage.stats()
        assert stats["code"]["count"] == 1
        assert stats["docs"]["count"] == 0

    def test_filter_by_language(self, storage):
        """Should support language filtering in code search."""
        chunks = [
            Chunk(
                text="python func",
                collection="code",
                metadata={
                    "symbol_id": "s1",
                    "symbol_name": "py_func",
                    "kind": "function",
                    "language": "python",
                    "file_path": "x.py",
                    "start_line": 1,
                    "end_line": 1,
                },
            ),
            Chunk(
                text="rust func",
                collection="code",
                metadata={
                    "symbol_id": "s2",
                    "symbol_name": "rs_func",
                    "kind": "function",
                    "language": "rust",
                    "file_path": "x.rs",
                    "start_line": 1,
                    "end_line": 1,
                },
            ),
        ]
        vecs = np.array(
            [[1.0, 0, 0, 0, 0, 0, 0, 0], [0.9, 0.1, 0, 0, 0, 0, 0, 0]],
            dtype=np.float32,
        )
        storage.add_chunks(chunks, vecs)

        query_vec = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        results = storage.search("code", query_vec, limit=5, where="language = 'python'")
        assert len(results) == 1
        assert results[0]["language"] == "python"
