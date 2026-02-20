"""Tests for LanceDB vector storage — schema and workspace_id support."""

import numpy as np
import pytest

from eros.chunking import Chunk
from eros.config import ErosConfig
from eros.storage import VectorStorage


@pytest.fixture
def storage(tmp_path):
    """Create a VectorStorage backed by a temp directory."""
    config = ErosConfig(
        project_root=tmp_path,
        eros_data_dir=tmp_path / ".eros",
    )
    return VectorStorage(config, embedding_dims={"code": 8, "docs": 8})


def _make_code_chunks(workspace_id: str = "", count: int = 2) -> list[Chunk]:
    """Helper to create code chunks with workspace_id."""
    chunks = []
    for i in range(count):
        chunks.append(
            Chunk(
                text=f"def func_{i}(): pass",
                collection="code",
                metadata={
                    "symbol_id": f"sym_{i}",
                    "symbol_name": f"func_{i}",
                    "kind": "function",
                    "language": "python",
                    "file_path": f"src/mod_{i}.py",
                    "start_line": 1,
                    "end_line": 2,
                    "workspace_id": workspace_id,
                },
            )
        )
    return chunks


def _random_vectors(count: int, dim: int = 8) -> np.ndarray:
    return np.random.randn(count, dim).astype(np.float32)


class TestWorkspaceIdInSchema:
    """Test that workspace_id is stored and filterable in LanceDB."""

    def test_code_table_stores_workspace_id(self, storage):
        """Code chunks should have workspace_id persisted."""
        chunks = _make_code_chunks(workspace_id="ws_primary")
        vecs = _random_vectors(len(chunks))
        storage.add_chunks(chunks, vecs)

        results = storage.search("code", vecs[0], limit=10)
        assert len(results) > 0
        assert results[0]["workspace_id"] == "ws_primary"

    def test_code_table_defaults_empty_workspace_id(self, storage):
        """Chunks without workspace_id should default to empty string."""
        chunks = [
            Chunk(
                text="def no_ws(): pass",
                collection="code",
                metadata={
                    "symbol_id": "sym_x",
                    "symbol_name": "no_ws",
                    "kind": "function",
                    "language": "python",
                    "file_path": "src/x.py",
                    "start_line": 1,
                    "end_line": 2,
                    # No workspace_id key
                },
            )
        ]
        vecs = _random_vectors(1)
        storage.add_chunks(chunks, vecs)

        results = storage.search("code", vecs[0], limit=10)
        assert results[0]["workspace_id"] == ""

    def test_filter_by_workspace_id(self, storage):
        """Should be able to filter search results by workspace_id."""
        ws1_chunks = _make_code_chunks(workspace_id="ws1", count=3)
        ws2_chunks = _make_code_chunks(workspace_id="ws2", count=2)
        all_chunks = ws1_chunks + ws2_chunks
        vecs = _random_vectors(len(all_chunks))

        storage.add_chunks(ws1_chunks, vecs[: len(ws1_chunks)])
        storage.add_chunks(ws2_chunks, vecs[len(ws1_chunks) :])

        # Search with workspace filter
        results = storage.search("code", vecs[0], limit=10, where="workspace_id = 'ws1'")
        assert all(r["workspace_id"] == "ws1" for r in results)

        results = storage.search("code", vecs[0], limit=10, where="workspace_id = 'ws2'")
        assert all(r["workspace_id"] == "ws2" for r in results)

    def test_docs_table_stores_workspace_id(self, storage):
        """Doc chunks should also have workspace_id persisted."""
        chunks = [
            Chunk(
                text="# Authentication\n\nUse JWT tokens.",
                collection="docs",
                metadata={
                    "file_path": "docs/auth.md",
                    "section": "Authentication",
                    "workspace_id": "ws_primary",
                },
            )
        ]
        vecs = _random_vectors(1)
        storage.add_chunks(chunks, vecs)

        results = storage.search("docs", vecs[0], limit=10)
        assert results[0]["workspace_id"] == "ws_primary"
