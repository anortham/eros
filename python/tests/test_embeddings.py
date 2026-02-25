"""Tests for dual-model embedding manager and vector storage."""

import logging
import time
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest
from eros.chunking import Chunk
from eros.config import ErosConfig
from eros.embeddings import DualEmbeddingManager, _ModelSlot, adaptive_batch_size, resolve_device
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


class TestModelSlotTiming:
    """Timing instrumentation in _ModelSlot.load() and encode()."""

    @pytest.fixture
    def loaded_slot(self):
        """A _ModelSlot with a mocked model (already loaded)."""
        slot = _ModelSlot("fake-model", "cpu")
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 8).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 8
        slot._model = mock_model
        slot._dimensions = 8
        slot.last_use_time = time.time()
        return slot

    def test_encode_logs_timing(self, loaded_slot, caplog):
        with caplog.at_level(logging.DEBUG, logger="eros.embeddings"):
            loaded_slot.encode(["a", "b", "c"])
        assert any("Encoded 3 texts" in r.message for r in caplog.records)
        assert any("texts/sec" in r.message for r in caplog.records)

    def test_load_logs_timing(self, caplog, monkeypatch):
        slot = _ModelSlot("fake-model", "cpu")
        mock_instance = MagicMock()
        mock_instance.get_sentence_embedding_dimension.return_value = 8
        mock_cls = MagicMock(return_value=mock_instance)

        monkeypatch.setattr(
            "sentence_transformers.SentenceTransformer",
            mock_cls,
        )

        with caplog.at_level(logging.INFO, logger="eros.embeddings"):
            slot.load()
        assert any("loaded in" in r.message for r in caplog.records)


class TestResolveDevice:
    """Device resolution from config string."""

    def test_cpu_explicit(self):
        device, dtype = resolve_device("cpu")
        assert device == "cpu"
        assert dtype == "cpu"

    def test_auto_returns_something(self):
        device, dtype = resolve_device("auto")
        assert device in ("cpu", "mps", "cuda")
        assert dtype in ("cpu", "mps", "cuda")

    def test_config_default_is_cpu(self):
        config = ErosConfig()
        assert config.device == "cpu"


class TestAdaptiveBatchSizing:
    """Adaptive embedding batch sizing by device type."""

    def test_query_batch_size_is_one(self):
        assert adaptive_batch_size("cuda", 100, target="query") == 1

    def test_cpu_index_batch_size_capped(self):
        assert adaptive_batch_size("cpu", 100, target="index") == 16

    def test_cuda_index_batch_size_capped_by_text_count(self):
        assert adaptive_batch_size("cuda", 10, target="index") == 10

    def test_unknown_device_uses_safe_default(self):
        assert adaptive_batch_size("weird", 100, target="index") == 16


class TestSelfTuningBatching:
    """Runtime self-tuning for indexing batch size."""

    class FakeSlot:
        def __init__(self):
            self.calls = []

        def encode(self, texts, batch_size=32):
            self.calls.append(batch_size)
            # Simulate OOM if batch is too large
            if batch_size > 8:
                raise RuntimeError("CUDA out of memory")
            return np.random.randn(len(texts), 8).astype(np.float32)

    def test_autotune_reduces_batch_on_oom(self, tmp_path):
        manager = DualEmbeddingManager(ErosConfig(device="cpu", eros_data_dir=tmp_path / ".eros"))
        manager.code_model = cast(Any, self.FakeSlot())

        vectors = manager.embed_code([f"x{i}" for i in range(40)])
        state = manager.batch_tuning_state("code", avg_text_len=2)
        calls = cast(Any, manager.code_model).calls

        assert vectors.shape[0] == 40
        assert any(call > 8 for call in calls)
        assert any(call <= 8 for call in calls)
        assert state["batch_size"] <= 12

    def test_autotune_grows_batch_on_successful_large_workload(self, tmp_path):
        manager = DualEmbeddingManager(ErosConfig(device="cpu", eros_data_dir=tmp_path / ".eros"))

        class AlwaysOkSlot:
            def __init__(self):
                self.calls = []

            def encode(self, texts, batch_size=32):
                self.calls.append(batch_size)
                return np.random.randn(len(texts), 8).astype(np.float32)

        manager.code_model = cast(Any, AlwaysOkSlot())
        before = manager.batch_tuning_state("code", avg_text_len=2)["batch_size"]

        manager.embed_code([f"x{i}" for i in range(200)])
        after = manager.batch_tuning_state("code", avg_text_len=2)["batch_size"]

        assert cast(Any, manager.code_model).calls
        assert after >= before

    def test_tuning_persists_across_manager_restarts(self, tmp_path):
        config = ErosConfig(device="cpu", eros_data_dir=tmp_path / ".eros")
        manager1 = DualEmbeddingManager(config)
        manager1.code_model = cast(Any, self.FakeSlot())
        manager1.embed_code([f"x{i}" for i in range(40)])
        tuned = manager1.batch_tuning_state("code", avg_text_len=2)["batch_size"]

        manager2 = DualEmbeddingManager(config)
        loaded = manager2.batch_tuning_state("code", avg_text_len=2)["batch_size"]
        assert loaded == tuned

    def test_tuning_is_bucket_specific(self, tmp_path):
        manager = DualEmbeddingManager(ErosConfig(device="cpu", eros_data_dir=tmp_path / ".eros"))
        manager.code_model = cast(Any, self.FakeSlot())

        manager.embed_code([f"s{i}" for i in range(40)])
        small_bucket = manager.batch_tuning_state("code", avg_text_len=2)["batch_size"]
        medium_bucket = manager.batch_tuning_state("code", avg_text_len=1000)["batch_size"]

        assert small_bucket != medium_bucket
