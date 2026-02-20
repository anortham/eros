"""
LanceDB vector store management.

Manages two collections (LanceDB tables):
  - code: Embeddings from code symbols (via code model)
  - docs: Embeddings from documentation (via prose model)

Each collection has its own schema tailored to its metadata needs.
"""

import logging
from pathlib import Path

import lancedb
import numpy as np
import pyarrow as pa

from eros.chunking import Chunk
from eros.config import ErosConfig

logger = logging.getLogger("eros.storage")

# Schema for code chunks
CODE_COLUMNS = [
    ("vector", None),  # Set dynamically based on embedding dims
    ("text", pa.string()),
    ("symbol_id", pa.string()),
    ("symbol_name", pa.string()),
    ("kind", pa.string()),
    ("language", pa.string()),
    ("file_path", pa.string()),
    ("start_line", pa.int32()),
    ("end_line", pa.int32()),
]

# Schema for doc chunks
DOC_COLUMNS = [
    ("vector", None),  # Set dynamically
    ("text", pa.string()),
    ("file_path", pa.string()),
    ("section", pa.string()),
]


def _make_schema(columns: list[tuple], dim: int) -> pa.Schema:
    """Build a PyArrow schema, substituting vector dimension."""
    fields = []
    for name, dtype in columns:
        if name == "vector":
            fields.append(pa.field(name, pa.list_(pa.float32(), dim)))
        else:
            fields.append(pa.field(name, dtype))
    return pa.Schema.from_pylist([], schema=pa.schema(fields)).schema if False else pa.schema(fields)


class VectorStorage:
    """LanceDB-backed vector storage with separate code and docs collections."""

    def __init__(self, config: ErosConfig, embedding_dims: dict[str, int] | None = None):
        """Initialize vector storage.

        Args:
            config: Eros configuration
            embedding_dims: Dict of collection -> embedding dimension.
                            If not provided, tables are created on first add.
        """
        self._config = config
        self._db_path = config.vectors_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self._db_path))
        self._dims = embedding_dims or {}
        self._tables: dict[str, lancedb.table.Table] = {}

        # Open existing tables
        for name in ["code", "docs"]:
            try:
                self._tables[name] = self._db.open_table(name)
            except Exception:
                pass  # Table doesn't exist yet

    def _get_or_create_table(self, collection: str, dim: int) -> lancedb.table.Table:
        """Get existing table or create it with the right schema."""
        if collection in self._tables:
            return self._tables[collection]

        if collection == "code":
            schema = _make_schema(CODE_COLUMNS, dim)
        elif collection == "docs":
            schema = _make_schema(DOC_COLUMNS, dim)
        else:
            raise ValueError(f"Unknown collection: {collection}")

        table = self._db.create_table(collection, schema=schema, mode="overwrite")
        self._tables[collection] = table
        return table

    def add_chunks(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Add chunks with their embedding vectors to the appropriate collection.

        All chunks must belong to the same collection.

        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        collection = chunks[0].collection
        dim = vectors.shape[1]
        table = self._get_or_create_table(collection, dim)

        rows = []
        for chunk, vec in zip(chunks, vectors):
            row = {"vector": vec.tolist(), "text": chunk.text}
            row.update(chunk.metadata)
            # Ensure integer fields are Python ints (not None)
            if collection == "code":
                row["start_line"] = row.get("start_line") or 0
                row["end_line"] = row.get("end_line") or 0
            rows.append(row)

        table.add(rows)
        return len(rows)

    def search(
        self,
        collection: str,
        query_vector: np.ndarray,
        limit: int = 20,
        where: str | None = None,
    ) -> list[dict]:
        """Search a collection by vector similarity.

        Args:
            collection: "code" or "docs"
            query_vector: Query embedding vector
            limit: Maximum results
            where: Optional SQL WHERE clause for filtering (e.g., "language = 'python'")

        Returns list of result dicts with metadata and _distance score.
        """
        if collection not in self._tables:
            return []

        table = self._tables[collection]

        try:
            query = table.search(query_vector.tolist()).limit(limit)
            if where:
                query = query.where(where)
            results = query.to_list()
            return results
        except Exception as e:
            logger.warning("Search failed on collection '%s': %s", collection, e)
            return []

    def clear_collection(self, collection: str):
        """Drop a collection. It will be recreated with the correct schema on next add."""
        self._tables.pop(collection, None)
        try:
            self._db.drop_table(collection)
        except Exception:
            pass

    def stats(self) -> dict:
        """Return statistics for each collection."""
        result = {}
        for name in ["code", "docs"]:
            if name in self._tables:
                try:
                    count = self._tables[name].count_rows()
                    result[name] = {"count": count}
                except Exception:
                    result[name] = {"count": 0}
            else:
                result[name] = {"count": 0}
        return result
