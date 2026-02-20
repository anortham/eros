"""
Configuration for Eros — model selection, paths, environment variables.

All model choices are configurable via environment variables, making it
easy to A/B test different embedding models for RAG research.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ErosConfig:
    """Immutable configuration loaded once at startup."""

    # --- Embedding Models ---
    # Code model: optimized for source code understanding
    code_model: str = field(
        default_factory=lambda: os.environ.get(
            "EROS_CODE_MODEL", "nomic-ai/CodeRankEmbed"
        )
    )
    # Prose model: optimized for natural language / documentation
    docs_model: str = field(
        default_factory=lambda: os.environ.get("EROS_DOCS_MODEL", "BAAI/bge-small-en-v1.5")
    )
    # Reranker model: cross-encoder for result reranking
    reranker_model: str = field(
        default_factory=lambda: os.environ.get(
            "EROS_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2"
        )
    )

    # --- Paths ---
    # Project root — Julie stores .julie/ inside the project directory.
    # Eros discovers Julie's data at {project_root}/.julie/
    # EROS_WORKSPACE is the primary env var (matches Julie/Goldfish convention).
    # EROS_PROJECT_ROOT is kept as a fallback for backwards compatibility.
    project_root: Path = field(
        default_factory=lambda: Path(
            os.environ.get("EROS_WORKSPACE", os.environ.get("EROS_PROJECT_ROOT", os.getcwd()))
        ).expanduser()
    )
    # Eros's own data directory (for LanceDB vector store)
    eros_data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("EROS_DATA_DIR", ".eros"))
    )

    # --- Chunking ---
    # Max characters per code chunk (symbol body)
    max_code_chunk_chars: int = 4000
    # Max characters per doc chunk (markdown section)
    max_doc_chunk_chars: int = 2000
    # Overlap between doc chunks (characters)
    doc_chunk_overlap: int = 200

    # --- Search ---
    # Default result limit
    default_search_limit: int = 20
    # RRF fusion constant (k parameter in 1/(k+rank))
    rrf_k: int = 60

    # --- Performance ---
    # Compute device: "cpu" (default, safe), "mps", "cuda", or "auto" (detect best)
    device: str = field(
        default_factory=lambda: os.environ.get("EROS_DEVICE", "cpu")
    )
    # Auto-unload models after this many seconds of inactivity
    model_idle_timeout_secs: int = 300
    # Check interval for idle model unloading
    model_check_interval_secs: int = 60

    @property
    def logs_dir(self) -> Path:
        """Path to Eros log directory."""
        return self.eros_data_dir / "logs"

    @property
    def vectors_path(self) -> Path:
        """Path to LanceDB vector store."""
        return self.eros_data_dir / "vectors.lance"

    @property
    def julie_dir(self) -> Path:
        """Path to Julie's project-local data directory."""
        return self.project_root / ".julie"

    @property
    def julie_workspace_registry(self) -> Path:
        """Path to Julie's workspace registry in this project."""
        return self.julie_dir / "workspace_registry.json"

    @property
    def julie_indexes_dir(self) -> Path:
        """Path to Julie's workspace indexes in this project."""
        return self.julie_dir / "indexes"

    def doc_extensions(self) -> frozenset[str]:
        """File extensions treated as documentation."""
        return frozenset({".md", ".txt", ".rst", ".adoc"})
