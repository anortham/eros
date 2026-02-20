"""
Dual-model embedding manager.

Two models, two purposes:
  - Code model (Jina-code-v2 default): Optimized for source code understanding
  - Prose model (BGE-small default): Optimized for natural language / documentation

Each model embeds into its own LanceDB collection, enabling scope-targeted search.

GPU detection and model loading patterns adapted from Miller's battle-tested
EmbeddingManager. Key differences:
  - Two model instances instead of one
  - Models are loaded lazily (first use, not at init)
  - Auto-unload after idle timeout to free GPU memory
"""

import logging
import time

import numpy as np
import torch

from eros.config import ErosConfig

logger = logging.getLogger("eros.embeddings")


def detect_device() -> tuple[str, str]:
    """Detect the best available compute device.

    Returns (device, device_type) tuple.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Using CUDA GPU: %s", gpu_name)
        return "cuda", "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS")
        return "mps", "mps"

    try:
        import torch_directml
        logger.info("Using DirectML")
        return str(torch_directml.device()), "directml"
    except ImportError:
        pass

    logger.info("Using CPU")
    return "cpu", "cpu"


class _ModelSlot:
    """Lazy-loaded embedding model slot with idle tracking."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._dimensions: int | None = None
        self.last_use_time: float | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def dimensions(self) -> int | None:
        return self._dimensions

    def load(self):
        """Load the model (called on first use)."""
        if self._model is not None:
            return

        from sentence_transformers import SentenceTransformer

        logger.info("Loading model '%s' on device '%s'...", self.model_name, self.device)

        # Always pass trust_remote_code — models that don't need it simply ignore it.
        self._model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )

        self._dimensions = self._model.get_sentence_embedding_dimension()
        self.last_use_time = time.time()
        logger.info(
            "Model '%s' loaded: %dD embeddings", self.model_name, self._dimensions
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embedding vectors."""
        self.load()
        self.last_use_time = time.time()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return np.array(embeddings, dtype=np.float32)

    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            logger.info("Unloading model '%s'", self.model_name)
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class DualEmbeddingManager:
    """Manages two embedding models: one for code, one for documentation.

    Models are loaded lazily on first use and auto-unloaded after idle timeout.
    """

    def __init__(self, config: ErosConfig):
        self._config = config
        device, device_type = detect_device()
        self._device = device
        self._device_type = device_type

        self.code_model = _ModelSlot(config.code_model, device)
        self.docs_model = _ModelSlot(config.docs_model, device)

    @property
    def code_dimensions(self) -> int | None:
        """Embedding dimensions for the code model (None if not yet loaded)."""
        return self.code_model.dimensions

    @property
    def docs_dimensions(self) -> int | None:
        """Embedding dimensions for the docs model (None if not yet loaded)."""
        return self.docs_model.dimensions

    def embed_code(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed code texts using the code-optimized model."""
        return self.code_model.encode(texts, batch_size=batch_size)

    def embed_docs(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed documentation texts using the prose-optimized model."""
        return self.docs_model.encode(texts, batch_size=batch_size)

    def embed_query(self, query: str, scope: str) -> np.ndarray:
        """Embed a search query using the appropriate model for the scope.

        Args:
            query: Search query text
            scope: "code", "docs", or "all"

        For "all" scope, returns embeddings from both models as a dict.
        For single scope, returns the embedding vector directly.
        """
        if scope == "code":
            return self.code_model.encode([query])[0]
        elif scope == "docs":
            return self.docs_model.encode([query])[0]
        else:
            raise ValueError(f"Use embed_query_both() for scope='all', got scope='{scope}'")

    def embed_query_both(self, query: str) -> dict[str, np.ndarray]:
        """Embed a query with both models (for scope='all' searches)."""
        return {
            "code": self.code_model.encode([query])[0],
            "docs": self.docs_model.encode([query])[0],
        }

    def unload(self):
        """Unload both models to free GPU memory."""
        self.code_model.unload()
        self.docs_model.unload()

    def check_idle_unload(self):
        """Check if models should be unloaded due to inactivity."""
        timeout = self._config.model_idle_timeout_secs
        now = time.time()
        for slot in [self.code_model, self.docs_model]:
            if slot.is_loaded and slot.last_use_time:
                idle = now - slot.last_use_time
                if idle > timeout:
                    logger.info(
                        "Model '%s' idle for %.0fs (timeout=%ds), unloading",
                        slot.model_name, idle, timeout,
                    )
                    slot.unload()
