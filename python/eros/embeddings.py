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

import json
import logging
import time
from dataclasses import dataclass

import numpy as np
import torch

from eros.config import ErosConfig

logger = logging.getLogger("eros.embeddings")

_INDEX_BATCH_BASE = {
    "cpu": 16,
    "mps": 48,
    "cuda": 128,
    "directml": 32,
}

_INDEX_BATCH_MAX = {
    "cpu": 64,
    "mps": 192,
    "cuda": 512,
    "directml": 128,
}

_LENGTH_BUCKETS = (
    (120, "tiny"),
    (600, "small"),
    (1800, "medium"),
)
_DEFAULT_BUCKET = "large"


def adaptive_batch_size(device_type: str, text_count: int, target: str = "index") -> int:
    """Choose a batch size based on device type and workload size."""
    if target == "query":
        return 1
    if text_count <= 0:
        return 1

    base = _INDEX_BATCH_BASE.get(device_type, 16)
    return max(1, min(base, text_count))


def _is_oom_error(exc: Exception) -> bool:
    """Best-effort detection of OOM errors across backends."""
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda out of memory" in msg or "mps" in msg and "memory" in msg


@dataclass
class _BatchTuningState:
    batch_size: int
    best_tps: float = 0.0


def _length_bucket(avg_chars: float) -> str:
    for threshold, label in _LENGTH_BUCKETS:
        if avg_chars <= threshold:
            return label
    return _DEFAULT_BUCKET


def _auto_detect_device() -> tuple[str, str]:
    """Auto-detect the best available compute device.

    Returns (device, device_type) tuple.
    """
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("Auto-detected CUDA GPU: %s", gpu_name)
        return "cuda", "cuda"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Auto-detected Apple MPS")
        return "mps", "mps"

    try:
        import torch_directml

        logger.info("Auto-detected DirectML")
        return str(torch_directml.device()), "directml"
    except ImportError:
        pass

    logger.info("Using CPU")
    return "cpu", "cpu"


def resolve_device(configured: str) -> tuple[str, str]:
    """Resolve the compute device from config.

    Args:
        configured: Device string from config — "cpu", "mps", "cuda", or "auto".

    Returns (device, device_type) tuple.
    """
    if configured == "auto":
        return _auto_detect_device()
    logger.info("Using device: %s (set EROS_DEVICE=auto to auto-detect)", configured)
    return configured, configured


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
        t0 = time.monotonic()

        # Always pass trust_remote_code — models that don't need it simply ignore it.
        self._model = SentenceTransformer(
            self.model_name, device=self.device, trust_remote_code=True
        )

        elapsed = time.monotonic() - t0
        self._dimensions = self._model.get_sentence_embedding_dimension()
        self.last_use_time = time.time()
        logger.info(
            "Model '%s' loaded in %.1fs: %dD embeddings",
            self.model_name,
            elapsed,
            self._dimensions,
        )

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embedding vectors."""
        self.load()
        self.last_use_time = time.time()
        t0 = time.monotonic()
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        elapsed = time.monotonic() - t0
        logger.debug(
            "Encoded %d texts with '%s' in %.2fs (batch=%d, %.0f texts/sec)",
            len(texts),
            self.model_name,
            elapsed,
            batch_size,
            len(texts) / elapsed if elapsed > 0 else 0,
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
        device, device_type = resolve_device(config.device)
        self._device = device
        self._device_type = device_type

        self.code_model = _ModelSlot(config.code_model, device)
        self.docs_model = _ModelSlot(config.docs_model, device)
        self._batch_tuning = self._init_batch_tuning()
        self._load_batch_tuning()

    def _init_batch_tuning(self) -> dict[str, _BatchTuningState]:
        default = adaptive_batch_size(self._device_type, 10_000, target="index")
        states: dict[str, _BatchTuningState] = {}
        bucket_names = [label for _, label in _LENGTH_BUCKETS] + [_DEFAULT_BUCKET]
        for scope in ("code", "docs"):
            for bucket in bucket_names:
                states[f"{scope}:{bucket}"] = _BatchTuningState(batch_size=default)
        return states

    def _load_batch_tuning(self) -> None:
        path = self._config.batch_tuning_path
        if not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(raw, dict):
            return
        states = raw.get("states", {})
        if not isinstance(states, dict):
            return
        ceiling = self._scope_batch_ceiling()
        for key, value in states.items():
            if key not in self._batch_tuning or not isinstance(value, dict):
                continue
            batch = value.get("batch_size")
            best_tps = value.get("best_tps")
            if isinstance(batch, int):
                self._batch_tuning[key].batch_size = max(1, min(batch, ceiling))
            if isinstance(best_tps, (int, float)):
                self._batch_tuning[key].best_tps = float(max(0.0, best_tps))

    def _save_batch_tuning(self) -> None:
        path = self._config.batch_tuning_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "device_type": self._device_type,
            "states": {
                key: {
                    "batch_size": value.batch_size,
                    "best_tps": round(value.best_tps, 6),
                }
                for key, value in self._batch_tuning.items()
            },
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def _state_key(self, scope: str, avg_text_len: float) -> str:
        return f"{scope}:{_length_bucket(avg_text_len)}"

    def _state_for(self, scope: str, avg_text_len: float) -> _BatchTuningState:
        key = self._state_key(scope, avg_text_len)
        if key not in self._batch_tuning:
            self._batch_tuning[key] = _BatchTuningState(
                batch_size=adaptive_batch_size(self._device_type, 10_000, target="index")
            )
        return self._batch_tuning[key]

    def _avg_text_len(self, texts: list[str]) -> float:
        return (sum(len(text) for text in texts) / len(texts)) if texts else 0.0

    @property
    def code_dimensions(self) -> int | None:
        """Embedding dimensions for the code model (None if not yet loaded)."""
        return self.code_model.dimensions

    @property
    def docs_dimensions(self) -> int | None:
        """Embedding dimensions for the docs model (None if not yet loaded)."""
        return self.docs_model.dimensions

    def index_batch_size(self, scope: str, text_count: int, avg_text_len: float = 0.0) -> int:
        """Batch size for indexing based on current compute device."""
        state = self._state_for(scope, avg_text_len)
        return max(1, min(state.batch_size, text_count))

    def batch_tuning_state(self, scope: str, avg_text_len: float = 0.0) -> dict[str, float | int]:
        """Expose current batch tuning state for diagnostics/tests."""
        state = self._state_for(scope, avg_text_len)
        return {
            "batch_size": state.batch_size,
            "best_tps": state.best_tps,
        }

    def _scope_batch_ceiling(self) -> int:
        return _INDEX_BATCH_MAX.get(self._device_type, 64)

    def _autotune_after_success(
        self, state: _BatchTuningState, text_count: int, elapsed: float, batch_size: int
    ) -> None:
        throughput = text_count / elapsed if elapsed > 0 else 0.0
        ceiling = self._scope_batch_ceiling()

        # If workload is larger than current batch and performance doesn't regress, explore bigger batches.
        if text_count >= batch_size * 2 and batch_size < ceiling:
            if state.best_tps == 0.0 or throughput >= state.best_tps * 0.95:
                state.batch_size = min(ceiling, max(batch_size + 1, int(batch_size * 1.25)))

        # If throughput regresses badly, back off slightly.
        if state.best_tps > 0 and throughput < state.best_tps * 0.65 and state.batch_size > 1:
            state.batch_size = max(1, int(state.batch_size * 0.8))

        state.best_tps = max(state.best_tps, throughput)
        self._save_batch_tuning()

    def _embed_with_autotune(
        self,
        scope: str,
        slot: _ModelSlot,
        texts: list[str],
        initial_batch_size: int | None = None,
    ) -> np.ndarray:
        avg_len = self._avg_text_len(texts)
        state = self._state_for(scope, avg_len)
        batch = max(1, min(state.batch_size, len(texts)))
        if initial_batch_size is not None:
            batch = max(1, min(initial_batch_size, len(texts)))

        while True:
            started = time.monotonic()
            try:
                vectors = slot.encode(texts, batch_size=batch)
            except RuntimeError as exc:
                if not _is_oom_error(exc) or batch == 1:
                    raise
                batch = max(1, batch // 2)
                state.batch_size = batch
                self._save_batch_tuning()
                logger.warning(
                    "OOM while embedding %s; retrying with batch=%d",
                    scope,
                    batch,
                )
                continue

            elapsed = time.monotonic() - started
            self._autotune_after_success(state, len(texts), elapsed, batch)
            return vectors

    def embed_code(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed code texts using the code-optimized model."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._embed_with_autotune(
            "code",
            self.code_model,
            texts,
            initial_batch_size=batch_size,
        )

    def embed_docs(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Embed documentation texts using the prose-optimized model."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return self._embed_with_autotune(
            "docs",
            self.docs_model,
            texts,
            initial_batch_size=batch_size,
        )

    def embed_query(self, query: str, scope: str) -> np.ndarray:
        """Embed a search query using the appropriate model for the scope.

        Args:
            query: Search query text
            scope: "code", "docs", or "all"

        For "all" scope, returns embeddings from both models as a dict.
        For single scope, returns the embedding vector directly.
        """
        if scope == "code":
            return self.code_model.encode(
                [query], batch_size=adaptive_batch_size(self._device_type, 1, target="query")
            )[0]
        elif scope == "docs":
            return self.docs_model.encode(
                [query], batch_size=adaptive_batch_size(self._device_type, 1, target="query")
            )[0]
        else:
            raise ValueError(f"Use embed_query_both() for scope='all', got scope='{scope}'")

    def embed_query_both(self, query: str) -> dict[str, np.ndarray]:
        """Embed a query with both models (for scope='all' searches)."""
        query_batch = adaptive_batch_size(self._device_type, 1, target="query")
        return {
            "code": self.code_model.encode([query], batch_size=query_batch)[0],
            "docs": self.docs_model.encode([query], batch_size=query_batch)[0],
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
                        slot.model_name,
                        idle,
                        timeout,
                    )
                    slot.unload()
