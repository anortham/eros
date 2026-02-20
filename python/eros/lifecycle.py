"""
Lifecycle management for Eros — startup and shutdown hooks.

CRITICAL: The MCP protocol requires servers to respond to the handshake
within ~100ms. Heavy ML imports (torch, sentence-transformers) take 5+ seconds.

Pattern:
  1. lifespan() spawns a background task and yields immediately
  2. Background task loads models via asyncio.to_thread() (non-blocking)
  3. First search request waits for models to be ready

This is the same battle-tested pattern from Miller's lifecycle.py.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger("eros")

# Module-level state — set during background initialization
_initialized = asyncio.Event()
_init_error: Exception | None = None

# These are set after background init completes
embeddings = None  # DualEmbeddingManager instance
storage = None  # LanceDB VectorStorage instance
config = None  # ErosConfig instance


async def wait_for_init():
    """Wait for background initialization to complete. Call before any search."""
    await _initialized.wait()
    if _init_error is not None:
        raise _init_error


async def _background_init():
    """Heavy initialization in background — models loaded via thread pool."""
    global embeddings, storage, config, _init_error

    try:
        # Import config (lightweight, no ML deps)
        from eros.config import ErosConfig

        config = ErosConfig()
        logger.info("Eros config loaded: code_model=%s, docs_model=%s", config.code_model, config.docs_model)

        # Heavy imports via thread pool — does NOT block the event loop
        def _sync_heavy_imports():
            from eros.embeddings import DualEmbeddingManager
            from eros.storage import VectorStorage

            return DualEmbeddingManager, VectorStorage

        DualEmbeddingManager, VectorStorage = await asyncio.to_thread(_sync_heavy_imports)

        # Initialize components
        embeddings = DualEmbeddingManager(config)
        storage = VectorStorage(config)

        logger.info("Eros initialized successfully")

    except Exception as e:
        _init_error = e
        logger.error("Eros initialization failed: %s", e, exc_info=True)
    finally:
        _initialized.set()


@asynccontextmanager
async def lifespan(_app):
    """
    FastMCP lifespan handler.

    CRITICAL: Must yield immediately! The MCP handshake must complete
    within ~100ms. All heavy work happens in the background task.
    """
    logger.info("Spawning background initialization task...")
    init_task = asyncio.create_task(_background_init())
    logger.info("Eros ready for MCP handshake (initialization running in background)")

    yield  # Server is now ready — client sees "Connected" immediately

    # Shutdown
    logger.info("Eros shutting down...")
    if not init_task.done():
        logger.info("Waiting for background initialization to complete...")
        await init_task

    # Unload models to free GPU memory
    if embeddings is not None:
        embeddings.unload()

    logger.info("Eros shutdown complete")
