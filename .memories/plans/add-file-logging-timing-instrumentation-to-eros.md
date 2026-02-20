---
id: add-file-logging-timing-instrumentation-to-eros
title: Add File Logging + Timing Instrumentation to Eros
status: active
created: 2026-02-20T20:29:35.902Z
updated: 2026-02-20T20:29:35.902Z
tags:
  - logging
  - timing
  - performance
  - instrumentation
---

# Add File Logging + Timing Instrumentation to Eros

## Steps
1. `config.py` — Add `logs_dir` property (TDD)
2. `server.py` — File logging setup with RotatingFileHandler (TDD)
3. `embeddings.py` — Timing on load() and encode() (TDD)
4. `retrieval.py` — Timing in `_build_index()` output (TDD)

## Key Files
- python/eros/config.py
- python/eros/server.py
- python/eros/embeddings.py
- python/eros/retrieval.py
- python/tests/test_logging.py (new)
- python/tests/test_embeddings.py
- python/tests/test_retrieval.py

## Verification
1. `pytest python/tests/ -v` — all pass
2. Restart Eros, `semantic_index(operation="index")` — output includes timing
3. `.eros/logs/eros.log` exists with DEBUG entries
