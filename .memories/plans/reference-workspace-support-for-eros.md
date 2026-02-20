---
id: reference-workspace-support-for-eros
title: Reference Workspace Support for Eros
status: completed
created: 2026-02-20T19:54:54.715Z
updated: 2026-02-20T20:11:13.623Z
tags:
  - feature
  - reference-workspaces
  - multi-workspace
---

# Reference Workspace Support for Eros

## Steps
1. JulieReader — Multi-Workspace Discovery (TDD)
2. Chunk Metadata — workspace_id (TDD)
3. Storage Schema — workspace_id Column (TDD)
4. Indexing Pipeline — Index All Workspaces (TDD)
5. Search Pipeline — Workspace Filtering (TDD)
6. MCP Tools — Wire workspace Param
7. Agent Instructions + Migration

## Key Files
- python/eros/julie_reader.py
- python/eros/chunking.py
- python/eros/storage.py
- python/eros/retrieval.py
- python/eros/server.py
- python/eros/lifecycle.py
- python/tests/conftest.py + test files

## Full plan at /Users/murphy/.claude/plans/rippling-knitting-chipmunk.md
