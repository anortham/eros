"""
Shared test fixtures for Eros.
"""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def julie_db(tmp_path):
    """Create a mock Julie SQLite database with realistic schema and data.

    Returns the path to the SQLite database file.
    """
    db_path = tmp_path / "symbols.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    # Create Julie's schema (mirrors real Julie v3.x)
    cursor.executescript("""
        CREATE TABLE files (
            path TEXT PRIMARY KEY,
            language TEXT NOT NULL,
            hash TEXT NOT NULL,
            size INTEGER NOT NULL,
            last_modified INTEGER NOT NULL,
            last_indexed INTEGER DEFAULT 0,
            parse_cache BLOB,
            symbol_count INTEGER DEFAULT 0,
            content TEXT
        );

        CREATE TABLE symbols (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            language TEXT NOT NULL,
            file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
            signature TEXT,
            start_line INTEGER,
            start_col INTEGER,
            end_line INTEGER,
            end_col INTEGER,
            start_byte INTEGER,
            end_byte INTEGER,
            doc_comment TEXT,
            visibility TEXT,
            code_context TEXT,
            parent_id TEXT REFERENCES symbols(id),
            metadata TEXT,
            file_hash TEXT,
            last_indexed INTEGER DEFAULT 0,
            semantic_group TEXT,
            confidence REAL DEFAULT 1.0,
            content_type TEXT DEFAULT NULL
        );

        CREATE INDEX idx_symbols_name ON symbols(name);
        CREATE INDEX idx_symbols_kind ON symbols(kind);
        CREATE INDEX idx_symbols_file ON symbols(file_path);
    """)

    # Insert sample files
    cursor.executemany(
        "INSERT INTO files (path, language, hash, size, last_modified, content) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "src/auth/handler.py",
                "python",
                "abc123",
                500,
                1700000000,
                'def authenticate(user: str, password: str) -> bool:\n    """Authenticate a user with credentials."""\n    return verify_hash(password, get_stored_hash(user))\n',
            ),
            (
                "src/auth/tokens.py",
                "python",
                "def456",
                300,
                1700000000,
                'def generate_token(user_id: str, expiry: int = 3600) -> str:\n    """Generate a JWT token for authenticated user."""\n    payload = {"sub": user_id, "exp": time.time() + expiry}\n    return jwt.encode(payload, SECRET_KEY)\n',
            ),
            (
                "src/models/user.py",
                "python",
                "ghi789",
                400,
                1700000000,
                'class User:\n    """User model with authentication fields."""\n    def __init__(self, username: str, email: str):\n        self.username = username\n        self.email = email\n        self.is_active = True\n',
            ),
        ],
    )

    # Insert sample symbols
    cursor.executemany(
        """INSERT INTO symbols (id, name, kind, language, file_path, signature,
           start_line, end_line, start_byte, end_byte, doc_comment, visibility)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "sym_auth_1",
                "authenticate",
                "function",
                "python",
                "src/auth/handler.py",
                "def authenticate(user: str, password: str) -> bool",
                1,
                3,
                0,
                150,
                "Authenticate a user with credentials.",
                "public",
            ),
            (
                "sym_token_1",
                "generate_token",
                "function",
                "python",
                "src/auth/tokens.py",
                "def generate_token(user_id: str, expiry: int = 3600) -> str",
                1,
                4,
                0,
                200,
                "Generate a JWT token for authenticated user.",
                "public",
            ),
            (
                "sym_user_1",
                "User",
                "class",
                "python",
                "src/models/user.py",
                "class User",
                1,
                6,
                0,
                250,
                "User model with authentication fields.",
                "public",
            ),
            (
                "sym_user_init",
                "__init__",
                "method",
                "python",
                "src/models/user.py",
                "def __init__(self, username: str, email: str)",
                3,
                6,
                80,
                250,
                None,
                "public",
            ),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def julie_project(tmp_path, julie_db):
    """Create a mock project directory with Julie's .julie/ structure.

    Julie stores data per-project:
      {project_root}/.julie/workspace_registry.json
      {project_root}/.julie/indexes/{workspace_id}/db/symbols.db

    Returns a dict with paths to the project components.
    """
    project_root = tmp_path / "my-project"
    project_root.mkdir()

    workspace_id = "my-project_abc12345"
    julie_dir = project_root / ".julie"
    indexes_dir = julie_dir / "indexes" / workspace_id / "db"
    indexes_dir.mkdir(parents=True)

    # Copy the mock database into the project-local .julie structure
    shutil.copy2(str(julie_db), str(indexes_dir / "symbols.db"))

    # Create workspace registry (matches real Julie v3.x format)
    registry = {
        "version": "1.0",
        "last_updated": 1700000000,
        "primary_workspace": {
            "id": workspace_id,
            "original_path": str(project_root),
            "directory_name": workspace_id,
            "display_name": "my-project",
            "workspace_type": "Primary",
            "created_at": 1700000000,
            "last_accessed": 1700000000,
            "expires_at": None,
            "symbol_count": 4,
            "file_count": 3,
            "index_size_bytes": 50000,
            "status": "Active",
        },
        "reference_workspaces": {},
        "orphaned_indexes": {},
        "config": {
            "default_ttl_seconds": 604800,
            "max_total_size_bytes": 524288000,
            "auto_cleanup_enabled": True,
            "cleanup_interval_seconds": 3600,
        },
        "statistics": {
            "total_workspaces": 1,
            "total_orphans": 0,
            "total_index_size_bytes": 50000,
            "total_symbols": 4,
            "total_files": 3,
            "last_cleanup": 1700000000,
        },
    }
    (julie_dir / "workspace_registry.json").write_text(json.dumps(registry, indent=2))

    return {
        "project_root": project_root,
        "workspace_id": workspace_id,
        "julie_dir": julie_dir,
        "db_path": indexes_dir / "symbols.db",
    }


@pytest.fixture
def sample_docs(tmp_path):
    """Create sample documentation files for testing doc chunking."""
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()

    (docs_dir / "auth.md").write_text(
        """# Authentication

## Overview

The authentication system uses JWT tokens for stateless auth.
Users authenticate with username and password, receiving a token
that expires after a configurable period.

## Token Flow

1. User sends credentials to `/api/login`
2. Server validates credentials against the database
3. Server generates JWT with user ID and expiry
4. Client stores token and includes it in subsequent requests

## Configuration

Set `AUTH_TOKEN_EXPIRY` environment variable to control token lifetime.
Default is 3600 seconds (1 hour).
"""
    )

    (docs_dir / "api.md").write_text(
        """# API Reference

## Endpoints

### POST /api/login

Authenticate a user and receive a JWT token.

**Request Body:**
```json
{"username": "string", "password": "string"}
```

**Response:**
```json
{"token": "string", "expires_in": 3600}
```

### GET /api/users

List all active users. Requires authentication.
"""
    )

    return docs_dir


@pytest.fixture
def ref_julie_db(tmp_path):
    """Create a second mock Julie SQLite database representing a reference workspace.

    Contains Rust symbols from a hypothetical "shared-lib" project to ensure
    they're clearly distinguishable from the primary workspace's Python symbols.
    """
    db_path = tmp_path / "ref_symbols.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE files (
            path TEXT PRIMARY KEY,
            language TEXT NOT NULL,
            hash TEXT NOT NULL,
            size INTEGER NOT NULL,
            last_modified INTEGER NOT NULL,
            last_indexed INTEGER DEFAULT 0,
            parse_cache BLOB,
            symbol_count INTEGER DEFAULT 0,
            content TEXT
        );

        CREATE TABLE symbols (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            language TEXT NOT NULL,
            file_path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
            signature TEXT,
            start_line INTEGER,
            start_col INTEGER,
            end_line INTEGER,
            end_col INTEGER,
            start_byte INTEGER,
            end_byte INTEGER,
            doc_comment TEXT,
            visibility TEXT,
            code_context TEXT,
            parent_id TEXT REFERENCES symbols(id),
            metadata TEXT,
            file_hash TEXT,
            last_indexed INTEGER DEFAULT 0,
            semantic_group TEXT,
            confidence REAL DEFAULT 1.0,
            content_type TEXT DEFAULT NULL
        );

        CREATE INDEX idx_symbols_name ON symbols(name);
        CREATE INDEX idx_symbols_kind ON symbols(kind);
        CREATE INDEX idx_symbols_file ON symbols(file_path);
    """)

    cursor.executemany(
        "INSERT INTO files (path, language, hash, size, last_modified, content) VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "src/lib.rs",
                "rust",
                "rst111",
                600,
                1700000000,
                'pub fn retry<F, T>(f: F, max_attempts: u32) -> Result<T>\nwhere F: Fn() -> Result<T> {\n    for _ in 0..max_attempts { if let Ok(v) = f() { return Ok(v); } }\n    f()\n}\n',
            ),
        ],
    )

    cursor.executemany(
        """INSERT INTO symbols (id, name, kind, language, file_path, signature,
           start_line, end_line, start_byte, end_byte, doc_comment, visibility)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                "sym_retry_1",
                "retry",
                "function",
                "rust",
                "src/lib.rs",
                "pub fn retry<F, T>(f: F, max_attempts: u32) -> Result<T>",
                1,
                5,
                0,
                150,
                "Retry a fallible operation up to max_attempts times.",
                "public",
            ),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def julie_project_with_refs(tmp_path, julie_db, ref_julie_db):
    """Create a project with both primary and reference workspaces.

    Primary workspace: Python symbols (from julie_db)
    Reference workspace: Rust symbols (from ref_julie_db)
    """
    project_root = tmp_path / "multi-ws-project"
    project_root.mkdir()

    primary_id = "multi-ws_primary"
    ref_id = "shared-lib_ref456"

    julie_dir = project_root / ".julie"

    # Set up primary workspace DB
    primary_db_dir = julie_dir / "indexes" / primary_id / "db"
    primary_db_dir.mkdir(parents=True)
    shutil.copy2(str(julie_db), str(primary_db_dir / "symbols.db"))

    # Set up reference workspace DB
    ref_db_dir = julie_dir / "indexes" / ref_id / "db"
    ref_db_dir.mkdir(parents=True)
    shutil.copy2(str(ref_julie_db), str(ref_db_dir / "symbols.db"))

    registry = {
        "version": "1.0",
        "last_updated": 1700000000,
        "primary_workspace": {
            "id": primary_id,
            "original_path": str(project_root),
            "directory_name": primary_id,
            "display_name": "multi-ws",
            "workspace_type": "Primary",
            "created_at": 1700000000,
            "last_accessed": 1700000000,
            "expires_at": None,
            "symbol_count": 4,
            "file_count": 3,
            "index_size_bytes": 50000,
            "status": "Active",
        },
        "reference_workspaces": {
            ref_id: {
                "id": ref_id,
                "original_path": "/some/other/project",
                "directory_name": ref_id,
                "display_name": "shared-lib",
                "workspace_type": "Reference",
                "created_at": 1700000000,
                "last_accessed": 1700000000,
                "expires_at": 1700604800,
                "symbol_count": 1,
                "file_count": 1,
                "index_size_bytes": 10000,
                "status": "Active",
            },
        },
        "orphaned_indexes": {},
        "config": {
            "default_ttl_seconds": 604800,
            "max_total_size_bytes": 524288000,
            "auto_cleanup_enabled": True,
            "cleanup_interval_seconds": 3600,
        },
        "statistics": {
            "total_workspaces": 2,
            "total_orphans": 0,
            "total_index_size_bytes": 60000,
            "total_symbols": 5,
            "total_files": 4,
            "last_cleanup": 1700000000,
        },
    }
    (julie_dir / "workspace_registry.json").write_text(json.dumps(registry, indent=2))

    return {
        "project_root": project_root,
        "primary_id": primary_id,
        "ref_id": ref_id,
        "julie_dir": julie_dir,
    }
