"""Incremental indexing state and change detection helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileDelta:
    """Changed and removed file sets for a workspace scope."""

    changed: set[str]
    removed: set[str]


def sha1_text(text: str) -> str:
    """Compute a deterministic SHA1 hash for text content."""
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


class IndexManifest:
    """Persistent manifest tracking previously indexed file hashes."""

    def __init__(self, file_path: Path):
        self._file_path = file_path
        self._data = {
            "version": 1,
            "code": {},
            "docs": {},
        }
        self._load()

    def _load(self) -> None:
        if not self._file_path.exists():
            return

        try:
            raw = json.loads(self._file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(raw, dict):
            return

        self._data["code"] = raw.get("code", {}) if isinstance(raw.get("code", {}), dict) else {}
        self._data["docs"] = raw.get("docs", {}) if isinstance(raw.get("docs", {}), dict) else {}

    def save(self) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )

    def clear(self, scope: str, workspace_id: str | None = None) -> None:
        if workspace_id is None:
            self._data[scope] = {}
            return
        self._data[scope].pop(workspace_id, None)

    def code_delta(self, workspace_id: str, current_hashes: dict[str, str]) -> FileDelta:
        return self._compute_delta("code", workspace_id, current_hashes)

    def docs_delta(self, workspace_id: str, current_hashes: dict[str, str]) -> FileDelta:
        return self._compute_delta("docs", workspace_id, current_hashes)

    def update_code(self, workspace_id: str, current_hashes: dict[str, str]) -> None:
        self._data["code"][workspace_id] = dict(current_hashes)

    def update_docs(self, workspace_id: str, current_hashes: dict[str, str]) -> None:
        self._data["docs"][workspace_id] = dict(current_hashes)

    def _compute_delta(
        self, scope: str, workspace_id: str, current_hashes: dict[str, str]
    ) -> FileDelta:
        previous = self._data.get(scope, {}).get(workspace_id, {})
        if not isinstance(previous, dict):
            previous = {}

        changed = {path for path, digest in current_hashes.items() if previous.get(path) != digest}
        removed = set(previous.keys()) - set(current_hashes.keys())
        return FileDelta(changed=changed, removed=removed)


def sql_quote(value: str) -> str:
    """Safely quote a string literal for LanceDB SQL WHERE expressions."""
    return value.replace("'", "''")


def where_workspace_and_paths(workspace_id: str, file_paths: set[str]) -> str:
    """Build a WHERE clause for workspace and file path membership."""
    ws = sql_quote(workspace_id)
    if not file_paths:
        return f"workspace_id = '{ws}'"

    quoted = ", ".join(f"'{sql_quote(path)}'" for path in sorted(file_paths))
    return f"workspace_id = '{ws}' AND file_path IN ({quoted})"
