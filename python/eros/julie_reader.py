"""
Julie Reader — discover and read Julie's project-local SQLite databases.

Julie stores indexed data per-project:
  {project_root}/.julie/workspace_registry.json   — workspace metadata
  {project_root}/.julie/indexes/{id}/db/symbols.db — symbols + files

Key tables:
  - symbols: Code symbols (functions, classes, methods) with signatures, byte offsets
  - files: Source files with language, content, and metadata

We NEVER write to Julie's databases — read-only access.
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceInfo:
    """Metadata about a Julie workspace (primary or reference)."""

    id: str
    display_name: str
    workspace_type: str  # "Primary" or "Reference"
    db_path: Path


@dataclass
class JulieSymbol:
    """A code symbol extracted from Julie's SQLite database."""

    id: str
    name: str
    kind: str
    language: str
    file_path: str
    signature: str | None
    start_line: int | None
    end_line: int | None
    start_byte: int | None
    end_byte: int | None
    doc_comment: str | None
    visibility: str | None
    parent_id: str | None

    def extract_body(self, file_content: str) -> str:
        """Extract this symbol's body text from the file content using byte offsets.

        Falls back to the full content if byte offsets are missing.
        """
        if not file_content:
            return ""
        if self.start_byte is not None and self.end_byte is not None:
            content_bytes = file_content.encode("utf-8", errors="replace")
            return content_bytes[self.start_byte : self.end_byte].decode("utf-8", errors="replace")
        return file_content


class JulieReader:
    """Read-only access to Julie's per-project SQLite databases.

    Supports primary and reference workspaces. Discovery reads the workspace
    registry; reference workspaces with missing databases are silently skipped.

    Usage:
        reader = JulieReader(Path("/path/to/project"))
        symbols = reader.read_symbols()
        contents = reader.read_all_file_contents()

        # Multi-workspace:
        for ws in reader.list_workspaces():
            symbols = reader.read_symbols(workspace_id=ws.id)
    """

    def __init__(self, project_root: Path):
        self._project_root = Path(project_root)
        self._julie_dir = self._project_root / ".julie"

        if not self._julie_dir.is_dir():
            raise FileNotFoundError(
                f"No .julie directory found at {self._project_root}. "
                f"Is Julie installed and has it indexed this project?"
            )

        registry_path = self._julie_dir / "workspace_registry.json"
        if not registry_path.exists():
            raise FileNotFoundError(
                f"No workspace_registry.json found in {self._julie_dir}. "
                f"Julie may not have indexed this project yet."
            )

        registry = json.loads(registry_path.read_text())
        primary = registry.get("primary_workspace")
        if not primary:
            raise ValueError(
                f"No primary workspace found in {registry_path}. "
                f"Julie may not have indexed this project yet."
            )

        self._workspace_id = primary["directory_name"]
        self._db_path = self._workspace_db_path(self._workspace_id)

        if not self._db_path.exists():
            raise FileNotFoundError(
                f"Julie database not found at {self._db_path}. "
                f"Workspace '{self._workspace_id}' may need re-indexing."
            )

        # Build primary WorkspaceInfo
        self._primary = WorkspaceInfo(
            id=self._workspace_id,
            display_name=primary.get("display_name", self._workspace_id),
            workspace_type="Primary",
            db_path=self._db_path,
        )

        # Discover valid reference workspaces
        self._references: list[WorkspaceInfo] = []
        for ref_id, ref_data in registry.get("reference_workspaces", {}).items():
            dir_name = ref_data.get("directory_name", ref_id)
            ref_db = self._workspace_db_path(dir_name)
            if ref_db.exists():
                self._references.append(
                    WorkspaceInfo(
                        id=dir_name,
                        display_name=ref_data.get("display_name", dir_name),
                        workspace_type="Reference",
                        db_path=ref_db,
                    )
                )
            else:
                logger.warning(
                    "Reference workspace '%s' has no database at %s — skipping",
                    dir_name,
                    ref_db,
                )

    def _workspace_db_path(self, workspace_id: str) -> Path:
        return self._julie_dir / "indexes" / workspace_id / "db" / "symbols.db"

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    @property
    def db_path(self) -> Path:
        return self._db_path

    def list_workspaces(self) -> list[WorkspaceInfo]:
        """Return all valid workspaces (primary + references with existing DBs)."""
        return [self._primary] + self._references

    def resolve_workspace(self, workspace: str) -> list[WorkspaceInfo]:
        """Resolve a workspace selector to a list of WorkspaceInfo.

        Args:
            workspace: "primary", "all", or a specific workspace ID.

        Returns:
            List of matching WorkspaceInfo objects.

        Raises:
            ValueError: If a specific workspace ID is not found.
        """
        if workspace == "primary":
            return [self._primary]
        if workspace == "all":
            return self.list_workspaces()

        # Specific workspace ID
        for ws in self.list_workspaces():
            if ws.id == workspace:
                return [ws]
        raise ValueError(
            f"Workspace '{workspace}' not found. "
            f"Available: {[ws.id for ws in self.list_workspaces()]}"
        )

    def _connect(self, workspace_id: str | None = None) -> sqlite3.Connection:
        """Open a read-only connection to a workspace's SQLite database."""
        if workspace_id and workspace_id != self._workspace_id:
            db_path = self._workspace_db_path(workspace_id)
        else:
            db_path = self._db_path
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def read_symbols(
        self,
        kinds: list[str] | None = None,
        exclude_kinds: list[str] | None = None,
        languages: list[str] | None = None,
        file_paths: list[str] | None = None,
        workspace_id: str | None = None,
    ) -> list[JulieSymbol]:
        """Read symbols from Julie's database with optional filtering.

        Args:
            kinds: Only include these symbol kinds (e.g., ["function", "class"])
            exclude_kinds: Exclude these symbol kinds (e.g., ["import", "variable"])
            languages: Only include these languages (e.g., ["python", "typescript"])
            file_paths: Only include symbols from these file paths
            workspace_id: Read from this workspace's DB (default: primary)
        """
        query = """
            SELECT id, name, kind, language, file_path, signature,
                   start_line, end_line, start_byte, end_byte,
                   doc_comment, visibility, parent_id
            FROM symbols
            WHERE 1=1
        """
        params: list = []

        if kinds:
            placeholders = ", ".join("?" for _ in kinds)
            query += f" AND kind IN ({placeholders})"
            params.extend(kinds)

        if exclude_kinds:
            placeholders = ", ".join("?" for _ in exclude_kinds)
            query += f" AND kind NOT IN ({placeholders})"
            params.extend(exclude_kinds)

        if languages:
            placeholders = ", ".join("?" for _ in languages)
            query += f" AND language IN ({placeholders})"
            params.extend(languages)

        if file_paths:
            placeholders = ", ".join("?" for _ in file_paths)
            query += f" AND file_path IN ({placeholders})"
            params.extend(file_paths)

        conn = self._connect(workspace_id)
        try:
            rows = conn.execute(query, params).fetchall()
            return [
                JulieSymbol(
                    id=row["id"],
                    name=row["name"],
                    kind=row["kind"],
                    language=row["language"],
                    file_path=row["file_path"],
                    signature=row["signature"],
                    start_line=row["start_line"],
                    end_line=row["end_line"],
                    start_byte=row["start_byte"],
                    end_byte=row["end_byte"],
                    doc_comment=row["doc_comment"],
                    visibility=row["visibility"],
                    parent_id=row["parent_id"],
                )
                for row in rows
            ]
        finally:
            conn.close()

    def read_file_content(self, file_path: str, workspace_id: str | None = None) -> str | None:
        """Read the stored content for a specific file.

        Returns None if the file is not found in Julie's database.
        """
        conn = self._connect(workspace_id)
        try:
            row = conn.execute("SELECT content FROM files WHERE path = ?", (file_path,)).fetchone()
            return row["content"] if row else None
        finally:
            conn.close()

    def read_file_contents(
        self, file_paths: list[str], workspace_id: str | None = None
    ) -> dict[str, str]:
        """Read content for a selected set of files.

        Returns a dict of file_path -> content for found files.
        """
        if not file_paths:
            return {}

        placeholders = ", ".join("?" for _ in file_paths)
        conn = self._connect(workspace_id)
        try:
            rows = conn.execute(
                f"SELECT path, content FROM files WHERE content IS NOT NULL AND path IN ({placeholders})",
                file_paths,
            ).fetchall()
            return {row["path"]: row["content"] for row in rows}
        finally:
            conn.close()

    def read_all_file_contents(self, workspace_id: str | None = None) -> dict[str, str]:
        """Read content for all files in Julie's database.

        Returns a dict of file_path -> content.
        """
        conn = self._connect(workspace_id)
        try:
            rows = conn.execute(
                "SELECT path, content FROM files WHERE content IS NOT NULL"
            ).fetchall()
            return {row["path"]: row["content"] for row in rows}
        finally:
            conn.close()

    def read_file_hashes(self, workspace_id: str | None = None) -> dict[str, str]:
        """Read file hashes from Julie's files table.

        Returns a dict of file_path -> hash.
        """
        conn = self._connect(workspace_id)
        try:
            rows = conn.execute("SELECT path, hash FROM files").fetchall()
            return {row["path"]: row["hash"] for row in rows}
        finally:
            conn.close()
