"""Tests for Julie Reader — discovering and reading Julie's SQLite databases."""


import pytest
from eros.julie_reader import JulieReader


class TestJulieDiscovery:
    """Test discovering Julie's workspace in a project directory."""

    def test_discover_primary_workspace(self, julie_project):
        """Should find the primary workspace from workspace_registry.json."""
        reader = JulieReader(julie_project["project_root"])
        assert reader.workspace_id == julie_project["workspace_id"]
        assert reader.db_path.exists()

    def test_discover_fails_without_julie_dir(self, tmp_path):
        """Should raise clear error when no .julie directory exists."""
        with pytest.raises(FileNotFoundError, match=r"\.julie"):
            JulieReader(tmp_path)

    def test_discover_fails_with_empty_registry(self, tmp_path):
        """Should raise clear error when workspace_registry.json is missing."""
        julie_dir = tmp_path / ".julie"
        julie_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="workspace_registry"):
            JulieReader(tmp_path)

    def test_discover_fails_with_no_primary_workspace(self, tmp_path):
        """Should raise error when registry exists but has no primary workspace."""
        julie_dir = tmp_path / ".julie"
        julie_dir.mkdir()
        (julie_dir / "workspace_registry.json").write_text('{"version": "1.0"}')
        with pytest.raises(ValueError, match="primary"):
            JulieReader(tmp_path)


class TestReadSymbols:
    """Test reading symbols from Julie's SQLite database."""

    def test_read_all_symbols(self, julie_project):
        """Should return all symbols from the database."""
        reader = JulieReader(julie_project["project_root"])
        symbols = reader.read_symbols()
        assert len(symbols) == 4  # authenticate, generate_token, User, __init__

    def test_symbol_fields(self, julie_project):
        """Each symbol should have the expected fields populated."""
        reader = JulieReader(julie_project["project_root"])
        symbols = reader.read_symbols()
        auth = next(s for s in symbols if s.name == "authenticate")
        assert auth.kind == "function"
        assert auth.language == "python"
        assert auth.file_path == "src/auth/handler.py"
        assert "user: str" in auth.signature
        assert auth.doc_comment == "Authenticate a user with credentials."
        assert auth.start_byte == 0
        assert auth.end_byte == 150

    def test_filter_by_kind(self, julie_project):
        """Should filter symbols by kind."""
        reader = JulieReader(julie_project["project_root"])
        functions = reader.read_symbols(kinds=["function"])
        assert len(functions) == 2
        assert all(s.kind == "function" for s in functions)

    def test_filter_by_language(self, julie_project):
        """Should filter symbols by language."""
        reader = JulieReader(julie_project["project_root"])
        py_symbols = reader.read_symbols(languages=["python"])
        assert len(py_symbols) == 4  # All are Python in our fixture

    def test_exclude_kinds(self, julie_project):
        """Should be able to exclude certain symbol kinds (e.g., imports)."""
        reader = JulieReader(julie_project["project_root"])
        no_methods = reader.read_symbols(exclude_kinds=["method"])
        assert len(no_methods) == 3
        assert not any(s.kind == "method" for s in no_methods)


class TestReadFileContent:
    """Test reading file content from Julie's files table."""

    def test_read_file_content(self, julie_project):
        """Should return file content for a given path."""
        reader = JulieReader(julie_project["project_root"])
        content = reader.read_file_content("src/auth/handler.py")
        assert "def authenticate" in content

    def test_read_file_content_not_found(self, julie_project):
        """Should return None for nonexistent file."""
        reader = JulieReader(julie_project["project_root"])
        content = reader.read_file_content("nonexistent.py")
        assert content is None

    def test_read_all_file_contents(self, julie_project):
        """Should return a dict of path -> content for all files."""
        reader = JulieReader(julie_project["project_root"])
        contents = reader.read_all_file_contents()
        assert len(contents) == 3
        assert "src/auth/handler.py" in contents
        assert "def authenticate" in contents["src/auth/handler.py"]


class TestSymbolWithBody:
    """Test extracting symbol body text from file content."""

    def test_extract_body_from_content(self, julie_project):
        """Should extract the symbol body using byte offsets."""
        reader = JulieReader(julie_project["project_root"])
        symbols = reader.read_symbols()
        contents = reader.read_all_file_contents()

        auth = next(s for s in symbols if s.name == "authenticate")
        body = auth.extract_body(contents.get(auth.file_path, ""))
        assert "def authenticate" in body
        assert "verify_hash" in body


class TestMultiWorkspaceDiscovery:
    """Test discovering and resolving multiple workspaces (primary + references)."""

    def test_list_workspaces_primary_only(self, julie_project):
        """With no references, list_workspaces returns just the primary."""
        reader = JulieReader(julie_project["project_root"])
        workspaces = reader.list_workspaces()
        assert len(workspaces) == 1
        assert workspaces[0].id == julie_project["workspace_id"]
        assert workspaces[0].workspace_type == "Primary"

    def test_list_workspaces_with_refs(self, julie_project_with_refs):
        """With references, list_workspaces returns primary + valid references."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        workspaces = reader.list_workspaces()
        assert len(workspaces) == 2
        ids = {ws.id for ws in workspaces}
        assert julie_project_with_refs["primary_id"] in ids
        assert julie_project_with_refs["ref_id"] in ids

    def test_list_workspaces_skips_missing_db(self, julie_project_with_refs):
        """References with missing DBs are silently skipped."""
        # Remove the reference workspace's DB
        ref_id = julie_project_with_refs["ref_id"]
        ref_db = (
            julie_project_with_refs["julie_dir"]
            / "indexes"
            / ref_id
            / "db"
            / "symbols.db"
        )
        ref_db.unlink()

        reader = JulieReader(julie_project_with_refs["project_root"])
        workspaces = reader.list_workspaces()
        assert len(workspaces) == 1
        assert workspaces[0].workspace_type == "Primary"

    def test_resolve_primary(self, julie_project_with_refs):
        """resolve_workspace('primary') returns only the primary workspace."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        workspaces = reader.resolve_workspace("primary")
        assert len(workspaces) == 1
        assert workspaces[0].id == julie_project_with_refs["primary_id"]

    def test_resolve_all(self, julie_project_with_refs):
        """resolve_workspace('all') returns primary + all valid references."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        workspaces = reader.resolve_workspace("all")
        assert len(workspaces) == 2

    def test_resolve_specific_ref(self, julie_project_with_refs):
        """resolve_workspace(ref_id) returns just that reference workspace."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        ref_id = julie_project_with_refs["ref_id"]
        workspaces = reader.resolve_workspace(ref_id)
        assert len(workspaces) == 1
        assert workspaces[0].id == ref_id
        assert workspaces[0].workspace_type == "Reference"

    def test_resolve_unknown_workspace_raises(self, julie_project_with_refs):
        """resolve_workspace with unknown ID raises ValueError."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        with pytest.raises(ValueError, match="not-a-workspace"):
            reader.resolve_workspace("not-a-workspace")

    def test_workspace_info_has_db_path(self, julie_project_with_refs):
        """Each WorkspaceInfo should have a valid db_path."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        for ws in reader.list_workspaces():
            assert ws.db_path.exists(), f"DB missing for {ws.id}"


class TestReadFromReferenceWorkspace:
    """Test reading symbols and files from reference workspace databases."""

    def test_read_symbols_from_reference(self, julie_project_with_refs):
        """Should read symbols from a reference workspace's DB."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        ref_id = julie_project_with_refs["ref_id"]
        symbols = reader.read_symbols(workspace_id=ref_id)
        assert len(symbols) == 1
        assert symbols[0].name == "retry"
        assert symbols[0].language == "rust"

    def test_read_symbols_default_is_primary(self, julie_project_with_refs):
        """read_symbols() without workspace_id reads from primary."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        symbols = reader.read_symbols()
        assert len(symbols) == 4  # Primary has 4 Python symbols
        assert all(s.language == "python" for s in symbols)

    def test_read_file_content_from_reference(self, julie_project_with_refs):
        """Should read file content from a reference workspace's DB."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        ref_id = julie_project_with_refs["ref_id"]
        content = reader.read_file_content("src/lib.rs", workspace_id=ref_id)
        assert "pub fn retry" in content

    def test_read_all_file_contents_from_reference(self, julie_project_with_refs):
        """Should read all file contents from reference workspace."""
        reader = JulieReader(julie_project_with_refs["project_root"])
        ref_id = julie_project_with_refs["ref_id"]
        contents = reader.read_all_file_contents(workspace_id=ref_id)
        assert len(contents) == 1
        assert "src/lib.rs" in contents
