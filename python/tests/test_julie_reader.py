"""Tests for Julie Reader — discovering and reading Julie's SQLite databases."""

import pytest

from eros.julie_reader import JulieReader, JulieSymbol


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
