"""Tests for code and documentation chunking strategies."""

from pathlib import Path

from eros.chunking import (
    chunk_code_symbols,
    chunk_doc_file,
    chunk_doc_files,
    chunk_documentation,
    discover_doc_sources,
)
from eros.julie_reader import JulieSymbol


class TestCodeChunking:
    """Test creating code chunks from Julie symbols."""

    def test_chunk_per_symbol(self):
        """Each symbol should produce one chunk."""
        symbols = [
            JulieSymbol(
                id="sym1",
                name="authenticate",
                kind="function",
                language="python",
                file_path="src/auth.py",
                signature="def authenticate(user: str, password: str) -> bool",
                start_line=1,
                end_line=5,
                start_byte=0,
                end_byte=100,
                doc_comment="Authenticate a user.",
                visibility="public",
                parent_id=None,
            ),
            JulieSymbol(
                id="sym2",
                name="User",
                kind="class",
                language="python",
                file_path="src/models.py",
                signature="class User",
                start_line=1,
                end_line=10,
                start_byte=0,
                end_byte=200,
                doc_comment="User model.",
                visibility="public",
                parent_id=None,
            ),
        ]
        file_contents = {
            "src/auth.py": "def authenticate(user: str, password: str) -> bool:\n    return check(user, password)\n",
            "src/models.py": "class User:\n    def __init__(self):\n        pass\n",
        }
        chunks = chunk_code_symbols(symbols, file_contents)
        assert len(chunks) == 2

    def test_chunk_text_includes_signature_and_body(self):
        """Chunk text should include the signature and body."""
        symbols = [
            JulieSymbol(
                id="sym1",
                name="greet",
                kind="function",
                language="python",
                file_path="src/hello.py",
                signature="def greet(name: str) -> str",
                start_line=1,
                end_line=2,
                start_byte=0,
                end_byte=60,
                doc_comment="Say hello.",
                visibility="public",
                parent_id=None,
            ),
        ]
        file_contents = {
            "src/hello.py": 'def greet(name: str) -> str:\n    return f"Hello, {name}!"\n',
        }
        chunks = chunk_code_symbols(symbols, file_contents)
        chunk = chunks[0]
        assert "def greet(name: str) -> str" in chunk.text
        assert chunk.metadata["symbol_name"] == "greet"
        assert chunk.metadata["kind"] == "function"
        assert chunk.metadata["language"] == "python"
        assert chunk.metadata["file_path"] == "src/hello.py"
        assert chunk.collection == "code"

    def test_chunk_includes_doc_comment(self):
        """Doc comment should be included in the chunk text."""
        symbols = [
            JulieSymbol(
                id="sym1",
                name="validate",
                kind="function",
                language="python",
                file_path="src/val.py",
                signature="def validate(data: dict) -> bool",
                start_line=1,
                end_line=3,
                start_byte=0,
                end_byte=80,
                doc_comment="Validate input data against schema.",
                visibility="public",
                parent_id=None,
            ),
        ]
        file_contents = {"src/val.py": "def validate(data: dict) -> bool:\n    return True\n"}
        chunks = chunk_code_symbols(symbols, file_contents)
        assert "Validate input data against schema" in chunks[0].text

    def test_chunk_without_file_content(self):
        """Should still create a chunk from signature + doc_comment when no body available."""
        symbols = [
            JulieSymbol(
                id="sym1",
                name="mystery",
                kind="function",
                language="rust",
                file_path="src/lib.rs",
                signature="fn mystery() -> bool",
                start_line=1,
                end_line=5,
                start_byte=0,
                end_byte=100,
                doc_comment="Does something mysterious.",
                visibility="public",
                parent_id=None,
            ),
        ]
        # No file content available for this file
        chunks = chunk_code_symbols(symbols, {})
        assert len(chunks) == 1
        assert "fn mystery() -> bool" in chunks[0].text
        assert "Does something mysterious" in chunks[0].text

    def test_code_chunks_carry_workspace_id(self):
        """Code chunks should carry workspace_id in metadata when provided."""
        symbols = [
            JulieSymbol(
                id="sym1",
                name="greet",
                kind="function",
                language="python",
                file_path="src/hello.py",
                signature="def greet()",
                start_line=1,
                end_line=2,
                start_byte=None,
                end_byte=None,
                doc_comment=None,
                visibility="public",
                parent_id=None,
            ),
        ]
        chunks = chunk_code_symbols(symbols, {}, workspace_id="my-project_abc123")
        assert chunks[0].metadata["workspace_id"] == "my-project_abc123"

    def test_code_chunks_default_workspace_id(self):
        """Code chunks should default to empty workspace_id when not provided."""
        symbols = [
            JulieSymbol(
                id="sym1",
                name="greet",
                kind="function",
                language="python",
                file_path="src/hello.py",
                signature="def greet()",
                start_line=1,
                end_line=2,
                start_byte=None,
                end_byte=None,
                doc_comment=None,
                visibility="public",
                parent_id=None,
            ),
        ]
        chunks = chunk_code_symbols(symbols, {})
        assert chunks[0].metadata["workspace_id"] == ""

    def test_truncates_long_bodies(self):
        """Should truncate symbol bodies that exceed max_chars."""
        long_body = "x" * 10000
        symbols = [
            JulieSymbol(
                id="sym1",
                name="big",
                kind="function",
                language="python",
                file_path="src/big.py",
                signature="def big()",
                start_line=1,
                end_line=500,
                start_byte=0,
                end_byte=10000,
                doc_comment=None,
                visibility="public",
                parent_id=None,
            ),
        ]
        chunks = chunk_code_symbols(symbols, {"src/big.py": long_body}, max_chars=4000)
        assert len(chunks[0].text) <= 4200  # Some overhead for signature prefix


class TestDocChunking:
    """Test splitting documentation files into chunks."""

    def test_splits_by_sections(self, sample_docs):
        """Should split markdown by ## headings."""
        chunks = chunk_documentation(sample_docs)
        # auth.md has 3 ## sections, api.md has 1 ## section with 2 ### subsections
        assert len(chunks) >= 3

    def test_chunk_metadata(self, sample_docs):
        """Each doc chunk should have file path and section title metadata."""
        chunks = chunk_documentation(sample_docs)
        for chunk in chunks:
            assert chunk.collection == "docs"
            assert "file_path" in chunk.metadata
            assert "section" in chunk.metadata

    def test_preserves_section_content(self, sample_docs):
        """Section content should be preserved in chunks."""
        chunks = chunk_documentation(sample_docs)
        # Find the "Token Flow" section from auth.md
        token_chunks = [c for c in chunks if "Token Flow" in c.metadata.get("section", "")]
        assert len(token_chunks) == 1
        assert "credentials" in token_chunks[0].text
        assert "/api/login" in token_chunks[0].text

    def test_respects_max_chars(self):
        """Should split sections that exceed max_chars."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            doc_path = Path(tmpdir)
            # Create a document with one very long section
            long_section = "## Big Section\n\n" + ("This is a long paragraph. " * 200) + "\n"
            (doc_path / "long.md").write_text(f"# Title\n\n{long_section}")
            chunks = chunk_documentation(doc_path, max_chars=500)
            # Should be split into multiple chunks
            assert len(chunks) >= 2
            # All chunks should be under max_chars (with some tolerance for overlap)
            for chunk in chunks:
                assert len(chunk.text) <= 700  # 500 + overlap allowance

    def test_handles_empty_docs_dir(self, tmp_path):
        """Should return empty list for directory with no doc files."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        chunks = chunk_documentation(empty_dir)
        assert chunks == []

    def test_ignores_non_doc_files(self, tmp_path):
        """Should only process documentation file extensions."""
        docs_dir = tmp_path / "mixed"
        docs_dir.mkdir()
        (docs_dir / "readme.md").write_text("# Hello\n\nWorld\n")
        (docs_dir / "code.py").write_text("print('hello')\n")
        (docs_dir / "data.json").write_text('{"key": "value"}\n')
        chunks = chunk_documentation(docs_dir)
        assert all(c.metadata["file_path"].endswith(".md") for c in chunks)


class TestChunkDocFile:
    """Test chunking a single documentation file."""

    def test_chunks_single_markdown_file(self, tmp_path):
        """Should produce chunks from a single .md file."""
        md = tmp_path / "README.md"
        md.write_text("# Project\n\n## Setup\n\nRun `pip install`.\n\n## Usage\n\nImport and go.\n")
        chunks = chunk_doc_file(md)
        assert len(chunks) >= 2
        assert all(c.collection == "docs" for c in chunks)

    def test_metadata_uses_filename(self, tmp_path):
        """Metadata file_path should be the file name when no base_path given."""
        md = tmp_path / "GUIDE.md"
        md.write_text("# Guide\n\nSome content here.\n")
        chunks = chunk_doc_file(md)
        assert chunks[0].metadata["file_path"] == "GUIDE.md"

    def test_metadata_uses_relative_path(self, tmp_path):
        """When base_path is given, file_path should be relative to it."""
        subdir = tmp_path / "nested"
        subdir.mkdir()
        md = subdir / "deep.md"
        md.write_text("# Deep\n\nContent.\n")
        chunks = chunk_doc_file(md, base_path=tmp_path)
        assert chunks[0].metadata["file_path"] == "nested/deep.md"

    def test_returns_empty_for_nonexistent_file(self, tmp_path):
        """Should return empty list for a file that doesn't exist."""
        missing = tmp_path / "nope.md"
        chunks = chunk_doc_file(missing)
        assert chunks == []

    def test_respects_max_chars(self, tmp_path):
        """Should split long sections when max_chars is small."""
        md = tmp_path / "long.md"
        md.write_text("# Title\n\n" + ("word " * 500))
        chunks = chunk_doc_file(md, max_chars=200)
        assert len(chunks) >= 2


class TestChunkDocFiles:
    """Test chunking a list of individual doc files."""

    def test_chunks_multiple_files(self, tmp_path):
        """Should produce chunks from all provided files."""
        readme = tmp_path / "README.md"
        readme.write_text("# Readme\n\nHello world.\n")
        changelog = tmp_path / "CHANGELOG.md"
        changelog.write_text("# Changelog\n\n## v1.0\n\nInitial release.\n")
        chunks = chunk_doc_files([readme, changelog])
        files_seen = {c.metadata["file_path"] for c in chunks}
        assert "README.md" in files_seen
        assert "CHANGELOG.md" in files_seen

    def test_skips_missing_files(self, tmp_path):
        """Should silently skip files that don't exist."""
        real = tmp_path / "real.md"
        real.write_text("# Real\n\nContent.\n")
        fake = tmp_path / "fake.md"
        chunks = chunk_doc_files([real, fake])
        assert len(chunks) >= 1
        assert all("real.md" in c.metadata["file_path"] for c in chunks)

    def test_empty_list(self):
        """Should return empty list for empty input."""
        assert chunk_doc_files([]) == []


class TestDiscoverDocSources:
    """Test automatic discovery of documentation sources."""

    def test_finds_root_level_md_files(self, tmp_path):
        """Should find .md files at the project root."""
        (tmp_path / "README.md").write_text("# Hi")
        (tmp_path / "CLAUDE.md").write_text("# Rules")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        dirs, files = discover_doc_sources(tmp_path)
        filenames = {f.name for f in files}
        assert "README.md" in filenames
        assert "CLAUDE.md" in filenames

    def test_finds_docs_directories(self, tmp_path):
        """Should find docs/, doc/, documentation/ directories."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        dirs, files = discover_doc_sources(tmp_path)
        assert any(d.name == "docs" for d in dirs)

    def test_does_not_include_root_files_already_in_dirs(self, tmp_path):
        """Root-level files should NOT include files inside discovered directories."""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "guide.md").write_text("# Guide")
        (tmp_path / "README.md").write_text("# Hi")
        dirs, files = discover_doc_sources(tmp_path)
        # Files should only be root-level, not inside docs/
        for f in files:
            assert f.parent == tmp_path

    def test_skips_hidden_and_build_dirs(self, tmp_path):
        """Should not discover files in .git, .venv, node_modules, etc."""
        hidden = tmp_path / ".git"
        hidden.mkdir()
        (hidden / "README.md").write_text("# Git internal")
        venv = tmp_path / ".venv"
        venv.mkdir()
        (venv / "notes.md").write_text("# Venv")
        (tmp_path / "README.md").write_text("# Real")
        dirs, files = discover_doc_sources(tmp_path)
        filenames = {f.name for f in files}
        dir_names = {d.name for d in dirs}
        assert ".git" not in dir_names
        assert ".venv" not in dir_names
        # Root README should still be found
        assert "README.md" in filenames

    def test_finds_txt_and_rst_files(self, tmp_path):
        """Should discover .txt and .rst doc files at root level."""
        (tmp_path / "INSTALL.txt").write_text("Install instructions")
        (tmp_path / "CHANGES.rst").write_text("Changes\n=======\n")
        dirs, files = discover_doc_sources(tmp_path)
        extensions = {f.suffix for f in files}
        assert ".txt" in extensions
        assert ".rst" in extensions

    def test_empty_project(self, tmp_path):
        """Should return empty lists for a project with no docs."""
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hi')")
        dirs, files = discover_doc_sources(tmp_path)
        assert dirs == []
        assert files == []
