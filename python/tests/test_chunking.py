"""Tests for code and documentation chunking strategies."""

import pytest

from eros.chunking import (
    Chunk,
    chunk_code_symbols,
    chunk_documentation,
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
        from pathlib import Path

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
