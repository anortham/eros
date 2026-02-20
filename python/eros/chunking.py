"""
Chunking strategies for code and documentation.

Code chunks: One chunk per symbol from Julie's data (signature + doc + body).
Doc chunks: Section-based splitting of .md/.txt/.rst with configurable overlap.

Each chunk carries metadata (file_path, language, kind, section) and a
collection tag ("code" or "docs") that determines which embedding model
processes it.
"""

from dataclasses import dataclass, field
from pathlib import Path

from eros.julie_reader import JulieSymbol

# File extensions treated as documentation
DOC_EXTENSIONS = frozenset({".md", ".txt", ".rst", ".adoc"})


@dataclass
class Chunk:
    """A text chunk ready for embedding."""

    text: str
    collection: str  # "code" or "docs"
    metadata: dict = field(default_factory=dict)


def chunk_code_symbols(
    symbols: list[JulieSymbol],
    file_contents: dict[str, str],
    max_chars: int = 4000,
) -> list[Chunk]:
    """Create one chunk per code symbol.

    Each chunk combines:
      1. Signature (always present from Julie)
      2. Doc comment (if available)
      3. Body text extracted via byte offsets (if file content available)

    Truncates body to max_chars if too long.
    """
    chunks = []
    for sym in symbols:
        parts: list[str] = []

        # Signature first — this is the most important identifier
        if sym.signature:
            parts.append(sym.signature)

        # Doc comment adds semantic meaning
        if sym.doc_comment:
            parts.append(sym.doc_comment)

        # Body text from file content (extracted via byte offsets)
        content = file_contents.get(sym.file_path, "")
        if content:
            body = sym.extract_body(content)
            if body:
                # Avoid duplicating the signature if body starts with it
                if sym.signature and body.strip().startswith(sym.signature.strip()):
                    # Body already includes signature, use it directly
                    parts = [body]
                    if sym.doc_comment:
                        parts.insert(0, sym.doc_comment)
                else:
                    parts.append(body)

        text = "\n".join(parts)

        # Truncate if too long
        if len(text) > max_chars:
            text = text[:max_chars]

        chunks.append(
            Chunk(
                text=text,
                collection="code",
                metadata={
                    "symbol_id": sym.id,
                    "symbol_name": sym.name,
                    "kind": sym.kind,
                    "language": sym.language,
                    "file_path": sym.file_path,
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                },
            )
        )

    return chunks


def chunk_doc_file(
    file_path: Path,
    base_path: Path | None = None,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk a single documentation file into section-based chunks.

    Args:
        file_path: Path to the documentation file.
        base_path: If given, metadata file_path is relative to this.
                   Otherwise uses the file name alone.
        max_chars: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        return []

    text = file_path.read_text(encoding="utf-8", errors="replace")
    rel_path = (
        str(file_path.relative_to(base_path)) if base_path else file_path.name
    )

    chunks: list[Chunk] = []
    sections = _split_markdown_sections(text)
    for section_title, section_text in sections:
        section_chunks = _split_with_overlap(section_text, max_chars, overlap)
        for i, chunk_text in enumerate(section_chunks):
            suffix = f" (part {i + 1})" if len(section_chunks) > 1 else ""
            chunks.append(
                Chunk(
                    text=chunk_text,
                    collection="docs",
                    metadata={
                        "file_path": rel_path,
                        "section": f"{section_title}{suffix}",
                    },
                )
            )
    return chunks


def chunk_doc_files(
    files: list[Path],
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk a list of individual documentation files.

    Each file's metadata uses its filename (no base_path relativity).
    """
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_doc_file(f, max_chars=max_chars, overlap=overlap))
    return chunks


def chunk_documentation(
    docs_path: Path,
    max_chars: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Split all documentation files in a directory into section-based chunks.

    Recursively walks the directory for .md/.txt/.rst/.adoc files.
    Splits on ## headings for markdown. Sections exceeding max_chars
    are further split with overlap for context continuity.
    """
    docs_path = Path(docs_path)
    if not docs_path.is_dir():
        return []

    chunks: list[Chunk] = []
    for file_path in sorted(docs_path.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in DOC_EXTENSIONS:
            continue
        chunks.extend(
            chunk_doc_file(file_path, base_path=docs_path, max_chars=max_chars, overlap=overlap)
        )
    return chunks


def discover_doc_sources(
    project_root: Path,
) -> tuple[list[Path], list[Path]]:
    """Discover documentation directories and root-level doc files.

    Returns:
        (directories, files) — directories are scanned recursively by
        chunk_documentation; files are individual root-level docs.
    """
    project_root = Path(project_root)
    dirs: list[Path] = []
    files: list[Path] = []

    # Well-known doc directories
    for name in ("docs", "doc", "documentation"):
        candidate = project_root / name
        if candidate.is_dir():
            dirs.append(candidate)

    # Root-level doc files (non-recursive, immediate children only)
    for entry in sorted(project_root.iterdir()):
        if entry.is_file() and entry.suffix.lower() in DOC_EXTENSIONS:
            files.append(entry)

    return dirs, files


def _split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown text into (heading, content) pairs on ## boundaries.

    If the document has no ## headings, returns the entire text as one section.
    """
    lines = text.split("\n")
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in lines:
        # Split on ## (h2) or higher headings, but not code blocks
        if line.startswith("## ") or line.startswith("# "):
            # Save previous section if it has content
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_title, content))
            current_title = line.lstrip("#").strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    # Don't forget the last section
    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_title, content))

    return sections


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split text into chunks of max_chars with overlap for context continuity.

    Tries to split on paragraph boundaries for cleaner chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + max_chars

        if end >= len(text):
            chunks.append(text[start:])
            break

        # Try to find a paragraph boundary to split on
        split_point = text.rfind("\n\n", start, end)
        if split_point <= start:
            # No paragraph boundary — try a line boundary
            split_point = text.rfind("\n", start, end)
        if split_point <= start:
            # No line boundary either — hard split
            split_point = end

        chunks.append(text[start:split_point])
        start = max(split_point - overlap, start + 1)  # overlap for context

    return chunks
