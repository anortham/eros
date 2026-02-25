"""Index build and refresh logic for semantic_index."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from eros.chunking import Chunk, chunk_code_symbols, chunk_doc_file, discover_doc_sources
from eros.config import ErosConfig
from eros.embeddings import DualEmbeddingManager
from eros.incremental import IndexManifest, sha1_text, where_workspace_and_paths
from eros.julie_reader import JulieReader
from eros.storage import VectorStorage


async def build_index(
    full_rebuild: bool,
    workspace: str | None,
    doc_paths: list[str] | None,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    config: ErosConfig,
) -> str:
    """Build or refresh the vector index from Julie data and docs."""
    t_start = time.monotonic()
    lines: list[str] = []
    manifest = IndexManifest(config.index_manifest_path)
    target = workspace or "all"
    primary_ws_id = ""

    try:
        reader = JulieReader(config.project_root)
        primary_ws_id = reader.workspace_id
        workspaces = reader.resolve_workspace(target)

        if full_rebuild:
            storage.clear_collection("code")
            manifest.clear("code")

        total_chunks = 0
        total_symbols = 0
        for ws in workspaces:
            file_hashes = reader.read_file_hashes(workspace_id=ws.id)
            delta = manifest.code_delta(ws.id, file_hashes)
            changed_files = set(file_hashes.keys()) if full_rebuild else delta.changed
            removed_files = set() if full_rebuild else delta.removed
            affected_files = changed_files | removed_files

            if affected_files:
                storage.delete_where("code", where_workspace_and_paths(ws.id, affected_files))

            if changed_files:
                symbols = reader.read_symbols(
                    exclude_kinds=["import", "variable", "constant"],
                    file_paths=sorted(changed_files),
                    workspace_id=ws.id,
                )
                file_contents = reader.read_file_contents(sorted(changed_files), workspace_id=ws.id)
                code_chunks = chunk_code_symbols(
                    symbols,
                    file_contents,
                    max_chars=config.max_code_chunk_chars,
                    workspace_id=ws.id,
                )
                if code_chunks:
                    texts = [c.text for c in code_chunks]
                    t_ws = time.monotonic()
                    avg_len = sum(len(text) for text in texts) / len(texts)
                    batch_size = embeddings.index_batch_size(
                        "code", len(texts), avg_text_len=avg_len
                    )
                    vectors = await asyncio.to_thread(
                        embeddings.embed_code, texts, batch_size=batch_size
                    )
                    ws_elapsed = time.monotonic() - t_ws
                    count = storage.add_chunks(code_chunks, vectors)
                    total_chunks += count
                    total_symbols += len(symbols)
                    lines.append(
                        f"  {ws.display_name} ({ws.workspace_type}): {count} chunks from {len(symbols)} symbols [{ws_elapsed:.1f}s]"
                    )
            elif not full_rebuild:
                lines.append(f"  {ws.display_name} ({ws.workspace_type}): up-to-date")

            manifest.update_code(ws.id, file_hashes)

        if total_chunks:
            elapsed = time.monotonic() - t_start
            lines.insert(
                0,
                f"Indexed {total_chunks} code chunks from {total_symbols} symbols across {len(workspaces)} workspace(s) in {elapsed:.1f}s:",
            )
        elif full_rebuild:
            lines.append("No code symbols found to index")

    except FileNotFoundError as e:
        lines.append(f"Julie data not available: {e}")

    if primary_ws_id:
        lines.extend(
            await _index_docs(
                full_rebuild=full_rebuild,
                primary_ws_id=primary_ws_id,
                doc_paths=doc_paths,
                embeddings=embeddings,
                storage=storage,
                config=config,
                manifest=manifest,
            )
        )

    manifest.save()
    return "\n".join(lines) if lines else "Nothing to index"


async def _index_docs(
    full_rebuild: bool,
    primary_ws_id: str,
    doc_paths: list[str] | None,
    embeddings: DualEmbeddingManager,
    storage: VectorStorage,
    config: ErosConfig,
    manifest: IndexManifest,
) -> list[str]:
    lines: list[str] = []
    if full_rebuild:
        storage.clear_collection("docs")
        manifest.clear("docs")

    doc_files = _discover_doc_files(config.project_root, doc_paths)
    current_hashes = {
        str(path.relative_to(config.project_root)): sha1_text(
            path.read_text(encoding="utf-8", errors="replace")
        )
        for path in doc_files
    }
    delta = manifest.docs_delta(primary_ws_id, current_hashes)
    changed_docs = set(current_hashes.keys()) if full_rebuild else delta.changed
    removed_docs = set() if full_rebuild else delta.removed

    if changed_docs or removed_docs:
        storage.delete_where(
            "docs", where_workspace_and_paths(primary_ws_id, changed_docs | removed_docs)
        )

    changed_paths = [
        config.project_root / rel
        for rel in sorted(changed_docs)
        if (config.project_root / rel).is_file()
    ]
    doc_chunks: list[Chunk] = []
    for path in changed_paths:
        doc_chunks.extend(
            chunk_doc_file(
                path,
                base_path=config.project_root,
                max_chars=config.max_doc_chunk_chars,
                overlap=config.doc_chunk_overlap,
            )
        )

    for chunk in doc_chunks:
        chunk.metadata["workspace_id"] = primary_ws_id

    if doc_chunks:
        texts = [c.text for c in doc_chunks]
        t_doc = time.monotonic()
        avg_len = sum(len(text) for text in texts) / len(texts)
        batch_size = embeddings.index_batch_size("docs", len(texts), avg_text_len=avg_len)
        vectors = await asyncio.to_thread(embeddings.embed_docs, texts, batch_size=batch_size)
        elapsed = time.monotonic() - t_doc
        count = storage.add_chunks(doc_chunks, vectors)
        lines.append(
            f"Indexed {count} doc chunks from {len(changed_paths)} file(s) [{elapsed:.1f}s]"
        )
    elif full_rebuild:
        lines.append("No documentation found to index")
    elif doc_files:
        lines.append("Documentation up-to-date")

    manifest.update_docs(primary_ws_id, current_hashes)
    return lines


def _discover_doc_files(project_root: Path, doc_paths: list[str] | None) -> list[Path]:
    doc_dirs, root_doc_files = discover_doc_sources(project_root)
    explicit_files: list[Path] = []
    if doc_paths:
        for raw in doc_paths:
            path = Path(raw)
            if path.is_dir():
                doc_dirs.append(path)
            elif path.is_file():
                explicit_files.append(path)

    files = set(root_doc_files + explicit_files)
    doc_exts = ErosConfig().doc_extensions()
    for doc_dir in doc_dirs:
        files.update(
            path
            for path in doc_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in doc_exts
        )
    return sorted(path for path in files if path.suffix.lower() in doc_exts)
