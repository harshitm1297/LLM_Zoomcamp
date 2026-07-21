from __future__ import annotations

from pathlib import Path
from typing import Any

from cultural_mood_tracker.cli.embed_document_chunks import run_embed_document_chunks
from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.rag import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_DIR,
    ingest_embeddings_file,
)

from .documents import prepare_documents_run, refresh_tmdb_documents, write_json
from .sample_data import sample_documents


def bootstrap_application(
    *,
    project_root: Path,
    sample: bool = False,
    process_run_id: str | None = None,
    skip_vector_index: bool = False,
) -> dict[str, Any]:
    """Prepare documents and build the only application store: ChromaDB."""
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()
    bootstrap_run_id = make_run_id()

    if process_run_id:
        resolved_run_id = process_run_id
        chunk_path = paths.processed_root / resolved_run_id / "document_chunks.jsonl"
        if not chunk_path.is_file():
            raise RuntimeError(f"Prepared run {resolved_run_id!r} has no {chunk_path.name}")
        source = "prepared"
    else:
        source = "sample" if sample else "tmdb"
        documents = sample_documents() if sample else refresh_tmdb_documents()
        resolved_run_id, chunk_path = prepare_documents_run(
            project_root,
            documents,
            run_id="sample" if sample else None,
            source=source,
        )

    vector_manifest: dict[str, Any] = {"enabled": not skip_vector_index}
    if not skip_vector_index:
        embedding_path = run_embed_document_chunks(
            project_root,
            resolved_run_id,
            input_path=chunk_path,
        )
        persist_dir = project_root / DEFAULT_CHROMA_DB_DIR
        inserted = ingest_embeddings_file(
            embedding_path,
            persist_dir=persist_dir,
            collection_name=DEFAULT_CHROMA_COLLECTION,
            replace_collection=True,
        )
        vector_manifest.update(
            {
                "embedding_path": str(embedding_path),
                "persist_dir": str(persist_dir),
                "collection_name": DEFAULT_CHROMA_COLLECTION,
                "inserted": inserted,
            }
        )

    manifest = {
        "bootstrap_run_id": bootstrap_run_id,
        "mode": source,
        "process_run_id": resolved_run_id,
        "document_chunks_path": str(chunk_path),
        "vector": vector_manifest,
        "status": "completed",
    }
    report_path = paths.reports_root / bootstrap_run_id / "bootstrap_manifest.json"
    write_json(report_path, manifest)
    return {"manifest_path": str(report_path), **manifest}
