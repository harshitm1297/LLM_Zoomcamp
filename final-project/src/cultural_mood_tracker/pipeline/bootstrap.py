from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cultural_mood_tracker.cli.embed_document_chunks import run_embed_document_chunks
from cultural_mood_tracker.cli.extract_multisource import run_extraction
from cultural_mood_tracker.cli.transform_canonical import run_transform
from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.load import run_local_duckdb_load
from cultural_mood_tracker.rag import (
    DEFAULT_CHROMA_COLLECTION,
    DEFAULT_CHROMA_DB_DIR,
    ingest_embeddings_file,
)
from cultural_mood_tracker.transform.common import write_json

from .sample_data import create_sample_processed_run


def _full_extract_args(settings) -> SimpleNamespace:
    return SimpleNamespace(
        movie_count=settings.tmdb_movie_sample_size,
        tv_count=settings.tmdb_tv_sample_size,
        language=settings.tmdb_language,
        start_date=settings.tmdb_start_date,
        end_date=settings.tmdb_end_date,
        output_root=str(settings.local_data_root),
        guardian_api_key=settings.guardian_api_key,
        guardian_page_size=settings.guardian_page_size,
        gdelt_max_records=settings.gdelt_max_records,
        enable_gdelt=False,
        cleanup_old_raw=False,
        disable_critic_blogs=not settings.enable_critic_blog_sources,
    )


def bootstrap_application(
    *,
    project_root: Path,
    sample: bool = False,
    source_run_id: str | None = None,
    process_run_id: str | None = None,
    skip_vector_index: bool = False,
) -> dict[str, Any]:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()
    bootstrap_run_id = make_run_id()

    if sample:
        resolved_process_run_id = process_run_id or "sample"
        create_sample_processed_run(project_root, resolved_process_run_id)
        resolved_source_run_id = "sample"
    elif process_run_id:
        resolved_process_run_id = process_run_id
        resolved_source_run_id = source_run_id or "existing"
    else:
        resolved_source_run_id = source_run_id or run_extraction(
            project_root, _full_extract_args(settings)
        )
        resolved_process_run_id = run_transform(project_root, resolved_source_run_id)

    sql_manifest = run_local_duckdb_load(
        project_root=project_root,
        process_run_id=resolved_process_run_id,
    )

    vector_manifest: dict[str, Any] = {"enabled": not skip_vector_index}
    if not skip_vector_index:
        embedding_path = run_embed_document_chunks(
            project_root,
            resolved_process_run_id,
        )
        persist_dir = project_root / DEFAULT_CHROMA_DB_DIR
        inserted = ingest_embeddings_file(
            embedding_path,
            persist_dir=persist_dir,
            collection_name=DEFAULT_CHROMA_COLLECTION,
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
        "mode": "sample" if sample else "full",
        "source_run_id": resolved_source_run_id,
        "process_run_id": resolved_process_run_id,
        "sql": sql_manifest,
        "vector": vector_manifest,
        "status": "completed",
    }
    report_path = paths.reports_root / bootstrap_run_id / "bootstrap_manifest.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, manifest)
    return {"manifest_path": str(report_path), **manifest}
