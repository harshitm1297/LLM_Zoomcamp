from __future__ import annotations

import argparse
from pathlib import Path

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment
from cultural_mood_tracker.rag import DEFAULT_EMBEDDING_MODEL, embed_document_chunks, load_document_chunks
from cultural_mood_tracker.rag.document_chunks import resolve_local_document_chunks_path


def parse_args(default_process_run_id: str | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed processed document chunks into ChromaDB-ready JSONL records."
    )
    parser.add_argument(
        "--process-run-id",
        default=default_process_run_id,
        help="Processed run ID used for the output embedding file.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Explicit document_chunks JSONL/CSV path. Overrides automatic run discovery.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"SentenceTransformer model name. Defaults to {DEFAULT_EMBEDDING_MODEL}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Embedding batch size.",
    )
    parser.add_argument(
        "--output-name",
        default="document_chunk_embeddings.jsonl",
        help="Output filename written under the processed run directory.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable embedding normalization. Normalized vectors are recommended for cosine retrieval.",
    )
    return parser.parse_args()


def run_embed_document_chunks(
    project_root: Path,
    process_run_id: str,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    output_name: str = "document_chunk_embeddings.jsonl",
    normalize_embeddings: bool = True,
    input_path: Path | None = None,
) -> Path:
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    processed_dir = paths.processed_root / process_run_id
    output_path = processed_dir / output_name
    chunks = load_document_chunks(
        input_path=input_path,
        process_run_id=process_run_id,
    )
    count = embed_document_chunks(
        chunks,
        output_path,
        model_name=model_name,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )
    print(f"Embedded chunks written to: {output_path}")
    print(f"model={model_name} chunks={count} normalized={normalize_embeddings}")
    return output_path


def main() -> int:
    project_root = load_project_environment(Path(__file__))
    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()

    default_process_run_id = settings.process_run_id or None
    if default_process_run_id is None:
        try:
            default_path = resolve_local_document_chunks_path(project_root)
            default_process_run_id = default_path.parent.name
        except RuntimeError:
            default_process_run_id = None

    args = parse_args(default_process_run_id)
    if not args.process_run_id and args.input_path is None:
        raise RuntimeError(
            "No prepared run available. Run bootstrap/refresh first or pass --process-run-id."
        )

    process_run_id = args.process_run_id or args.input_path.parent.name

    run_embed_document_chunks(
        project_root,
        process_run_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
        output_name=args.output_name,
        normalize_embeddings=not args.no_normalize,
        input_path=args.input_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
