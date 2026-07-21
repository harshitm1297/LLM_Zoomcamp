from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import load_project_environment


DEFAULT_DOCUMENT_CHUNKS_FILENAME = "document_chunks.jsonl"


def _project_root() -> Path:
    return load_project_environment(Path.cwd())


def _parse_metadata(value: Any, *, row_number: int) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Row {row_number} has invalid JSON metadata") from exc
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Row {row_number} has non-object metadata")


def _normalize_chunk_row(row: dict[str, Any], *, row_number: int) -> dict[str, Any]:
    chunk_id = row.get("chunk_id")
    chunk_text = row.get("chunk_text")
    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError(f"Row {row_number} is missing a non-empty chunk_id")
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        raise ValueError(f"Row {row_number} is missing non-empty chunk_text")

    if "metadata" in row:
        metadata = _parse_metadata(row.get("metadata"), row_number=row_number)
    else:
        metadata = {
            key: value
            for key, value in row.items()
            if key not in {"chunk_id", "chunk_text"} and value is not None
        }

    return {
        "chunk_id": chunk_id,
        "chunk_text": chunk_text,
        "metadata": metadata,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            chunks.append(_normalize_chunk_row(row, row_number=line_number))
    return chunks


def _load_csv(path: Path) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_number, row in enumerate(reader, start=2):
            chunks.append(_normalize_chunk_row(dict(row), row_number=row_number))
    return chunks


def _candidate_chunk_path(path: Path) -> Path | None:
    if path.exists() and path.is_file():
        return path
    csv_path = path.with_suffix(".csv")
    if csv_path.exists() and csv_path.is_file():
        return csv_path
    return None


def resolve_local_document_chunks_path(
    project_root: Path,
    *,
    input_path: Path | None = None,
    process_run_id: str | None = None,
) -> Path:
    """Resolve chunks explicitly, by run ID, or from the newest valid processed run."""
    settings = load_settings()
    configured_path = input_path or settings.document_chunks_path
    if configured_path is not None:
        resolved = configured_path if configured_path.is_absolute() else project_root / configured_path
        candidate = _candidate_chunk_path(resolved.resolve())
        if candidate is None:
            raise RuntimeError(f"Configured document chunks file does not exist: {resolved}")
        return candidate

    paths = settings.build_paths(project_root)
    requested_run = process_run_id or settings.process_run_id
    if requested_run:
        expected = paths.processed_root / requested_run / DEFAULT_DOCUMENT_CHUNKS_FILENAME
        candidate = _candidate_chunk_path(expected)
        if candidate is None:
            raise RuntimeError(f"Processed run {requested_run!r} has no document chunks file: {expected}")
        return candidate

    if paths.processed_root.exists():
        for run_dir in sorted(
            (path for path in paths.processed_root.iterdir() if path.is_dir()),
            key=lambda path: path.name,
            reverse=True,
        ):
            candidate = _candidate_chunk_path(run_dir / DEFAULT_DOCUMENT_CHUNKS_FILENAME)
            if candidate is not None:
                return candidate

    raise RuntimeError(
        "No processed document chunks were found. Pass --input-path/--process-run-id, "
        "set DOCUMENT_CHUNKS_PATH/PROCESS_RUN_ID, or run the bootstrap/refresh command first."
    )


def _load_local_document_chunks(
    *, input_path: Path | None = None, process_run_id: str | None = None
) -> list[dict[str, Any]]:
    path = resolve_local_document_chunks_path(
        _project_root(), input_path=input_path, process_run_id=process_run_id
    )
    if path.suffix.lower() == ".csv":
        chunks = _load_csv(path)
    else:
        chunks = _load_jsonl(path)
    print(f"RAG document chunk source: local ({path})")
    print(f"Loaded document chunks: {len(chunks)}")
    return chunks


def load_document_chunks(
    *,
    input_path: Path | None = None,
    process_run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load prepared document chunks from the local processed-data directory."""
    return _load_local_document_chunks(input_path=input_path, process_run_id=process_run_id)
