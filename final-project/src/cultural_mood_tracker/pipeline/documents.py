from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings, make_run_id
from cultural_mood_tracker.sources.tmdb import discover_titles, fetch_details, fetch_reviews


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _year(value: str | None) -> int | None:
    if value and len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    return None


def _genres(details: dict[str, Any]) -> list[str]:
    return [
        str(item["name"])
        for item in details.get("genres", [])
        if isinstance(item, dict) and item.get("name")
    ]


def refresh_tmdb_documents() -> list[dict[str, Any]]:
    """Download a compact RAG corpus from TMDB using the configured date window."""
    settings = load_settings()
    if not settings.tmdb_api_key:
        raise RuntimeError("TMDB_API_KEY is required for a live data refresh")

    documents: list[dict[str, Any]] = []
    counts = {"movie": settings.tmdb_movie_sample_size, "tv": settings.tmdb_tv_sample_size}
    for content_type, sample_size in counts.items():
        discovered = discover_titles(
            api_key=settings.tmdb_api_key,
            content_type=content_type,
            language=settings.tmdb_language,
            start_date=settings.tmdb_start_date,
            end_date=settings.tmdb_end_date,
            sample_size=sample_size,
        )
        for item in discovered:
            tmdb_id = int(item["id"])
            details = fetch_details(
                settings.tmdb_api_key, content_type, tmdb_id, settings.tmdb_language
            )
            reviews = fetch_reviews(
                settings.tmdb_api_key, content_type, tmdb_id, settings.tmdb_language
            )
            title = details.get("title") or details.get("name") or f"{content_type}-{tmdb_id}"
            release_date = details.get("release_date") or details.get("first_air_date") or ""
            title_id = f"{content_type}_{tmdb_id}"
            common = {
                "title_id": title_id,
                "title_name": title,
                "content_type": content_type,
                "release_year": _year(release_date),
                "genres": _genres(details),
                "source_name": "tmdb",
            }
            overview = str(details.get("overview") or "").strip()
            if overview:
                documents.append(
                    {
                        **common,
                        "document_id": f"{title_id}:overview",
                        "document_type": "overview",
                        "source_url": f"https://www.themoviedb.org/{content_type}/{tmdb_id}",
                        "text": overview,
                    }
                )
            for review in reviews.get("results", []):
                text = str(review.get("content") or "").strip()
                review_id = str(review.get("id") or "").strip()
                if not text or not review_id:
                    continue
                documents.append(
                    {
                        **common,
                        "document_id": f"{title_id}:review:{review_id}",
                        "document_type": "user_review",
                        "source_url": review.get("url") or "",
                        "text": text,
                    }
                )
    return documents


def chunk_documents(
    documents: list[dict[str, Any]], *, chunk_words: int = 220, overlap_words: int = 35
) -> list[dict[str, Any]]:
    if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Require chunk_words > overlap_words >= 0")
    chunks: list[dict[str, Any]] = []
    step = chunk_words - overlap_words
    for document in documents:
        words = str(document.get("text") or "").split()
        if not words:
            continue
        metadata = {key: value for key, value in document.items() if key != "text"}
        for index, start in enumerate(range(0, len(words), step), start=1):
            text = " ".join(words[start : start + chunk_words]).strip()
            if not text:
                continue
            chunks.append(
                {
                    "chunk_id": f"{document['document_id']}:chunk_{index:03d}",
                    "chunk_text": text,
                    **metadata,
                    "chunk_index": index,
                }
            )
            if start + chunk_words >= len(words):
                break
    return chunks


def prepare_documents_run(
    project_root: Path,
    documents: list[dict[str, Any]],
    *,
    run_id: str | None = None,
    source: str,
) -> tuple[str, Path]:
    resolved_run_id = run_id or make_run_id()
    paths = load_settings().build_paths(project_root)
    paths.ensure()
    run_dir = paths.processed_root / resolved_run_id
    normalized = [{**document, "process_run_id": resolved_run_id} for document in documents]
    chunks = chunk_documents(normalized)
    document_path = run_dir / "documents.jsonl"
    chunk_path = run_dir / "document_chunks.jsonl"
    write_jsonl(document_path, normalized)
    write_jsonl(chunk_path, chunks)
    write_json(
        run_dir / "run_manifest.json",
        {
            "process_run_id": resolved_run_id,
            "source": source,
            "documents": len(normalized),
            "chunks": len(chunks),
            "outputs": ["documents.jsonl", "document_chunks.jsonl"],
        },
    )
    return resolved_run_id, chunk_path
