from __future__ import annotations

from pathlib import Path
from typing import Any

from cultural_mood_tracker.transform.common import write_json, write_jsonl
from cultural_mood_tracker.config import load_settings


SAMPLE_TITLES = [
    ("movie_1275779", "Disclosure Day", "movie", ["Science Fiction", "Thriller"], "2026-06-12"),
    ("movie_687163", "Project Hail Mary", "movie", ["Science Fiction", "Drama"], "2026-03-20"),
    ("tv_259909", "Dexter: Resurrection", "tv", ["Crime", "Drama"], "2025-07-11"),
    ("movie_1288341", "Seven Snipers", "movie", ["Action", "Thriller"], "2026-02-10"),
    ("movie_1398050", "Driver's Ed", "movie", ["Comedy", "Drama"], "2026-04-03"),
    ("movie_1020047", "In the Hand of Dante", "movie", ["Mystery", "Drama"], "2025-09-05"),
    ("tv_sample_obsession", "Obsession", "tv", ["Drama", "Thriller"], "2025-10-01"),
    ("movie_sample_leviticus", "Leviticus", "movie", ["Drama", "Mystery"], "2026-01-15"),
]

SAMPLE_TEXT = {
    "movie_1275779": "A whistleblower uncovers evidence of extraterrestrial life and exposes a corporate and government cover-up.",
    "movie_687163": "A scientist wakes alone in space with amnesia and must solve a stellar crisis to save the Sun and humanity.",
    "tv_259909": "After waking from a coma, a serial killer travels through New York while searching for his missing son.",
    "movie_1288341": "A former assassin protects her daughter from the warlord who once held her captive.",
    "movie_1398050": "Teenagers steal a car and take a chaotic road trip to reunite their heartbroken friend with his college girlfriend.",
    "movie_1020047": "A rare medieval manuscript is stolen from the Vatican shortly after experts authenticate it.",
    "tv_sample_obsession": "A tense relationship becomes a public obsession as secrets, ambition, and anxiety collide.",
    "movie_sample_leviticus": "A mysterious family conflict explores identity, faith, responsibility, and reconciliation.",
}


def _rows(process_run_id: str) -> dict[str, list[dict[str, Any]]]:
    titles: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    chunks: list[dict[str, Any]] = []
    ratings: list[dict[str, Any]] = []
    attention: list[dict[str, Any]] = []
    title_themes: list[dict[str, Any]] = []
    genre_themes: list[dict[str, Any]] = []
    audience_editorial: list[dict[str, Any]] = []
    attention_reception: list[dict[str, Any]] = []
    people: list[dict[str, Any]] = []
    title_cast: list[dict[str, Any]] = []

    for index, (title_id, title, content_type, genres, release_date) in enumerate(SAMPLE_TITLES):
        text = SAMPLE_TEXT[title_id]
        document_id = f"{title_id}:tmdb_overview"
        chunk_id = f"{document_id}:chunk_001"
        titles.append(
            {
                "title_id": title_id,
                "title_name": title,
                "normalized_title": title.casefold(),
                "content_type": content_type,
                "genres": genres,
                "imdb_genres": genres,
                "tvmaze_genres": genres,
                "release_date": release_date,
                "process_run_id": process_run_id,
            }
        )
        documents.append(
            {
                "document_id": document_id,
                "title_id": title_id,
                "title_name": title,
                "source_name": "sample",
                "document_type": "overview",
                "document_text": text,
                "published_at": f"{release_date}T00:00:00Z",
                "process_run_id": process_run_id,
            }
        )
        chunks.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "title_id": title_id,
                "title_name": title,
                "content_type": content_type,
                "source_name": "sample",
                "document_type": "overview",
                "chunk_text": text,
                "published_at": f"{release_date}T00:00:00Z",
                "process_run_id": process_run_id,
            }
        )
        rating = round(6.8 + index * 0.2, 1)
        ratings.append(
            {
                "rating_id": f"{title_id}:sample_aggregate",
                "title_id": title_id,
                "source_name": "sample",
                "rating_scope": "title_aggregate",
                "rating_value": rating,
                "rating_count": 1000 + index * 750,
                "published_at": f"{release_date}T00:00:00Z",
                "process_run_id": process_run_id,
            }
        )
        attention.append(
            {
                "attention_id": f"{title_id}:sample_attention",
                "title_id": title_id,
                "source_name": "sample",
                "signal_type": "pageviews",
                "signal_value": 10000 + index * 5000,
                "timestamp_utc": f"{release_date}T00:00:00Z",
                "process_run_id": process_run_id,
            }
        )
        theme = "escapism" if index % 2 == 0 else "identity"
        title_themes.append(
            {
                "title_id": title_id,
                "title_name": title,
                "dominant_themes": [theme],
                "chunk_count": 4 + index,
                "avg_sentiment_score": 0.12 if index % 2 else 0.01,
                "process_run_id": process_run_id,
            }
        )
        for source_group, sentiment in (("audience", 0.14), ("editorial", 0.06)):
            audience_editorial.append(
                {
                    "title_id": title_id,
                    "source_group": source_group,
                    "chunk_count": 2 + index,
                    "dominant_themes": [theme],
                    "avg_sentiment_score": sentiment if index % 2 else sentiment - 0.08,
                    "process_run_id": process_run_id,
                }
            )
        attention_reception.append(
            {
                "title_id": title_id,
                "title_name": title,
                "avg_rating_value": rating,
                "attention_total": 10000 + index * 5000,
                "attention_peak": 6000 + index * 2500,
                "avg_sentiment_score": 0.12 if index % 2 else 0.01,
                "dominant_themes": [theme],
                "chunk_count": 4 + index,
                "process_run_id": process_run_id,
            }
        )
        for cast_index in range(2):
            person_id = f"sample_person_{index}_{cast_index}"
            people.append(
                {
                    "person_id": person_id,
                    "source_name": "sample",
                    "name": f"Sample Performer {index * 2 + cast_index + 1}",
                    "process_run_id": process_run_id,
                }
            )
            title_cast.append(
                {
                    "title_id": title_id,
                    "person_id": person_id,
                    "source_name": "sample",
                    "character_name": f"Character {cast_index + 1}",
                    "billing_order": cast_index,
                    "process_run_id": process_run_id,
                }
            )

    for genre in sorted({genre for _, _, _, genres, _ in SAMPLE_TITLES for genre in genres}):
        genre_themes.append(
            {
                "genre": genre,
                "dominant_themes": ["identity"],
                "title_count": sum(genre in row[3] for row in SAMPLE_TITLES),
                "avg_sentiment_score": 0.08,
                "process_run_id": process_run_id,
            }
        )

    return {
        "titles": titles,
        "documents": documents,
        "document_chunks": chunks,
        "ratings": ratings,
        "attention_signals": attention,
        "people": people,
        "title_cast": title_cast,
        "title_theme_summary": title_themes,
        "genre_theme_summary": genre_themes,
        "audience_vs_editorial_summary": audience_editorial,
        "attention_vs_reception": attention_reception,
    }


def create_sample_processed_run(project_root: Path, process_run_id: str = "sample") -> Path:
    paths = load_settings().build_paths(project_root)
    processed_dir = paths.processed_root / process_run_id
    report_dir = paths.reports_root / process_run_id
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    tables = _rows(process_run_id)
    for table_name, rows in tables.items():
        write_jsonl(processed_dir / f"{table_name}.jsonl", rows)
    write_json(
        processed_dir / "run_manifest.json",
        {
            "source_run_id": "sample",
            "process_run_id": process_run_id,
            "sample": True,
            "outputs": sorted(f"{name}.jsonl" for name in tables),
        },
    )
    return processed_dir
