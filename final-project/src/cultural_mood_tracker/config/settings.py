from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import ProjectPaths, build_project_paths


@dataclass(frozen=True)
class Settings:
    tmdb_api_key: str
    tmdb_language: str
    tmdb_region: str
    tmdb_start_date: str
    tmdb_end_date: str
    tmdb_movie_sample_size: int
    tmdb_tv_sample_size: int
    guardian_api_key: str
    guardian_page_size: int
    gdelt_max_records: int
    enable_critic_blog_sources: bool
    critic_feed_entry_limit: int
    document_chunks_path: Path | None
    process_run_id: str
    local_duckdb_path: Path
    retrieval_strategy: str
    retrieval_top_k: int
    retrieval_candidate_k: int
    enable_query_rewriting: bool
    observability_db_path: Path
    prompt_variant: str
    llm_temperature: float
    local_data_root: Path
    log_level: str

    def build_paths(self, project_root: Path) -> ProjectPaths:
        return build_project_paths(project_root=project_root, data_root=self.local_data_root)


def _get_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _get_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    return Settings(
        # Extraction validates this key when it is actually needed. Keeping it
        # optional here lets prepared/sample application modes run without ETL secrets.
        tmdb_api_key=_get_env("TMDB_API_KEY"),
        tmdb_language=os.getenv("TMDB_LANGUAGE", "en-US").strip() or "en-US",
        tmdb_region=os.getenv("TMDB_REGION", "").strip(),
        tmdb_start_date=os.getenv("TMDB_START_DATE", "2025-06-29").strip() or "2025-06-29",
        tmdb_end_date=os.getenv("TMDB_END_DATE", "2026-06-29").strip() or "2026-06-29",
        tmdb_movie_sample_size=int(os.getenv("TMDB_MOVIE_SAMPLE_SIZE", "300")),
        tmdb_tv_sample_size=int(os.getenv("TMDB_TV_SAMPLE_SIZE", "200")),
        guardian_api_key=os.getenv("GUARDIAN_API_KEY", "test").strip() or "test",
        guardian_page_size=int(os.getenv("GUARDIAN_PAGE_SIZE", "5")),
        gdelt_max_records=int(os.getenv("GDELT_MAX_RECORDS", "5")),
        enable_critic_blog_sources=_get_bool("ENABLE_CRITIC_BLOG_SOURCES", default=True),
        critic_feed_entry_limit=int(os.getenv("CRITIC_FEED_ENTRY_LIMIT", "40")),
        document_chunks_path=(
            Path(value) if (value := _get_env("DOCUMENT_CHUNKS_PATH")) else None
        ),
        process_run_id=_get_env("PROCESS_RUN_ID"),
        local_duckdb_path=Path(_get_env("LOCAL_DUCKDB_PATH", "data/warehouse/cultural_mood_tracker.duckdb")),
        retrieval_strategy=_get_env("RETRIEVAL_STRATEGY", "vector").lower() or "vector",
        retrieval_top_k=int(_get_env("RETRIEVAL_TOP_K", "5")),
        retrieval_candidate_k=int(_get_env("RETRIEVAL_CANDIDATE_K", "20")),
        enable_query_rewriting=_get_bool("ENABLE_QUERY_REWRITING", default=True),
        observability_db_path=Path(_get_env("OBSERVABILITY_DB_PATH", "data/monitoring/observability.db")),
        prompt_variant=_get_env("PROMPT_VARIANT", "strict").lower() or "strict",
        llm_temperature=float(_get_env("LLM_TEMPERATURE", "0.2")),
        local_data_root=Path(os.getenv("LOCAL_DATA_ROOT", "data")),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip() or "INFO",
    )
