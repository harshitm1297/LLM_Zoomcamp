from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .paths import ProjectPaths, build_project_paths


@dataclass(frozen=True)
class Settings:
    tmdb_api_key: str
    tmdb_language: str
    tmdb_start_date: str
    tmdb_end_date: str
    tmdb_movie_sample_size: int
    tmdb_tv_sample_size: int
    document_chunks_path: Path | None
    process_run_id: str
    retrieval_strategy: str
    retrieval_top_k: int
    retrieval_candidate_k: int
    enable_query_rewriting: bool
    observability_db_path: Path
    prompt_variant: str
    llm_temperature: float
    local_data_root: Path

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
        tmdb_start_date=os.getenv("TMDB_START_DATE", "2025-07-21").strip() or "2025-07-21",
        tmdb_end_date=os.getenv("TMDB_END_DATE", "2026-07-21").strip() or "2026-07-21",
        tmdb_movie_sample_size=int(os.getenv("TMDB_MOVIE_SAMPLE_SIZE", "300")),
        tmdb_tv_sample_size=int(os.getenv("TMDB_TV_SAMPLE_SIZE", "200")),
        document_chunks_path=(
            Path(value) if (value := _get_env("DOCUMENT_CHUNKS_PATH")) else None
        ),
        process_run_id=_get_env("PROCESS_RUN_ID"),
        retrieval_strategy=_get_env("RETRIEVAL_STRATEGY", "vector").lower() or "vector",
        retrieval_top_k=int(_get_env("RETRIEVAL_TOP_K", "5")),
        retrieval_candidate_k=int(_get_env("RETRIEVAL_CANDIDATE_K", "20")),
        enable_query_rewriting=_get_bool("ENABLE_QUERY_REWRITING", default=True),
        observability_db_path=Path(_get_env("OBSERVABILITY_DB_PATH", "data/monitoring/observability.db")),
        prompt_variant=_get_env("PROMPT_VARIANT", "strict").lower() or "strict",
        llm_temperature=float(_get_env("LLM_TEMPERATURE", "0.2")),
        local_data_root=Path(os.getenv("LOCAL_DATA_ROOT", "data")),
    )
