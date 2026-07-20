from __future__ import annotations

from pathlib import Path
from typing import Any

from cultural_mood_tracker.config import load_settings
from cultural_mood_tracker.core import resolve_path
from cultural_mood_tracker.transform.common import write_json


BASE_TABLE_NAMES = (
    "titles",
    "documents",
    "document_chunks",
    "ratings",
    "attention_signals",
)

OPTIONAL_TABLE_NAMES = (
    "people",
    "title_cast",
    "title_crew",
    "episodes",
    "title_videos",
    "chunk_annotations",
    "title_theme_summary",
    "genre_theme_summary",
    "monthly_theme_trends",
    "audience_vs_editorial_summary",
    "attention_vs_reception",
)

TABLE_CASTS: dict[str, dict[str, str]] = {
    "titles": {"release_date": "DATE"},
    "documents": {"published_at": "TIMESTAMPTZ"},
    "document_chunks": {"published_at": "TIMESTAMPTZ"},
    "ratings": {"published_at": "TIMESTAMPTZ"},
    "attention_signals": {"timestamp_utc": "TIMESTAMPTZ"},
    "episodes": {"airdate": "DATE", "airstamp": "TIMESTAMPTZ"},
    "title_videos": {"published_at": "TIMESTAMPTZ"},
}


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _build_create_table_sql(*, table_name: str, source_path: Path) -> str:
    cast_config = TABLE_CASTS.get(table_name, {})
    source_sql = _sql_path(source_path)
    if not cast_config:
        return f'''CREATE OR REPLACE TABLE "{table_name}" AS
                   SELECT * FROM read_json_auto('{source_sql}')'''
    replacements = ",\n".join(
        f"TRY_CAST({column} AS {target_type}) AS {column}"
        for column, target_type in cast_config.items()
    )
    return f'''CREATE OR REPLACE TABLE "{table_name}" AS
               WITH source AS (SELECT * FROM read_json_auto('{source_sql}'))
               SELECT * REPLACE ({replacements}) FROM source'''


def load_processed_tables(*, connection, processed_dir: Path) -> dict[str, list[dict[str, Any]]]:
    loaded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for table_name in (*BASE_TABLE_NAMES, *OPTIONAL_TABLE_NAMES):
        source_path = processed_dir / f"{table_name}.jsonl"
        if not source_path.exists() or source_path.stat().st_size == 0:
            skipped.append(
                {
                    "table_name": table_name,
                    "reason": "missing_file" if not source_path.exists() else "empty_file",
                }
            )
            continue
        safe_table = table_name.replace('"', '""')
        connection.execute(_build_create_table_sql(table_name=safe_table, source_path=source_path))
        row_count = connection.execute(f'SELECT COUNT(*) FROM "{safe_table}"').fetchone()[0]
        loaded.append(
            {"table_name": table_name, "row_count": row_count, "source_path": str(source_path)}
        )
    return {"loaded_tables": loaded, "skipped_tables": skipped}


def list_database_tables(*, connection) -> list[str]:
    return sorted(row[0] for row in connection.execute("SHOW TABLES").fetchall())


def preview_table_rows(*, connection, table_name: str, limit: int = 20) -> list[dict[str, Any]]:
    if limit <= 0:
        raise ValueError("Preview limit must be greater than zero")
    safe_table = table_name.replace('"', '""')
    cursor = connection.execute(f'SELECT * FROM "{safe_table}" LIMIT {int(limit)}')
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]


def run_local_duckdb_load(
    *, project_root: Path, process_run_id: str, database_path: Path | None = None
) -> dict[str, Any]:
    try:
        import duckdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Missing dependency duckdb. Install requirements first.") from exc

    settings = load_settings()
    paths = settings.build_paths(project_root)
    paths.ensure()
    processed_dir = paths.processed_root / process_run_id
    if not processed_dir.exists():
        raise RuntimeError(f"Missing processed directory: {processed_dir}")

    resolved_database = resolve_path(project_root, database_path or settings.local_duckdb_path)
    resolved_database.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(resolved_database))
    try:
        load_results = load_processed_tables(connection=connection, processed_dir=processed_dir)
    finally:
        connection.close()

    manifest = {
        "enabled": True,
        "backend": "local_duckdb",
        "process_run_id": process_run_id,
        "database_path": str(resolved_database),
        "loaded_table_count": len(load_results["loaded_tables"]),
        "skipped_table_count": len(load_results["skipped_tables"]),
        **load_results,
    }
    report_dir = paths.reports_root / process_run_id
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = report_dir / "local_duckdb_load_manifest.json"
    write_json(manifest_path, manifest)
    return {"manifest_path": str(manifest_path), **manifest}
